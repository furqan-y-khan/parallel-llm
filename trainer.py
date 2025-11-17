"""
Distributed Trainer with FSDP2, DeepSpeed ZeRO, and torch.compile support
"""
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
import functools
from typing import Optional, Dict, Any
from pathlib import Path
import logging
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """
    Production-ready distributed trainer with all optimizations
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_config,
        model_config,
        train_dataloader,
        eval_dataloader=None,
    ):
        self.model = model
        self.train_config = train_config
        self.model_config = model_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Initialize distributed
        self._init_distributed()

        # Setup model for distributed training
        self._setup_distributed_model()

        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()

        # Setup mixed precision
        self._setup_mixed_precision()

        # Compile model if requested
        if train_config.use_torch_compile:
            self._compile_model()

        # Setup logging
        self._setup_logging()

        self.global_step = 0
        self.epoch = 0

    def _init_distributed(self):
        """Initialize distributed training"""
        if not dist.is_initialized():
            dist.init_process_group(backend=self.train_config.distributed_backend)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")

        if self.rank == 0:
            logger.info(f"Initialized distributed training with {self.world_size} GPUs")

    def _setup_distributed_model(self):
        """Setup model with FSDP or DeepSpeed"""
        self.model = self.model.to(self.device)

        if self.train_config.use_deepspeed:
            self._setup_deepspeed()
        elif self.train_config.use_fsdp:
            self._setup_fsdp()
        else:
            # Standard DDP
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

    def _setup_fsdp(self):
        """Setup Fully Sharded Data Parallel (FSDP2)"""
        # Define auto wrap policy for transformer blocks
        from parallel_llm.core.diffusion_transformer import DiffusionTransformerBlock

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={DiffusionTransformerBlock},
        )

        # Mixed precision configuration
        if self.train_config.mixed_precision == "bf16":
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif self.train_config.mixed_precision == "fp16":
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:
            mp_policy = None

        # Sharding strategy
        sharding_strategy_map = {
            "full": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }
        sharding_strategy = sharding_strategy_map[self.train_config.fsdp_sharding_strategy]

        # Wrap model with FSDP
        self.model = FSDP(
            self.model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=sharding_strategy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE if self.train_config.fsdp_backward_prefetch else None,
            forward_prefetch=self.train_config.fsdp_forward_prefetch,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
        )

        # Apply activation checkpointing
        if self.train_config.gradient_checkpointing:
            check_fn = lambda submodule: isinstance(submodule, DiffusionTransformerBlock)
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            apply_activation_checkpointing(
                self.model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=check_fn,
            )

        if self.rank == 0:
            logger.info(f"FSDP initialized with {sharding_strategy}")

    def _setup_deepspeed(self):
        """Setup DeepSpeed ZeRO"""
        import deepspeed

        ds_config = {
            "train_batch_size": self.train_config.batch_size * self.world_size,
            "train_micro_batch_size_per_gpu": self.train_config.batch_size,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.train_config.learning_rate,
                    "betas": [self.train_config.adam_beta1, self.train_config.adam_beta2],
                    "eps": self.train_config.adam_epsilon,
                    "weight_decay": self.train_config.weight_decay,
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "total_num_steps": self.train_config.num_train_steps,
                    "warmup_num_steps": self.train_config.warmup_steps,
                }
            },
            "fp16": {"enabled": self.train_config.mixed_precision == "fp16"},
            "bf16": {"enabled": self.train_config.mixed_precision == "bf16"},
            "zero_optimization": {
                "stage": self.train_config.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.train_config.zero_offload_optimizer else "none",
                    "pin_memory": True,
                },
                "offload_param": {
                    "device": "cpu" if self.train_config.zero_offload_params else "none",
                    "pin_memory": True,
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
            },
            "gradient_clipping": self.train_config.max_grad_norm,
            "wall_clock_breakdown": False,
        }

        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            config=ds_config,
        )

        if self.rank == 0:
            logger.info(f"DeepSpeed ZeRO-{self.train_config.zero_stage} initialized")

    def _setup_optimizer(self):
        """Setup AdamW optimizer"""
        if self.train_config.use_deepspeed:
            return  # DeepSpeed handles optimizer

        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "norm" in name or "embed" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.train_config.learning_rate,
            betas=(self.train_config.adam_beta1, self.train_config.adam_beta2),
            eps=self.train_config.adam_epsilon,
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.train_config.use_deepspeed:
            return  # DeepSpeed handles scheduler

        if self.train_config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_config.num_train_steps - self.train_config.warmup_steps,
            )
        elif self.train_config.lr_scheduler == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.train_config.num_train_steps - self.train_config.warmup_steps,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

        # Add warmup
        if self.train_config.warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.0,
                end_factor=1.0,
                total_iters=self.train_config.warmup_steps,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, self.scheduler],
                milestones=[self.train_config.warmup_steps],
            )

    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        if self.train_config.use_deepspeed or self.train_config.use_fsdp:
            self.scaler = None  # Handled by FSDP/DeepSpeed
        elif self.train_config.mixed_precision == "fp16":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def _compile_model(self):
        """Compile model with torch.compile"""
        if not self.train_config.use_deepspeed:
            self.model = torch.compile(
                self.model,
                mode=self.train_config.torch_compile_mode,
                fullgraph=False,
                dynamic=True,
            )
            if self.rank == 0:
                logger.info(f"Model compiled with mode={self.train_config.torch_compile_mode}")

    def _setup_logging(self):
        """Setup logging and monitoring"""
        if self.rank == 0 and self.train_config.use_wandb:
            wandb.init(
                project=self.train_config.wandb_project,
                config={
                    "model": self.model_config.__dict__,
                    "training": self.train_config.__dict__,
                },
            )

    def train(self):
        """Main training loop"""
        self.model.train()

        if self.rank == 0:
            progress_bar = tqdm(total=self.train_config.num_train_steps, desc="Training")

        while self.global_step < self.train_config.num_train_steps:
            for batch in self.train_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}

                # Forward pass
                loss = self.training_step(batch)

                # Backward pass
                if self.train_config.use_deepspeed:
                    self.model.backward(loss)
                    self.model.step()
                else:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.train_config.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.train_config.max_grad_norm
                        )
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.scheduler.step()

                self.global_step += 1

                # Logging
                if self.global_step % self.train_config.logging_steps == 0:
                    self._log_metrics({"train/loss": loss.item()})

                # Evaluation
                if self.global_step % self.train_config.eval_steps == 0:
                    self.evaluate()

                # Checkpointing
                if self.global_step % self.train_config.save_steps == 0:
                    self.save_checkpoint()

                if self.rank == 0:
                    progress_bar.update(1)

                if self.global_step >= self.train_config.num_train_steps:
                    break

        if self.rank == 0:
            progress_bar.close()

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step - to be implemented by specific model"""
        raise NotImplementedError("Implement training_step in subclass")

    def evaluate(self):
        """Evaluation loop"""
        if self.eval_dataloader is None:
            return

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                loss = self.training_step(batch)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        self._log_metrics({"eval/loss": avg_loss})

        self.model.train()

    def save_checkpoint(self):
        """Save model checkpoint"""
        if self.rank == 0:
            checkpoint_dir = Path(self.train_config.output_dir) / f"checkpoint-{self.global_step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save model state
            if self.train_config.use_fsdp:
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType

                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                    state_dict = self.model.state_dict()
                    torch.save(state_dict, checkpoint_dir / "pytorch_model.bin")
            else:
                torch.save(self.model.state_dict(), checkpoint_dir / "pytorch_model.bin")

            # Save optimizer and scheduler
            torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
            torch.save(self.scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

            # Save training state
            torch.save({
                "global_step": self.global_step,
                "epoch": self.epoch,
            }, checkpoint_dir / "training_state.pt")

            logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to wandb"""
        if self.rank == 0:
            metrics["step"] = self.global_step
            if self.train_config.use_wandb:
                wandb.log(metrics)
            else:
                logger.info(f"Step {self.global_step}: {metrics}")
