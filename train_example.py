"""
Example: Training a Unimodal LLM with Parallel-LLM
Demonstrates distributed training with FSDP and torch.compile
"""
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from datasets import load_dataset

from parallel_llm.core import DiffusionTransformer, ModelConfig
from parallel_llm.training import DistributedTrainer, TrainingConfig
from parallel_llm.training.losses import DiffusionLoss
from parallel_llm.utils.data import TextDataset


def setup_distributed():
    """Initialize distributed training environment"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


class DiffusionLLMTrainer(DistributedTrainer):
    """Custom trainer for diffusion LLM"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = DiffusionLoss(
            vocab_size=self.model_config.vocab_size,
            use_energy_model=self.model_config.use_energy_model,
        )

    def training_step(self, batch):
        """Single training step with diffusion loss"""
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        # Sample random diffusion timestep
        batch_size = input_ids.shape[0]
        timestep = torch.randint(
            0,
            self.model_config.num_diffusion_steps,
            (batch_size,),
            device=input_ids.device,
        )

        # Apply noise (masking) based on timestep
        masked_input, mask_positions = self._apply_mask(input_ids, timestep)

        # Forward pass
        logits, confidence = self.model(
            masked_input,
            timestep,
            attention_mask=attention_mask,
            return_confidence=True,
        )

        # Compute loss
        loss = self.criterion(
            logits=logits,
            targets=input_ids,
            mask_positions=mask_positions,
            confidence=confidence,
        )

        return loss

    def _apply_mask(self, input_ids, timestep):
        """Apply masking based on diffusion schedule"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Cosine masking schedule
        mask_ratio = torch.cos(
            (timestep.float() / self.model_config.num_diffusion_steps) * 3.14159 / 2
        )

        # Determine number of tokens to mask
        num_masked = (mask_ratio * seq_len).long()

        # Create mask
        mask_positions = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        for i in range(batch_size):
            indices = torch.randperm(seq_len, device=device)[:num_masked[i]]
            mask_positions[i, indices] = True

        # Apply mask
        masked_input = input_ids.clone()
        masked_input[mask_positions] = self.model.padding_idx

        return masked_input, mask_positions


def main():
    """Main training function"""
    # Setup distributed
    local_rank = setup_distributed()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if local_rank == 0:
        print("Loading dataset...")

    dataset = load_dataset("wikitext", "wikitext-103-v1")
    train_dataset = TextDataset(
        dataset["train"],
        tokenizer,
        max_length=1024,
    )

    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Configure model
    model_config = ModelConfig(
        vocab_size=len(tokenizer),
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,  # GQA
        intermediate_size=8192,
        max_position_embeddings=4096,
        num_diffusion_steps=10,
        use_energy_model=True,
        use_flash_attention=True,
        dtype=torch.bfloat16,
    )

    # Create model
    if local_rank == 0:
        print("Creating model...")

    model = DiffusionTransformer(model_config)

    # Configure training
    train_config = TrainingConfig(
        batch_size=8,
        learning_rate=3e-4,
        weight_decay=0.1,
        num_train_steps=100000,
        warmup_steps=2000,
        lr_scheduler="cosine",

        # Distributed settings
        use_fsdp=True,
        fsdp_sharding_strategy="full",
        fsdp_backward_prefetch=True,
        fsdp_forward_prefetch=True,

        # Mixed precision
        mixed_precision="bf16",

        # Optimization
        gradient_checkpointing=True,
        gradient_checkpointing_policy="selective",
        use_torch_compile=True,
        torch_compile_mode="max-autotune",

        # Logging
        logging_steps=10,
        eval_steps=1000,
        save_steps=5000,
        use_wandb=True,
        wandb_project="parallel-llm-training",

        output_dir="./checkpoints",
    )

    # Create trainer
    if local_rank == 0:
        print("Initializing trainer...")

    trainer = DiffusionLLMTrainer(
        model=model,
        train_config=train_config,
        model_config=model_config,
        train_dataloader=train_dataloader,
    )

    # Train!
    if local_rank == 0:
        print("Starting training...")

    trainer.train()

    if local_rank == 0:
        print("Training complete!")


if __name__ == "__main__":
    main()
