"""
Configuration system for Parallel-LLM
Supports unimodal and multimodal models with full parallelism
"""
from dataclasses import dataclass, field
from typing import Optional, List, Literal
import torch


@dataclass
class ModelConfig:
    """Base model configuration"""
    # Architecture
    vocab_size: int = 50257
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None  # For GQA
    intermediate_size: int = 8192
    max_position_embeddings: int = 4096

    # Diffusion parameters
    num_diffusion_steps: int = 10
    noise_schedule: Literal["linear", "cosine", "sqrt"] = "cosine"
    self_condition: bool = True
    confidence_threshold: float = 0.9

    # Energy-based model
    use_energy_model: bool = True
    energy_hidden_size: int = 4096
    energy_num_layers: int = 4

    # Attention
    use_flash_attention: bool = True
    use_sliding_window: bool = False
    sliding_window_size: Optional[int] = None

    # Normalization and activations
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"

    # Dropout
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Precision
    dtype: torch.dtype = torch.bfloat16
    use_fp8: bool = False

    # Initialization
    initializer_range: float = 0.02


@dataclass
class MultimodalConfig(ModelConfig):
    """Configuration for multimodal models"""
    # Vision encoder
    vision_encoder: Literal["vit", "clip", "siglip"] = "vit"
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    vision_hidden_size: int = 1024
    vision_num_layers: int = 24
    vision_num_heads: int = 16

    # Fusion
    fusion_type: Literal["cross_attention", "perceiver", "moe"] = "cross_attention"
    num_cross_attention_layers: int = 4

    # Contrastive learning
    use_contrastive: bool = True
    contrastive_temperature: float = 0.07
    contrastive_dim: int = 512


@dataclass
class TrainingConfig:
    """Training configuration with distributed support"""
    # Basic training
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Schedule
    num_train_steps: int = 100000
    warmup_steps: int = 2000
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"

    # Mixed precision
    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = "bf16"

    # Distributed training
    distributed_backend: Literal["nccl", "gloo"] = "nccl"
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # FSDP
    use_fsdp: bool = True
    fsdp_sharding_strategy: Literal["full", "shard_grad_op", "no_shard"] = "full"
    fsdp_backward_prefetch: bool = True
    fsdp_forward_prefetch: bool = True

    # DeepSpeed ZeRO
    use_deepspeed: bool = False
    zero_stage: Literal[0, 1, 2, 3] = 3
    zero_offload_optimizer: bool = False
    zero_offload_params: bool = False

    # Gradient checkpointing
    gradient_checkpointing: bool = True
    gradient_checkpointing_policy: Literal["full", "selective"] = "selective"

    # Compilation
    use_torch_compile: bool = True
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "max-autotune"
    use_cuda_graphs: bool = True

    # Logging
    logging_steps: int = 10
    eval_steps: int = 1000
    save_steps: int = 5000
    save_total_limit: int = 3

    # Monitoring
    use_wandb: bool = True
    wandb_project: str = "parallel-llm"

    # Checkpointing
    output_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None


@dataclass
class InferenceConfig:
    """Inference configuration for parallel generation"""
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0

    # Parallel generation
    num_parallel_tokens: int = 64  # Generate this many at once
    num_refinement_steps: int = 5
    confidence_threshold: float = 0.9
    use_adaptive_refinement: bool = True

    # Batching
    batch_size: int = 1
    use_continuous_batching: bool = True
    max_batch_size: int = 128

    # KV cache
    use_paged_attention: bool = True
    block_size: int = 16
    max_num_blocks: int = 2048

    # Speculative decoding
    use_speculative_decoding: bool = False
    draft_model_path: Optional[str] = None
    num_speculative_tokens: int = 5

    # Quantization
    quantization: Optional[Literal["int8", "fp8", "int4"]] = None

    # Parallelism
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # Performance
    use_torch_compile: bool = True
    use_cuda_graphs: bool = True


def get_default_config(model_type: Literal["unimodal", "multimodal"] = "unimodal"):
    """Get default configuration for model type"""
    if model_type == "unimodal":
        return ModelConfig()
    elif model_type == "multimodal":
        return MultimodalConfig()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
