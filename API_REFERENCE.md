# API Reference

## Core Modules

### parallel_llm.core

#### DiffusionTransformer

```python
class DiffusionTransformer(nn.Module):
    """
    Transformer with masked diffusion for parallel generation

    Args:
        config (ModelConfig): Model configuration

    Methods:
        forward(input_ids, timestep, attention_mask=None, return_confidence=False):
            Forward pass for training or generation

        generate_parallel(batch_size, seq_len, num_steps=10, confidence_threshold=0.9):
            Generate tokens in parallel using iterative refinement
    """
```

#### ModelConfig

```python
@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 50257
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None  # For GQA
    intermediate_size: int = 8192
    max_position_embeddings: int = 4096

    # Diffusion
    num_diffusion_steps: int = 10
    noise_schedule: Literal["linear", "cosine", "sqrt"] = "cosine"
    confidence_threshold: float = 0.9

    # Attention
    use_flash_attention: bool = True

    # Precision
    dtype: torch.dtype = torch.bfloat16
    use_fp8: bool = False
```

### parallel_llm.training

#### DistributedTrainer

```python
class DistributedTrainer:
    """
    Production-ready distributed trainer

    Supports:
    - FSDP2 with full/gradient/no sharding
    - DeepSpeed ZeRO stages 1/2/3
    - Mixed precision (FP16/BF16/FP8)
    - Gradient checkpointing
    - torch.compile
    - CUDA graphs

    Args:
        model: torch.nn.Module
        train_config: TrainingConfig
        model_config: ModelConfig
        train_dataloader: DataLoader
        eval_dataloader: Optional[DataLoader]

    Methods:
        train(): Main training loop
        evaluate(): Evaluation loop
        save_checkpoint(): Save model checkpoint
        training_step(batch): Single training step (to be implemented by subclass)
    """
```

#### TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Basic
    batch_size: int = 8
    learning_rate: float = 3e-4
    num_train_steps: int = 100000

    # Distributed
    use_fsdp: bool = True
    fsdp_sharding_strategy: Literal["full", "shard_grad_op", "no_shard"] = "full"
    use_deepspeed: bool = False
    zero_stage: Literal[0, 1, 2, 3] = 3

    # Optimization
    use_torch_compile: bool = True
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "max-autotune"
    use_cuda_graphs: bool = True
    gradient_checkpointing: bool = True

    # Mixed precision
    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = "bf16"
```

### parallel_llm.inference

#### ParallelGenerator

```python
class ParallelGenerator:
    """
    Ultra-fast parallel token generator

    Features:
    - One-shot generation (all tokens at once)
    - Paged KV cache
    - CUDA graphs
    - Continuous batching
    - Speculative decoding

    Args:
        model: DiffusionTransformer
        config: GenerationConfig
        use_kv_cache: bool = True
        use_cuda_graphs: bool = True

    Methods:
        generate(prompt_tokens, max_new_tokens=None):
            Generate tokens in parallel

        generate_batch(prompts, max_new_tokens=None):
            Generate for multiple prompts with continuous batching
    """
```

#### GenerationConfig

```python
@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    num_refinement_steps: int = 5
    confidence_threshold: float = 0.9
    use_adaptive_steps: bool = True
```

## Usage Examples

### Basic Training

```python
from parallel_llm import DiffusionTransformer, ModelConfig, TrainingConfig, DistributedTrainer

# Create model
model_config = ModelConfig(hidden_size=2048, num_hidden_layers=24)
model = DiffusionTransformer(model_config)

# Configure training
train_config = TrainingConfig(
    batch_size=8,
    use_fsdp=True,
    mixed_precision="bf16",
)

# Train
trainer = DistributedTrainer(model, train_config, model_config, train_dataloader)
trainer.train()
```

### Parallel Generation

```python
from parallel_llm import ParallelGenerator, GenerationConfig

# Configure generation
gen_config = GenerationConfig(
    max_new_tokens=512,
    num_refinement_steps=5,
)

# Create generator
generator = ParallelGenerator(model, gen_config, use_cuda_graphs=True)

# Generate
output = generator.generate(prompt_tokens)
```

### Multimodal Training

```python
from parallel_llm.multimodal import MultimodalModel, MultimodalConfig

# Configure model
config = MultimodalConfig(
    vision_encoder="clip",
    fusion_type="cross_attention",
    use_contrastive=True,
)

# Create model
model = MultimodalModel(config)

# Train with image-text pairs
trainer = MultimodalTrainer(model, train_config, config, dataloader)
trainer.train()
```

## Advanced Features

### Custom Kernels

```python
from parallel_llm.kernels import flash_attention, parallel_decode

# Use Flash Attention
output = flash_attention(query, key, value)

# Parallel token decoding
tokens = parallel_decode(logits, num_parallel=64)
```

### Quantization

```python
from parallel_llm.quantization import quantize_model

# Quantize to FP8
model_fp8 = quantize_model(model, precision="fp8")
```

### Distributed Launch

```bash
# Single node, 8 GPUs
torchrun --nproc_per_node=8 train.py

# Multi-node, 4 nodes × 8 GPUs
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    --master_addr=node0 \
    --master_port=29500 \
    train.py
```

## Performance Tips

### Training

1. **Use FSDP2** for models > 1B parameters
2. **Enable torch.compile** with `mode="max-autotune"`
3. **Use BF16** on Ampere+ GPUs, FP8 on Hopper
4. **Enable gradient checkpointing** for large models
5. **Use Flash Attention 3** for best performance

### Inference

1. **Use CUDA graphs** for batched inference
2. **Enable paged KV cache** for long sequences
3. **Tune num_refinement_steps** (lower = faster, higher = better quality)
4. **Use adaptive steps** for dynamic speed/quality tradeoff
5. **Compile model** with `mode="reduce-overhead"`

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
train_config.batch_size = 4

# Enable gradient checkpointing
train_config.gradient_checkpointing = True

# Use FSDP with full sharding
train_config.fsdp_sharding_strategy = "full"

# Offload to CPU (DeepSpeed)
train_config.zero_offload_optimizer = True
```

### Slow Training

```python
# Enable torch.compile
train_config.use_torch_compile = True
train_config.torch_compile_mode = "max-autotune"

# Use Flash Attention
model_config.use_flash_attention = True

# Enable prefetching
train_config.fsdp_backward_prefetch = True
train_config.fsdp_forward_prefetch = True
```

### Low Quality Generation

```python
# Increase refinement steps
gen_config.num_refinement_steps = 10

# Raise confidence threshold
gen_config.confidence_threshold = 0.95

# Enable adaptive steps
gen_config.use_adaptive_steps = True

# Adjust temperature
gen_config.temperature = 0.8
```
