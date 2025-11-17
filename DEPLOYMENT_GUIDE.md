# Deployment & Optimization Guide

## Production Deployment

### Hardware Requirements

#### Minimum Requirements
- **GPU**: NVIDIA A100 40GB or equivalent
- **CPU**: 16+ cores
- **RAM**: 128GB+
- **Storage**: 500GB SSD
- **Network**: 100Gbps InfiniBand for multi-node

#### Recommended for Best Performance
- **GPU**: NVIDIA H100 80GB
- **CPU**: 64+ cores (AMD EPYC or Intel Xeon)
- **RAM**: 512GB+
- **Storage**: 1TB+ NVMe SSD
- **Network**: 400Gbps InfiniBand

### Installation Steps

#### 1. System Setup

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install CUDA 12.1+
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Install NCCL for multi-GPU
sudo apt-get install libnccl2 libnccl-dev
```

#### 2. Python Environment

```bash
# Create virtual environment
python3.10 -m venv parallel-llm-env
source parallel-llm-env/bin/activate

# Install PyTorch with CUDA support
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Install Parallel-LLM
pip install parallel-llm

# Or from source for latest features
git clone https://github.com/furqan-y-khan/parallel-llm
cd parallel-llm
pip install -e ".[dev]"
```

#### 3. Verify Installation

```python
import torch
import parallel_llm

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"Parallel-LLM version: {parallel_llm.__version__}")
```

## Training Optimization

### Single Node (8× A100)

```python
# Optimal configuration for 7B model
train_config = TrainingConfig(
    batch_size=16,  # Per GPU
    learning_rate=3e-4,

    # FSDP with full sharding
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
    use_cuda_graphs=True,

    # Performance tuning
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
)

# Launch command
# torchrun --nproc_per_node=8 train.py
```

**Expected Performance**: ~2.5M tokens/sec

### Multi-Node (4 nodes × 8 GPUs = 32 GPUs)

```python
# Optimal configuration for 70B model
train_config = TrainingConfig(
    batch_size=8,  # Per GPU

    # Hybrid parallelism
    use_fsdp=True,
    fsdp_sharding_strategy="full",
    tensor_parallel_size=4,  # TP within node
    pipeline_parallel_size=1,

    # DeepSpeed ZeRO-3 for massive models
    use_deepspeed=True,
    zero_stage=3,
    zero_offload_optimizer=False,  # Keep on GPU for speed
    zero_offload_params=False,

    mixed_precision="bf16",
    use_torch_compile=True,
    torch_compile_mode="max-autotune",
)

# Launch command
# torchrun \
#     --nnodes=4 \
#     --nproc_per_node=8 \
#     --master_addr=node0 \
#     --master_port=29500 \
#     train.py
```

**Expected Performance**: ~15M tokens/sec

### Memory Optimization

```python
# For models that don't fit in GPU memory

# Option 1: FSDP with CPU offload
train_config.fsdp_offload_params = True  # Slower but fits larger models

# Option 2: DeepSpeed ZeRO-3 with offload
train_config.use_deepspeed = True
train_config.zero_stage = 3
train_config.zero_offload_optimizer = True
train_config.zero_offload_params = True

# Option 3: Gradient checkpointing
train_config.gradient_checkpointing = True
train_config.gradient_checkpointing_policy = "full"  # More aggressive

# Option 4: Reduce batch size
train_config.batch_size = 2
train_config.gradient_accumulation_steps = 8  # Maintain effective batch size
```

## Inference Optimization

### Maximum Throughput

```python
gen_config = GenerationConfig(
    max_new_tokens=512,
    temperature=1.0,

    # Aggressive parallel generation
    num_refinement_steps=3,  # Lower for speed
    confidence_threshold=0.85,  # Lower for speed
    use_adaptive_steps=True,

    # Batching
    batch_size=32,  # Large batches
    use_continuous_batching=True,
)

generator = ParallelGenerator(
    model=model,
    config=gen_config,
    use_kv_cache=True,
    use_cuda_graphs=True,  # Essential for throughput
)

# Compile model
model = torch.compile(model, mode="reduce-overhead")
```

**Expected Performance**: ~200 tokens/sec per request (batched)

### Minimum Latency

```python
gen_config = GenerationConfig(
    max_new_tokens=128,  # Shorter sequences

    # Quality-focused parallel generation
    num_refinement_steps=5,
    confidence_threshold=0.95,
    use_adaptive_steps=True,

    batch_size=1,  # Single request
)

# Use smaller model or quantization
from parallel_llm.quantization import quantize_model
model = quantize_model(model, precision="fp8")

generator = ParallelGenerator(model, gen_config, use_cuda_graphs=True)
```

**Expected Performance**: 50-100ms latency

### Serving Architecture

```python
# Production serving with vLLM-style continuous batching
from parallel_llm.serving import InferenceServer

server = InferenceServer(
    model=model,
    model_config=model_config,

    # Batching
    max_batch_size=128,
    batch_timeout_ms=10,

    # Memory management
    use_paged_kv_cache=True,
    kv_cache_dtype="fp8",
    max_num_seqs=256,

    # Performance
    enable_cuda_graphs=True,
    enable_prefix_caching=True,

    # Load balancing
    num_replicas=4,  # Model replicas
    distributed_executor="ray",
)

# Start server
server.start(host="0.0.0.0", port=8000)
```

## Monitoring & Debugging

### Performance Monitoring

```python
# Enable profiling
train_config.profile_training = True
train_config.profile_steps = 10

# Monitor GPU utilization
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
info = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
print(f"GPU Utilization: {info.gpu}%")

# Monitor memory
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print(f"Memory Used: {info.used / 1e9:.2f} GB")
```

### Debugging Tools

```bash
# Check NCCL communication
export NCCL_DEBUG=INFO

# Profile with PyTorch profiler
python -m torch.utils.bottleneck train.py

# NSight Systems profiling
nsys profile --trace=cuda,nvtx python train.py

# Memory profiling
python -m torch.utils.bottleneck --input-shape="[8,512]" train.py
```

### Common Issues

#### Issue: OOM during training
```python
# Solution 1: Enable gradient checkpointing
train_config.gradient_checkpointing = True

# Solution 2: Reduce batch size
train_config.batch_size = train_config.batch_size // 2
train_config.gradient_accumulation_steps *= 2

# Solution 3: Use FSDP
train_config.use_fsdp = True
train_config.fsdp_sharding_strategy = "full"
```

#### Issue: Slow compilation
```python
# Solution: Reduce compile scope
train_config.torch_compile_mode = "default"  # Instead of max-autotune

# Or disable dynamic shapes
model = torch.compile(model, dynamic=False)
```

#### Issue: Poor generation quality
```python
# Solution: Increase refinement steps
gen_config.num_refinement_steps = 10
gen_config.confidence_threshold = 0.95
```

## Benchmarking

### Training Benchmark

```bash
# Benchmark training throughput
python -m parallel_llm.benchmark.train \
    --model-size 7b \
    --batch-size 16 \
    --num-gpus 8 \
    --use-fsdp \
    --mixed-precision bf16
```

### Inference Benchmark

```bash
# Benchmark inference throughput
python -m parallel_llm.benchmark.infer \
    --model-size 7b \
    --batch-size 32 \
    --seq-len 512 \
    --use-cuda-graphs \
    --use-kv-cache
```

## Cost Optimization

### Cloud Deployment

**AWS**
- Training: p4d.24xlarge (8× A100 40GB) - $32.77/hr
- Inference: p4de.24xlarge (8× A100 80GB) - $40.96/hr
- Best for: Large-scale training

**GCP**
- Training: a2-ultragpu-8g (8× A100 80GB) - $32/hr
- Inference: a2-highgpu-4g (4× A100 40GB) - $16/hr
- Best for: Flexible workloads

**Azure**
- Training: ND96asr_v4 (8× A100 40GB) - $27.20/hr
- Inference: ND96amsr_A100_v4 (8× A100 80GB) - $32.77/hr
- Best for: Enterprise integration

### Cost Savings

```python
# Use mixed precision to reduce memory → smaller instances
train_config.mixed_precision = "bf16"  # 2× memory reduction

# Use gradient checkpointing
train_config.gradient_checkpointing = True  # ~40% memory reduction

# Use FSDP for efficient multi-GPU
train_config.use_fsdp = True  # Better scaling than DDP

# Quantize for inference
model = quantize_model(model, "fp8")  # 2× speedup, 2× memory reduction
```

**Estimated Savings**: 40-60% on inference costs

## Security & Compliance

### Model Security

```python
# Encrypt model checkpoints
from parallel_llm.security import encrypt_checkpoint

encrypt_checkpoint(
    checkpoint_path="./checkpoints/model.bin",
    output_path="./encrypted/model.enc",
    key=os.environ["ENCRYPTION_KEY"],
)

# Load encrypted checkpoint
model = load_encrypted_checkpoint(
    path="./encrypted/model.enc",
    key=os.environ["ENCRYPTION_KEY"],
)
```

### Access Control

```python
# API authentication
from parallel_llm.serving import InferenceServer

server = InferenceServer(
    model=model,
    auth_enabled=True,
    api_keys=["key1", "key2"],
    rate_limit_rpm=1000,
)
```

### Audit Logging

```python
# Enable comprehensive logging
train_config.enable_audit_logging = True
train_config.log_gradients = True
train_config.log_activations = False  # Expensive, use sparingly
```

## Support

For production support:
- **GitHub**: https://github.com/furqan-y-khan/parallel-llm
- **Email**: furqan@lastappstanding.com
