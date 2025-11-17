"""
Example: Fast Inference with Parallel Generation
Demonstrates one-shot token generation
"""
import torch
from transformers import AutoTokenizer

from parallel_llm.core import DiffusionTransformer, ModelConfig
from parallel_llm.inference import ParallelGenerator, GenerationConfig


def main():
    """Main inference function"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Configure model (should match training config)
    model_config = ModelConfig(
        vocab_size=len(tokenizer),
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        use_flash_attention=True,
    )

    # Create model
    model = DiffusionTransformer(model_config)

    # Load checkpoint
    checkpoint_path = "./checkpoints/checkpoint-100000/pytorch_model.bin"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # Move to GPU
    model = model.cuda()
    model.eval()

    # Compile for fast inference
    model = torch.compile(model, mode="reduce-overhead")

    # Configure generation
    gen_config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        num_refinement_steps=5,
        confidence_threshold=0.9,
        use_adaptive_steps=True,
    )

    # Create generator
    generator = ParallelGenerator(
        model=model,
        config=gen_config,
        use_kv_cache=True,
        use_cuda_graphs=True,
    )

    # Prepare prompt
    prompt = "Once upon a time, in a land far away,"
    prompt_tokens = tokenizer.encode(prompt, return_tensors="pt").cuda()

    print(f"Prompt: {prompt}")
    print(f"Generating {gen_config.max_new_tokens} tokens in parallel...")

    # Generate!
    with torch.no_grad():
        generated_tokens = generator.generate(prompt_tokens)

    # Decode
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    print(f"\nGenerated text:\n{generated_text}")

    # Benchmark speed
    import time

    num_runs = 10
    start = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            _ = generator.generate(prompt_tokens)

    end = time.time()
    avg_time = (end - start) / num_runs
    tokens_per_sec = gen_config.max_new_tokens / avg_time

    print(f"\nPerformance:")
    print(f"Average time: {avg_time:.3f}s")
    print(f"Tokens/sec: {tokens_per_sec:.1f}")
    print(f"Speedup vs autoregressive: ~{tokens_per_sec / 25:.1f}×")


if __name__ == "__main__":
    main()
