"""
Parallel Token Generator with vLLM-style optimizations
Implements one-shot generation through iterative refinement
"""
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import math


@dataclass
class GenerationConfig:
    """Configuration for parallel generation"""
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    num_refinement_steps: int = 5
    confidence_threshold: float = 0.9
    use_adaptive_steps: bool = True
    batch_size: int = 1


class PagedKVCache:
    """
    Paged KV cache for memory-efficient attention
    Inspired by vLLM's PagedAttention
    """
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_blocks: int = 2048,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.dtype = dtype
        self.device = device

        # Pre-allocate memory pool
        self.key_cache = torch.zeros(
            (num_layers, max_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.value_cache = torch.zeros(
            (num_layers, max_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        # Block allocation tracking
        self.free_blocks = list(range(max_blocks))
        self.allocated_blocks: Dict[int, List[int]] = {}  # sequence_id -> block_ids

    def allocate(self, sequence_id: int, num_tokens: int) -> List[int]:
        """Allocate blocks for a sequence"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError("Out of KV cache memory")

        blocks = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        self.allocated_blocks[sequence_id] = blocks
        return blocks

    def free(self, sequence_id: int):
        """Free blocks for a sequence"""
        if sequence_id in self.allocated_blocks:
            self.free_blocks.extend(self.allocated_blocks[sequence_id])
            del self.allocated_blocks[sequence_id]

    def get_kv(self, sequence_id: int, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get KV cache for a sequence at a specific layer"""
        blocks = self.allocated_blocks.get(sequence_id, [])
        if not blocks:
            return None, None

        # Gather from blocks
        keys = self.key_cache[layer_idx, blocks]
        values = self.value_cache[layer_idx, blocks]

        return keys, values

    def update_kv(
        self,
        sequence_id: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        token_pos: int,
    ):
        """Update KV cache for a specific position"""
        blocks = self.allocated_blocks[sequence_id]
        block_idx = token_pos // self.block_size
        offset = token_pos % self.block_size

        self.key_cache[layer_idx, blocks[block_idx], offset] = key
        self.value_cache[layer_idx, blocks[block_idx], offset] = value


class ParallelGenerator:
    """
    Ultra-fast parallel token generator
    Combines masked diffusion, speculative decoding, and continuous batching
    """
    def __init__(
        self,
        model: torch.nn.Module,
        config: GenerationConfig,
        use_kv_cache: bool = True,
        use_cuda_graphs: bool = True,
    ):
        self.model = model
        self.config = config
        self.use_kv_cache = use_kv_cache
        self.use_cuda_graphs = use_cuda_graphs

        # Setup KV cache if enabled
        if use_kv_cache:
            self.kv_cache = PagedKVCache(
                num_layers=model.config.num_hidden_layers,
                num_heads=model.config.num_attention_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
                device=next(model.parameters()).device,
            )

        # Compile model for faster inference
        if use_cuda_graphs:
            self._setup_cuda_graphs()

    def _setup_cuda_graphs(self):
        """Setup CUDA graphs for maximum throughput"""
        # Warmup to trigger compilation
        device = next(self.model.parameters()).device
        dummy_input = torch.randint(0, 100, (1, 64), device=device)
        dummy_timestep = torch.zeros(1, device=device)

        for _ in range(3):
            with torch.no_grad():
                _ = self.model(dummy_input, dummy_timestep)

        # Capture CUDA graph
        self.static_input = torch.zeros((self.config.batch_size, 512), dtype=torch.long, device=device)
        self.static_timestep = torch.zeros(self.config.batch_size, device=device)
        self.static_output = None

        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.graph):
            self.static_output, _ = self.model(self.static_input, self.static_timestep)

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens in parallel

        Args:
            prompt_tokens: [batch, prompt_len] input tokens
            max_new_tokens: maximum number of tokens to generate

        Returns:
            generated_tokens: [batch, prompt_len + max_new_tokens]
        """
        batch_size, prompt_len = prompt_tokens.shape
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        device = prompt_tokens.device

        # Initialize full sequence with MASK tokens
        total_len = prompt_len + max_new_tokens
        generated = torch.cat([
            prompt_tokens,
            torch.full(
                (batch_size, max_new_tokens),
                self.model.padding_idx,
                dtype=torch.long,
                device=device
            )
        ], dim=1)

        # Track which positions are still masked
        is_masked = torch.zeros((batch_size, total_len), dtype=torch.bool, device=device)
        is_masked[:, prompt_len:] = True

        # Iterative parallel refinement
        num_steps = self.config.num_refinement_steps

        for step in range(num_steps):
            # Adaptive step calculation
            if self.config.use_adaptive_steps:
                current_step = self._get_adaptive_timestep(is_masked, step, num_steps)
            else:
                current_step = num_steps - step

            timestep = torch.full((batch_size,), current_step, device=device, dtype=torch.long)

            # Forward pass - predict all positions simultaneously
            if self.use_cuda_graphs and generated.shape[1] <= self.static_input.shape[1]:
                # Use CUDA graph for fixed-size inputs
                self.static_input[:batch_size, :total_len].copy_(generated)
                self.static_timestep[:batch_size].copy_(timestep)
                self.graph.replay()
                logits = self.static_output[:batch_size, :total_len]
                confidence = None
            else:
                logits, confidence = self.model(generated, timestep, return_confidence=True)

            # Apply temperature
            logits = logits / self.config.temperature

            # Apply top-k filtering
            if self.config.top_k > 0:
                logits = self._top_k_filtering(logits, self.config.top_k)

            # Apply top-p (nucleus) filtering
            if self.config.top_p < 1.0:
                logits = self._top_p_filtering(logits, self.config.top_p)

            # Apply repetition penalty
            if self.config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated, self.config.repetition_penalty)

            # Sample tokens
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(
                probs.view(-1, self.model.config.vocab_size),
                num_samples=1
            ).view(batch_size, total_len)

            # Compute confidence if not returned by model
            if confidence is None:
                confidence = probs.max(dim=-1).values

            # Determine which positions to keep based on confidence
            keep_mask = (confidence > self.config.confidence_threshold) & is_masked

            # Update generated tokens
            generated = torch.where(keep_mask, next_tokens, generated)
            is_masked = is_masked & ~keep_mask

            # Early stopping if all tokens are decided
            if not is_masked.any():
                break

        # Final pass to fill any remaining masked positions
        if is_masked.any():
            timestep = torch.zeros(batch_size, device=device, dtype=torch.long)
            logits, _ = self.model(generated, timestep)
            final_tokens = torch.argmax(logits, dim=-1)
            generated = torch.where(is_masked, final_tokens, generated)

        return generated

    def _get_adaptive_timestep(
        self,
        is_masked: torch.Tensor,
        current_step: int,
        total_steps: int,
    ) -> int:
        """Adaptively determine timestep based on masking ratio"""
        mask_ratio = is_masked.float().mean()
        # Higher timestep for more masked tokens
        timestep = int(mask_ratio * total_steps)
        return max(1, timestep)

    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Filter logits to keep only top-k tokens"""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits

    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Filter logits using nucleus (top-p) sampling"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        return logits

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty to logits"""
        for i in range(generated.shape[0]):
            for token in set(generated[i].tolist()):
                if token != self.model.padding_idx:
                    # If score < 0 then repetition penalty has to be multiplied
                    # to reduce the score, otherwise it has to be divided
                    if logits[i, :, token] < 0:
                        logits[i, :, token] *= penalty
                    else:
                        logits[i, :, token] /= penalty
        return logits

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[torch.Tensor],
        max_new_tokens: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """
        Generate for multiple prompts with continuous batching
        Different prompts can have different lengths
        """
        if max_new_tokens is None:
            max_new_tokens = [self.config.max_new_tokens] * len(prompts)

        # Pad prompts to same length for batching
        max_prompt_len = max(p.shape[0] for p in prompts)
        batch_size = len(prompts)
        device = prompts[0].device

        padded_prompts = torch.full(
            (batch_size, max_prompt_len),
            self.model.padding_idx,
            dtype=torch.long,
            device=device,
        )

        for i, prompt in enumerate(prompts):
            padded_prompts[i, :len(prompt)] = prompt

        # Generate
        generated = self.generate(padded_prompts, max(max_new_tokens))

        # Unpad and return individual sequences
        results = []
        for i, prompt in enumerate(prompts):
            result = generated[i, :len(prompt) + max_new_tokens[i]]
            results.append(result)

        return results


def create_generator(
    model: torch.nn.Module,
    config: Optional[GenerationConfig] = None,
    use_kv_cache: bool = True,
    use_cuda_graphs: bool = True,
) -> ParallelGenerator:
    """Factory function to create parallel generator"""
    if config is None:
        config = GenerationConfig()

    return ParallelGenerator(
        model=model,
        config=config,
        use_kv_cache=use_kv_cache,
        use_cuda_graphs=use_cuda_graphs,
    )
