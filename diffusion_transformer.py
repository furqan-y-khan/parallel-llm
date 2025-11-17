"""
Diffusion Transformer for Parallel Token Generation
Implements masked diffusion with bidirectional attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False


class RotaryEmbedding(nn.Module):
    """RoPE positional embeddings"""
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to queries and keys"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ParallelAttention(nn.Module):
    """Multi-head attention with Flash Attention support"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings)
        self.use_flash = config.use_flash_attention and FLASH_AVAILABLE

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat k/v heads for grouped-query attention
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        if self.use_flash:
            # Use Flash Attention
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            attn_output = flash_attn_func(
                query_states, key_states, value_states,
                causal=False,  # Bidirectional for diffusion
                softmax_scale=1.0 / math.sqrt(self.head_dim)
            )
        else:
            # Standard attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class MLP(nn.Module):
    """Feed-forward network with SwiGLU activation"""
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class DiffusionTransformerBlock(nn.Module):
    """Single transformer block with diffusion conditioning"""
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ParallelAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

        # Diffusion time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        # Add time conditioning
        hidden_states = hidden_states + self.time_mlp(time_emb).unsqueeze(1)

        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DiffusionTransformer(nn.Module):
    """
    Transformer with masked diffusion for parallel generation
    Can generate all tokens simultaneously through iterative refinement
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size + 1, config.hidden_size, padding_idx=self.padding_idx)

        # Time embedding for diffusion steps
        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            DiffusionTransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output head (shared with embedding for weight tying)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Confidence predictor for adaptive refinement
        self.confidence_head = nn.Linear(config.hidden_size, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def get_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embeddings"""
        half_dim = self.config.hidden_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_embed(emb)

    def forward(
        self,
        input_ids: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_confidence: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for diffusion training or generation

        Args:
            input_ids: [batch, seq_len] - may contain MASK tokens
            timestep: [batch] - diffusion timestep
            attention_mask: [batch, seq_len] - attention mask
            return_confidence: whether to return confidence scores

        Returns:
            logits: [batch, seq_len, vocab_size]
            confidence: [batch, seq_len] if return_confidence else None
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Get time embedding
        time_emb = self.get_time_embedding(timestep)

        # Prepare attention mask for bidirectional attention
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, time_emb, attention_mask)

        hidden_states = self.norm(hidden_states)

        # Get logits
        logits = self.lm_head(hidden_states)

        # Get confidence scores if requested
        confidence = None
        if return_confidence:
            confidence = torch.sigmoid(self.confidence_head(hidden_states)).squeeze(-1)

        return logits, confidence

    @torch.no_grad()
    def generate_parallel(
        self,
        batch_size: int,
        seq_len: int,
        num_steps: int = 10,
        confidence_threshold: float = 0.9,
        temperature: float = 1.0,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Generate tokens in parallel using iterative refinement

        Args:
            batch_size: batch size
            seq_len: sequence length to generate
            num_steps: number of refinement steps
            confidence_threshold: threshold for accepting predictions
            temperature: sampling temperature
            device: device to generate on

        Returns:
            generated_ids: [batch, seq_len]
        """
        # Start with all MASK tokens
        generated_ids = torch.full(
            (batch_size, seq_len),
            self.padding_idx,
            dtype=torch.long,
            device=device
        )

        # Mask tracking which positions are still masked
        is_masked = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)

        # Iterative refinement
        for step in range(num_steps):
            # Current timestep (decreasing from num_steps to 0)
            timestep = torch.full((batch_size,), num_steps - step, device=device)

            # Forward pass
            logits, confidence = self.forward(
                generated_ids,
                timestep,
                return_confidence=True
            )

            # Sample from logits with temperature
            probs = F.softmax(logits / temperature, dim=-1)
            sampled_ids = torch.multinomial(
                probs.view(-1, self.config.vocab_size),
                num_samples=1
            ).view(batch_size, seq_len)

            # Determine which positions to keep based on confidence
            keep_mask = (confidence > confidence_threshold) & is_masked

            # Update generated tokens
            generated_ids = torch.where(keep_mask, sampled_ids, generated_ids)
            is_masked = is_masked & ~keep_mask

            # Early stopping if all tokens are decided
            if not is_masked.any():
                break

        # Replace any remaining MASK tokens with final predictions
        if is_masked.any():
            timestep = torch.zeros((batch_size,), device=device)
            logits, _ = self.forward(generated_ids, timestep)
            final_ids = torch.argmax(logits, dim=-1)
            generated_ids = torch.where(is_masked, final_ids, generated_ids)

        return generated_ids
