"""
Jina Embeddings v5 Text Nano - MLX Implementation

Pure MLX port of jina-embeddings-v5-text-nano (EuroBERT-210m backbone).
Zero dependency on PyTorch or transformers.

Features:
- Bidirectional encoder (no causal mask)
- Last-token pooling
- L2 normalization
- Max sequence length: 8192 tokens
- Embedding dimension: 768

Architecture:
- RoPE (rope_theta=1000000)
- SwiGLU MLP
- RMSNorm (eps=1e-05)
- No q_norm/k_norm (unlike Qwen3)
- No attention bias
"""

from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, BaseModelOutput


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    head_dim: int
    tie_word_embeddings: bool
    rope_theta: float = 1000000.0
    hidden_act: str = "silu"
    attention_bias: bool = False
    architectures: List[str] = field(default_factory=lambda: ["EuroBertModel"])


def initialize_rope(head_dim: int, base: float = 250000.0):
    """Create an nn.RoPE instance for EuroBERT.

    Uses traditional=False (rotate_half convention) to match PyTorch's implementation.
    nn.RoPE internally uses mx.fast.rope for optimized computation.
    """
    return nn.RoPE(head_dim, traditional=False, base=base)


class EuroBertAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.head_dim
        self.scale = head_dim**-0.5
        self.head_dim = head_dim

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        # EuroBERT does NOT have q_norm/k_norm
        # Use nn.RoPE which internally calls mx.fast.rope
        self.rotary_emb = initialize_rope(head_dim, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Reshape for multi-head attention (no q_norm/k_norm)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE via mx.fast.rope (through nn.RoPE)
        queries = self.rotary_emb(queries)
        keys = self.rotary_emb(keys)

        # Handle GQA (Grouped Query Attention) if needed
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            keys = mx.repeat(keys, n_rep, axis=1)
            values = mx.repeat(values, n_rep, axis=1)

        # Use mx.fast.scaled_dot_product_attention for better precision
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class EuroBertMLP(nn.Module):
    def __init__(self, dim, hidden_dim, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)

    def __call__(self, x) -> mx.array:
        # SwiGLU activation
        gate = nn.silu(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))


class EuroBertTransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = EuroBertAttention(args)
        self.mlp = EuroBertMLP(
            args.hidden_size, args.intermediate_size, bias=args.attention_bias
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        # Pre-norm architecture
        r = self.self_attn(self.input_layernorm(x), mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class EuroBertModel(nn.Module):
    """EuroBERT encoder model (bidirectional, no causal mask)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            EuroBertTransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, mask: Optional[mx.array] = None):
        h = self.embed_tokens(inputs)

        for layer in self.layers:
            h = layer(h, mask)
        return self.norm(h)


class Model(nn.Module):
    """Jina v5-text-nano embedding model with last-token pooling."""

    def __init__(self, config: dict):
        super().__init__()
        self.model = EuroBertModel(config)
        self.config = config

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ):
        """
        Forward pass with last-token pooling and L2 normalization.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Normalized embeddings [batch, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        # Bidirectional encoder - only use padding mask, NO causal mask
        if attention_mask is not None:
            # attention_mask is [batch, seq_len] with 1 for real tokens, 0 for padding
            # Convert to additive mask: 0 for attending, large negative for not attending
            padding_mask = mx.where(attention_mask == 0, -1e9, 0.0)
            # Expand to [batch, 1, 1, seq_len] for broadcasting
            mask = mx.expand_dims(mx.expand_dims(padding_mask, 1), 1)
            # Match dtype of model weights (important for float16 inference)
            h_dtype = self.model.embed_tokens.weight.dtype
            if mask.dtype != h_dtype:
                mask = mask.astype(h_dtype)
        else:
            mask = None

        # Get hidden states
        hidden_states = self.model(input_ids, mask)  # [batch, seq_len, hidden_size]

        # Last token pooling
        if attention_mask is not None:
            # Find the last non-padding token for each sequence
            sequence_lengths = mx.sum(attention_mask, axis=1) - 1  # [batch]
            batch_indices = mx.arange(hidden_states.shape[0])
            embeddings = hidden_states[batch_indices, sequence_lengths]
        else:
            # Take the last token
            embeddings = hidden_states[:, -1, :]

        # L2 normalization
        norms = mx.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        pooled_output = None

        # return embeddings
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            text_embeds=embeddings,
            pooler_output=pooled_output,
            hidden_states=hidden_states[1:],
        )

    def encode(
        self,
        texts: list[str],
        tokenizer,
        max_length: int = 8192,
        truncate_dim: Optional[int] = None,
        task_type: str = "retrieval.query",
    ):
        """
        Encode texts to embeddings.

        Args:
            texts: List of input texts
            tokenizer: Tokenizer instance (from tokenizers library)
            max_length: Maximum sequence length
            truncate_dim: Optional Matryoshka dimension (EuroBERT supports 768 max)
            task_type: Task prefix ("retrieval.query", "retrieval.passage", etc.)

        Returns:
            Embeddings array [batch, dim]
        """
        # Add task prefix
        prefix_map = {
            "retrieval.query": "Query: ",
            "retrieval.passage": "Document: ",
            "classification": "Document: ",
            "text-matching": "Document: ",
            "clustering": "Document: ",
        }
        prefix = prefix_map.get(task_type, "")

        if prefix:
            texts = [prefix + text for text in texts]

        # Tokenize
        encodings = tokenizer.encode_batch(texts)

        # Prepare inputs
        max_len = min(max_length, max(len(enc.ids) for enc in encodings))
        input_ids = []
        attention_mask = []

        for encoding in encodings:
            ids = encoding.ids[:max_len]
            mask = encoding.attention_mask[:max_len]

            # Pad if needed
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = ids + [0] * pad_len
                mask = mask + [0] * pad_len

            input_ids.append(ids)
            attention_mask.append(mask)

        # Convert to MLX arrays
        input_ids = mx.array(input_ids)
        attention_mask = mx.array(attention_mask)

        # Get embeddings
        embeddings = self(input_ids, attention_mask)

        # Truncate dimension if requested (Matryoshka)
        if truncate_dim is not None:
            embeddings = embeddings[:, :truncate_dim]
            # Re-normalize after truncation
            norms = mx.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        return embeddings


class ModelSentenceTransformers(Model):
    """Sanitization method for sentence transformers"""

    def __init__(self, config):
        super().__init__(config)

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            new_key = "model." + k
            sanitized_weights[new_key] = v
        return sanitized_weights
