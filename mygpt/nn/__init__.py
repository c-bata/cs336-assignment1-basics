import torch.nn
from jaxtyping import Float, Int
from einops import einsum, rearrange


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,  # final dimension of the input
        out_features: int,  # final dimension of the output.
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        sigma = (2 / (in_features + out_features)) ** 0.5
        weights: Float[torch.Tensor, "out_features in_features"] = torch.empty(
            (out_features, in_features),
            device=device,
            dtype=dtype
        )
        torch.nn.init.trunc_normal_(weights, 0, sigma, -3 * sigma, 3 * sigma)
        self.weights = torch.nn.Parameter(weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = x・W^t
        y = x @ self.weights.T
        return y


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,  # Size of the vocabulary
        embedding_dim: int,  # Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        weights: Float[torch.Tensor, "num_embeddings embedding_dim"] = torch.empty(
            (num_embeddings, embedding_dim),
            device=device,
            dtype=dtype
        )
        torch.nn.init.trunc_normal_(weights, 0, 1, -3, 3)
        self.weights = torch.nn.Parameter(weights)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Lookup the embedding vector for the given token IDs.
        return self.weights[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,  # Hidden dimension of the model
        eps: float = 1e-5,  # Epsilon value for numerical stability
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        weights: Float[torch.Tensor, "d_model"] = torch.ones(
            d_model,
            device=device,
            dtype=dtype
        )
        self.weights = torch.nn.Parameter(weights)
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        d_model = self.weights.shape[0]
        rms = torch.sqrt(1 / d_model * torch.einsum('ijk->ij', x ** 2) + self.eps)
        result = x / rms.unsqueeze(-1) * self.weights
        return result.to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        # d_model (int): Dimensionality of the feedforward input and output.
        d_model: int,
        # d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        w1: Float[torch.Tensor, "d_ff d_model"] = torch.empty(
            (d_ff, d_model),
            device=device,
            dtype=dtype
        )
        w2: Float[torch.Tensor, "d_model d_ff"] = torch.empty(
            (d_model, d_ff),
            device=device,
            dtype=dtype
        )
        w3: Float[torch.Tensor, "d_ff d_model"] = torch.empty(
            (d_ff, d_model),
            device=device,
            dtype=dtype
        )

        self.w1 = torch.nn.Parameter(w1)
        self.w2 = torch.nn.Parameter(w2)
        self.w3 = torch.nn.Parameter(w3)
    
    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> torch.Tensor:
        silu_input = torch.einsum("... d, fd -> ... f", x, self.w1)  # x @ self.w1.T
        silu_output: Float[torch.Tensor, "... d_ff"] = silu_input / (1 + torch.exp(-silu_input))
        silu_output *= torch.einsum("... d, fd -> ... f", x, self.w3)
        return torch.einsum("... f, df -> ... d", silu_output, self.w2)



# わからなかったのでカンニング
# https://github.com/stanford-cs336/assignment1-basics/compare/main...yusheng-ma:assignment1-basics:main#diff-60459a4c23bda530e330b82ac79f785fa0d977efafc672286d7b870b00f767c2
#class RotaryPositionalEmbedding(torch.nn.Module):
#    def __init__(
#        self,
#        theta: float,
#        d_k: int,  # dimension of query and key vectors
#        max_seq_len: int,  # Maximum sequence length that will be inputted
#        device: torch.device | None = None,
#    ) -> None:
#        super().__init__()
#
#        self.theta = theta
#        self.d_k = d_k
#        self.max_seq_len = max_seq_len
#        self.device = device
#
#        self.query = Linear(d_k, )
#        self.key = Linear(d_k, )
#    
#    def forward(
#        self,
#        x: Float[torch.Tensor, " ... sequence_length d_k"],
#        token_positions: Int[torch.Tensor, " ... sequence_length"],
#    ) -> torch.Tensor:
#
#        # スライドによると実装したらこんな感じになるっぽい
#        #
#        # query_states = self.q_proj(hidden_states)
#        # key_states = self.q_proj(hidden_states)
#        # value_states = self.v_proj(hidden_states)
#        #
#        # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#        # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#        # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#        # 
#        # # get the RoPE matrix cos/sin
#        # cos, sin = self.rotary_emb(value_states, token_positions)
#        # #multiply query/key inputs
#        # query_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
#
#        a = (self.query @ self.key) * x
#        angle = torch.diag(torch.arange(1, self.d_k+1)) * 
#        return ...
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None = None
    ):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be an odd number"

        # メモ: freqs変数の値がtheta_{i, k} に相当する。これは[max_seq_len, d_k/2]の2次元ベクトル。
        ## inverse frequency: constant^(2k/d).
        ## 次元はd_kではなく、d_k / 2なことに注意。さらに2kの計算をここではstep=2で実装している
        inv_freq: Float[torch.Tensor, "half_d_k"] = \
            1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        ## i
        positions: Float[torch.Tensor, "seq_len"] = \
            torch.arange(max_seq_len, device=device).float()
        ## freqs(angle, or theta): i * constant^(2k/d)
        freqs: Float[torch.Tensor, "seq_len half_d_k"] = \
            einsum(positions, inv_freq, "s, h_d -> s h_d")

        # Handouts資料より転載：
        # Since we only care about the relative rotation of tokens within a given sequence,
        # we can reuse the values we compute for cos(θi,k) and sin(θi,k) across layers,
        # and different batches.
        # If you would like to optimize it, you may use a single RoPE module referenced by all layers,
        # and it can have a 2d pre-computed buffer of sin and cos values created during init with
        # self.register_buffer(persistent=False), instead of a nn.Parameter.
        half_sin_buffer: Float[torch.Tensor, "seq_len half_d_k"] = torch.sin(freqs)
        half_cos_buffer: Float[torch.Tensor, "seq_len half_d_k"] = torch.cos(freqs)

        sin_buffer: Float[torch.Tensor, "seq_len d_k"] = self._interleave_half_buffer(half_sin_buffer)
        cos_buffer: Float[torch.Tensor, "seq_len d_k"] = self._interleave_half_buffer(half_cos_buffer)

        self.register_buffer("sin_buffer", sin_buffer, persistent=False)
        self.register_buffer("cos_buffer", cos_buffer, persistent=False)

    def _interleave_half_buffer(self, x: Float[torch.Tensor, "seq_len half_d_k"]) -> Float[torch.Tensor, "seq_len d_k"]:
        # stack_time is we stack x for 2 times
        # (d_k stack_time): first look d_k=0's all stack_time, then look d_k=1's all stack_time
        return rearrange([x, x], "stack_time seq_len d_k -> seq_len (d_k stack_time)")  # type: ignore

    def forward(
        self,
        x: Float[torch.Tensor, " ... sequence_length d_k"],
        token_positions: Int[torch.Tensor, " ... sequence_length"]
    ) -> Float[torch.Tensor, " ... sequence_length d_k"]:
        # print("=== RoPE DEBUG ===")
        # print(f"x.shape: {x.shape}")
        # print(f"token_positions.shape: {token_positions.shape}")
        # print(f"x.device: {x.device}")
        # print(f"token_positions.device: {token_positions.device}")

        x1: Float[torch.Tensor, " ... sequence_length half_d_k"] = x[..., ::2]  # even
        x2: Float[torch.Tensor, " ... sequence_length half_d_k"] = x[..., 1::2]  # odd

        # print(f"x1.shape (even): {x1.shape}")
        # print(f"x2.shape (odd): {x2.shape}")

        sin: Float[torch.Tensor, "... seq_len half_d_k"] = self.sin_buffer[token_positions][..., ::2]  # type: ignore
        cos: Float[torch.Tensor, "... seq_len half_d_k"] = self.cos_buffer[token_positions][..., ::2]  # type: ignore

        # 自動補維度（讓 sin/cos 和 x1 形狀一致）
        while sin.ndim < x1.ndim:
            sin = rearrange(sin, "... s d -> ... 1 s d")
            cos = rearrange(cos, "... s d -> ... 1 s d")

        # print(f"sin.shape: {sin.shape}")
        # print(f"cos.shape: {cos.shape}")

        x_even: Float[torch.Tensor, " ... sequence_length half_d_k"] = \
            einsum(x1, cos, "... s d, ... s d -> ... s d") - einsum(x2, sin, "... s d, ... s d -> ... s d")
        x_odd: Float[torch.Tensor, " ... sequence_length half_d_k"] = \
            einsum(x1, sin, "... s d, ... s d -> ... s d") + einsum(x2, cos, "... s d, ... s d -> ... s d")

        result = rearrange([x_even, x_odd], "stack_time ... seq_len d_k -> ... seq_len (d_k stack_time)")  # type: ignore
        # print(f"result.shape: {result.shape}")
        # print("==================")
        return result