import torch.nn
from jaxtyping import Float


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
