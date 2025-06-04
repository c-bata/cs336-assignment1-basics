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
