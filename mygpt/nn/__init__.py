import torch.nn


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,  # final dimension of the input
        out_features: int,  # final dimension of the output.
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        mean = torch.zeros((out_features, in_features), dtype=dtype)
        self._weights = torch.normal(mean, 3)

        if device is not None:
            self.weights.to(device=device)

    @property
    def weights(self) -> torch.Tensor:
        return self._weights
    
    @weights.setter
    def weights(self, w) -> None:
        assert self._weights.shape == w.shape
        self._weights = w
    
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
        mean = torch.zeros((num_embeddings, embedding_dim), dtype=dtype)
        self._weights = torch.normal(mean, 1)
        if device is not None:
            self._weights.to(device)

    @property
    def weights(self) -> torch.Tensor:
        return self._weights
    
    @weights.setter
    def weights(self, w) -> None:
        assert self._weights.shape == w.shape
        self._weights = w
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Lookup the embedding vector for the given token IDs.
        return self._weights[token_ids]
