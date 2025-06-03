import torch.nn


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,  # final dimension of the input
        out_features: int,  # final dimension of the output.
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        weights_shape = (out_features, in_features)
        self._weights = torch.normal(torch.zeros(weights_shape, dtype=dtype), 3)

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