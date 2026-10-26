"""3D CNN for the MRI-only (Model 1) and multimodal (Model 3) models.

Both models share the same convolutional encoder. The multimodal model adds a
small branch for the tabular metadata and concatenates its output with the
MRI embedding before the classification head (late fusion), so the two
models differ only in the extra inputs.

The network is deliberately small (about 73k parameters): the training set
has 178 visit pairs, so a larger network would memorize the patients.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class VisitPairDataset(Dataset):
    """MRI volume of visit t, optional tabular features, and the target.

    Args:
        volumes: Array of shape (n, depth, height, width).
        tabular: Array of shape (n, n_features), already standardized, or
            None for the MRI-only model.
        labels: Binary targets of shape (n,).
        augment: Apply random flips and shifts (training folds only).
    """

    # Largest random shift, in voxels, applied along each axis.
    MAX_SHIFT = 3

    def __init__(self, volumes: np.ndarray, tabular: np.ndarray | None,
                 labels: np.ndarray, augment: bool = False):
        self.volumes = torch.from_numpy(volumes).float().unsqueeze(1)
        self.tabular = (torch.from_numpy(tabular).float() if tabular is not None
                        else torch.zeros(len(labels), 0))
        self.labels = torch.from_numpy(labels).float()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        volume = self.volumes[index]
        if self.augment:
            volume = self._augment(volume)
        return volume, self.tabular[index], self.labels[index]

    def _augment(self, volume: torch.Tensor) -> torch.Tensor:
        """Left-right flip and small translations, both label-preserving.

        The first spatial axis is sagittal, so flipping it mirrors the left
        and right hemispheres. Atrophy is roughly symmetric, so the label
        still holds.
        """
        if torch.rand(1).item() < 0.5:
            volume = torch.flip(volume, dims=[1])
        shifts = torch.randint(-self.MAX_SHIFT, self.MAX_SHIFT + 1, (3,)).tolist()
        return torch.roll(volume, shifts=shifts, dims=[1, 2, 3])


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv3d -> BatchNorm -> ReLU -> 2x max pooling."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(2),
    )


class BrainCNN(nn.Module):
    """3D CNN encoder with an optional tabular branch.

    Args:
        n_tabular: Number of tabular features. 0 builds the MRI-only model.
        channels: Output channels of each convolutional block. Each block
            halves the spatial size (64 -> 32 -> 16 -> 8 -> 4).
        tabular_dim: Width of the tabular branch output.
        dropout: Dropout rate before the classification head.
    """

    def __init__(self, n_tabular: int = 0, channels: tuple[int, ...] = (8, 16, 32, 64),
                 tabular_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        blocks, in_channels = [], 1
        for out_channels in channels:
            blocks.append(conv_block(in_channels, out_channels))
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool3d(1), nn.Flatten())

        self.tabular = None
        head_inputs = channels[-1]
        if n_tabular > 0:
            self.tabular = nn.Sequential(nn.Linear(n_tabular, tabular_dim), nn.ReLU())
            head_inputs += tabular_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_inputs, 1))

    def forward(self, volume: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """Return one logit per sample (apply a sigmoid for the probability)."""
        features = self.encoder(volume)
        if self.tabular is not None:
            features = torch.cat([features, self.tabular(tabular)], dim=1)
        return self.head(features).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
