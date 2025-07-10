"""Embeddings classification dataset with optional in-memory preloading."""

import os
import torch
from typing_extensions import override

from eva.core.data.datasets import embeddings as embeddings_base


class EmbeddingsClassificationDataset(embeddings_base.EmbeddingsDataset[torch.Tensor]):
    """Embeddings dataset class for classification tasks (with optional in-memory preloading)."""

    def __init__(
        self,
        root: str,
        manifest_file: str,
        split: str | None = None,
        column_mapping: dict = {},
        embeddings_transforms=None,
        target_transforms=None,
        preload: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            manifest_file=manifest_file,
            split=split,
            column_mapping=column_mapping,
            embeddings_transforms=embeddings_transforms,
            target_transforms=target_transforms,
        )
        self._preload = preload
        self._cached_embeddings = {}

    @override
    def setup(self):
        super().setup()
        if not self._preload:
            return

        for idx in range(len(self._data)):
            path = os.path.join(self._root, self.filename(idx))
            tensor = torch.load(path, map_location="cpu")
            if isinstance(tensor, list):
                tensor = tensor[0]
            self._cached_embeddings[idx] = tensor.squeeze(0)

    @override
    def load_embeddings(self, index: int) -> torch.Tensor:
        if self._preload:
            return self._cached_embeddings[index]

        filename = self.filename(index)
        embeddings_path = os.path.join(self._root, filename)
        tensor = torch.load(embeddings_path, map_location="cpu")
        if isinstance(tensor, list):
            if len(tensor) > 1:
                raise ValueError(
                    f"Expected a single tensor in the .pt file, but found {len(tensor)}."
                )
            tensor = tensor[0]
        return tensor.squeeze(0)

    @override
    def load_target(self, index: int) -> torch.Tensor:
        target = self._data.at[index, self._column_mapping["target"]]
        return torch.tensor(target, dtype=torch.int64)

    @override
    def __len__(self) -> int:
        return len(self._data)
