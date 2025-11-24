import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb


def _normalize_signal(signal: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalizes signal per-channel to zero mean / unit variance.

    Args:
        signal: Tensor of shape (C, T).
        eps: Numerical stability constant.
    """
    mean = signal.mean(dim=1, keepdim=True)
    std = signal.std(dim=1, keepdim=True)
    return (signal - mean) / (std + eps)


class PTBXLWaveformDataset(Dataset):
    """
    Lightweight PTB-XL loader that supports random crops, lead selection,
    and optional caching of the processed tensor dataset.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        folds: Optional[Sequence[int]] = None,
        sampling_rate: int = 500,
        leads: Optional[Iterable[str]] = None,
        window_size: int = 1024,
        samples_per_record: int = 1,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
        cache_prefix: Optional[str] = None,
        seed: int = 42,
        auto_download: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing the PTB-XL files (ptbxl_database.csv,
                      records100/, records500/, ...). Run the PhysioNet
                      instructions to download this folder locally.
            split: Logical split. Options: "train", "val", "test", or "all".
                   Defaults to the recommended folds (1-8 train, 9 val, 10 test).
            folds: Optional explicit list of folds to load (overrides split).
            sampling_rate: 500 or 100 Hz variant.
            leads: Iterable of lead names (e.g., ["I", "II"]). Defaults to
                   all 12 leads when None. Lead names are matched case-insensitively.
            window_size: Number of samples per training example. If the record
                         is shorter than window_size it is zero-padded.
            samples_per_record: How many random crops to sample from each record.
            normalize: Whether to z-score each crop per channel.
            cache_dir: Optional directory for serialized tensors that
                       accelerate repeated runs.
            cache_prefix: Extra identifier for the cache filename (useful when
                          running different subsets).
            seed: RNG seed controlling crop positions.
            auto_download: If True and files missing, attempt wfdb.dl_database.
        """

        self.data_dir = os.path.abspath(data_dir)
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.samples_per_record = samples_per_record
        self.normalize = normalize
        self.leads = [lead.upper() for lead in leads] if leads else None
        self.rng = np.random.default_rng(seed)

        if auto_download:
            self._maybe_download()

        csv_path = os.path.join(self.data_dir, "ptbxl_database.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Missing ptbxl_database.csv in {self.data_dir}. "
                "Download PTB-XL from PhysioNet first: https://physionet.org/content/ptb-xl/1.0.3/"
            )

        df = pd.read_csv(csv_path)
        target_folds = self._resolve_folds(split, folds)
        if target_folds is not None:
            df = df[df["strat_fold"].isin(target_folds)]

        if df.empty:
            raise RuntimeError("No records selected for the given split/folds.")

        cache_dir = cache_dir or os.path.join(self.data_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_tag = cache_prefix or split
        lead_tag = "all" if self.leads is None else "-".join(self.leads)
        cache_name = (
            f"ptbxl_sr{sampling_rate}_win{window_size}_spr{samples_per_record}_"
            f"folds{'-'.join(map(str, target_folds)) if target_folds else 'all'}_"
            f"leads{lead_tag}_norm{int(normalize)}_{cache_tag}.pt"
        )
        self.cache_path = os.path.join(cache_dir, cache_name)

        if os.path.exists(self.cache_path):
            payload = torch.load(self.cache_path, map_location="cpu")
            self.segments = [seg.clone() for seg in payload["segments"]]
            return

        self.segments: List[torch.Tensor] = []
        self._build_dataset(df)
        if not self.segments:
            raise RuntimeError("Failed to collect any PTB-XL segments.")

        stacked = torch.stack(self.segments)
        torch.save({"segments": stacked}, self.cache_path)

    def _maybe_download(self) -> None:
        """Downloads PTB-XL via wfdb if files are missing."""
        markers = [
            os.path.join(self.data_dir, "ptbxl_database.csv"),
            os.path.join(self.data_dir, "records100"),
            os.path.join(self.data_dir, "records500"),
        ]
        if all(os.path.exists(marker) for marker in markers):
            return
        os.makedirs(self.data_dir, exist_ok=True)
        print("Downloading PTB-XL via wfdb.dl_database (this may take a while)...")
        wfdb.dl_database("ptb-xl", dl_dir=self.data_dir, keep_subdirs=True)

    @staticmethod
    def _resolve_folds(split: str, folds: Optional[Sequence[int]]) -> Optional[List[int]]:
        if folds is not None:
            return list(folds)

        split = split.lower()
        if split == "train":
            return list(range(1, 9))
        if split in ("val", "validation"):
            return [9]
        if split == "test":
            return [10]
        if split == "all":
            return None
        raise ValueError(f"Unknown split '{split}'. Expected train/val/test/all.")

    def _resolve_record_path(self, row: pd.Series) -> str:
        column = "filename_hr" if self.sampling_rate == 500 else "filename_lr"
        relative_path = row[column]
        return os.path.join(self.data_dir, relative_path)

    @staticmethod
    def _select_leads(
        signal: np.ndarray, sig_names: Sequence[str], targets: Optional[List[str]]
    ) -> np.ndarray:
        if targets is None:
            return signal.T  # (C, T) with all channels
        indices = []
        upper_names = [name.upper() for name in sig_names]
        for lead in targets:
            if lead not in upper_names:
                raise ValueError(f"Requested lead '{lead}' not present. Available: {sig_names}")
            indices.append(upper_names.index(lead))
        return signal[:, indices].T

    def _build_dataset(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            record_path = self._resolve_record_path(row)
            try:
                signal, fields = wfdb.rdsamp(record_path)
            except Exception as exc:
                print(f"Skipping {record_path}: {exc}")
                continue

            leads = self._select_leads(signal, fields["sig_name"], self.leads)
            length = leads.shape[1]
            if length < self.window_size:
                pad = self.window_size - length
                leads = np.pad(leads, ((0, 0), (0, pad)))
            elif length > self.window_size:
                for _ in range(self.samples_per_record):
                    start = self.rng.integers(0, max(1, length - self.window_size + 1))
                    crop = leads[:, start : start + self.window_size]
                    tensor = torch.from_numpy(crop).float()
                    if self.normalize:
                        tensor = _normalize_signal(tensor)
                    self.segments.append(tensor)
                continue

            tensor = torch.from_numpy(leads[:, : self.window_size]).float()
            if self.normalize:
                tensor = _normalize_signal(tensor)
            self.segments.append(tensor)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Output shape: (C, window_size)
        return self.segments[idx]


