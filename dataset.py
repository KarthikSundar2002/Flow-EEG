import torch
from torch.utils.data import Dataset
import wfdb
import numpy as np
import os

class MITBIH_Dataset(Dataset):
    def __init__(
        self,
        records=None,
        window_size=256,
        samples_per_record=100,
        normalize=True,
        cache_path=None,
        use_cache=True,
    ):
        """
        Args:
            records (list): List of record names (strings) to download/load. 
                            If None, loads a small subset for demo.
            window_size (int): Length of the ECG segment.
            samples_per_record (int): How many random crops to take from each record.
            normalize (bool): Whether to standardize the signal (0 mean, 1 std).
        """
        self.window_size = window_size
        self.normalize = normalize
        self.data = []
        self.use_cache = use_cache
        self.samples_per_record = samples_per_record

        # Derive a cache location that differentiates preprocessing configs
        if cache_path is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "cache")
            if use_cache:
                os.makedirs(cache_dir, exist_ok=True)
            cache_name = f"mitbih_ws{window_size}_spr{samples_per_record}_norm{int(normalize)}.pt"
            cache_path = os.path.join(cache_dir, cache_name)
        self.cache_path = cache_path

        if self.use_cache and os.path.exists(self.cache_path):
            print(f"Loading cached dataset from {self.cache_path} ...")
            cached = torch.load(self.cache_path, map_location="cpu")
            self.data = [seg.clone() for seg in cached["segments"]]
            print(f"Dataset ready (cached): {len(self.data)} segments of shape {self.data[0].shape}")
            return

        # Default small subset of MIT-BIH if none provided
        if records is None:
            records = ['100', '101', '102', '103', '104']

        print(f"Loading/Downloading {len(records)} records from MIT-BIH...")
        
        for rec_name in records:
            try:
                # pn_dir='mitdb' tells wfdb to look in the PhysioNet MIT-BIH database
                record = wfdb.rdrecord(rec_name, pn_dir='mitdb')
                
                # Extract Lead II (usually index 0, but we check to be safe)
                # record.p_signal is (Total_Time, Channels)
                signal = record.p_signal[:, 0]
                
                # Global normalization per record before slicing
                if self.normalize:
                    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

                # Create random slices
                max_idx = len(signal) - window_size
                for _ in range(samples_per_record):
                    start = np.random.randint(0, max_idx)
                    segment = signal[start : start + window_size]
                    
                    # Convert to tensor and add Channel dim: (1, Length)
                    segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
                    self.data.append(segment_tensor)
                    
            except Exception as e:
                print(f"Error loading record {rec_name}: {e}")

        if not self.data:
            raise RuntimeError("Dataset download failed; no segments collected.")

        if self.use_cache:
            stacked = torch.stack(self.data)
            torch.save({"segments": stacked}, self.cache_path)
            print(f"Cached dataset to {self.cache_path}")

        print(f"Dataset ready: {len(self.data)} segments of shape {self.data[0].shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    # Test the loader
    ds = MITBIH_Dataset()
    print(f"Sample shape: {ds[0].shape}")