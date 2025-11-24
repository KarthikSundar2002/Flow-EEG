import argparse
import os
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from dataset import MITBIH_Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hungarian matching between noise and MIT-BIH data.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--samples-per-record", type=int, default=200)
    parser.add_argument("--records", type=str, default="")
    parser.add_argument("--cache-path", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="hungarian_matched.pt")
    return parser.parse_args()


def parse_records(records_str: str) -> Optional[List[str]]:
    if not records_str:
        return None
    tokens = [token.strip() for token in records_str.split(",")]
    return [token for token in tokens if token]


def run_matching(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    default_records = [str(i) for i in range(100, 125) if i not in [110]]
    records = parse_records(args.records) or default_records

    dataset = MITBIH_Dataset(
        records=records,
        window_size=args.window_size,
        samples_per_record=args.samples_per_record,
        normalize=not args.no_normalize,
        cache_path=args.cache_path,
        use_cache=not args.no_cache,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    total_samples = len(dataset)
    if total_samples == 0:
        raise RuntimeError("Dataset returned zero samples; cannot run matching.")

    # Store noise/data as (N, window, 1) to keep the assignment axis explicit.
    matched_noise = torch.randn((total_samples, args.window_size, 1), device=device)
    matched_data = torch.zeros_like(matched_noise)

    offset = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Hungarian batches"):
            batch = batch.to(device)  # (B, 1, window)
            batch_size = batch.shape[0]

            data_points = batch.permute(0, 2, 1).contiguous()  # (B, window, 1)
            noise_chunk = matched_noise[offset : offset + batch_size]

            cost_matrix = torch.cdist(noise_chunk, data_points, p=2).cpu().numpy()

            for i in range(batch_size):
                row_ind, col_ind = linear_sum_assignment(cost_matrix[i], maximize=False)
                noise_chunk[i] = noise_chunk[i][row_ind]
                data_points[i] = data_points[i][col_ind]

            matched_noise[offset : offset + batch_size] = noise_chunk
            matched_data[offset : offset + batch_size] = data_points
            offset += batch_size

    matched_noise = matched_noise.permute(0, 2, 1).cpu()  # (N, 1, window)
    matched_data = matched_data.permute(0, 2, 1).cpu()

    base = os.path.splitext(args.output)[0]
    noise_path = f"{base}_noise.pt"
    data_path = f"{base}_data.pt"

    meta = {
        "window_size": args.window_size,
        "records": records,
        "samples_per_record": args.samples_per_record,
        "seed": args.seed,
    }

    torch.save({"tensor": matched_noise, **meta}, noise_path)
    torch.save({"tensor": matched_data, **meta}, data_path)
    print(f"Saved Hungarian noise to {noise_path}")
    print(f"Saved Hungarian data to {data_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    run_matching(cli_args)