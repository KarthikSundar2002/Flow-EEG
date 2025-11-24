import argparse
import os
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from model import UNet1D
from ptbxl_dataset import PTBXLWaveformDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train flow model on PTB-XL using Hungarian-matched noise/data pairs."
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--folds", type=str, default=None)
    parser.add_argument("--sampling-rate", type=int, default=500, choices=[100, 500])
    parser.add_argument("--lead", type=str, default="II")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--samples-per-record", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ptbxl_hungarian/latest.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="ptbxl-flow-hungarian")
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--save-every", type=int, default=500)
    return parser.parse_args()


def parse_fold_override(folds: Optional[str]) -> Optional[List[int]]:
    if not folds:
        return None
    return [int(token.strip()) for token in folds.split(",") if token.strip()]


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint.get("epoch", 0) + 1
    global_step = checkpoint.get("global_step", 0)
    wandb_id = checkpoint.get("wandb_run_id")
    print(f"Resumed from {path} at epoch {epoch}")
    return epoch, global_step, wandb_id


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    wandb_id: Optional[str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "wandb_run_id": wandb_id,
        },
        path,
    )
    print(f"Checkpoint saved to {path}")


def hungarian_align(noise: torch.Tensor, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Matches each noise example with a unique PTB-XL crop to minimize the L2 distance.
    Args:
        noise: (B, C, T)
        data: (B, C, T)
    Returns:
        Tuple of tensors reordered so noise[i] pairs with data[i].
    """
    batch_size = noise.shape[0]
    flat_noise = noise.view(batch_size, -1)
    flat_data = data.view(batch_size, -1)
    cost = torch.cdist(flat_noise, flat_data).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost, maximize=False)
    aligned_noise = noise[row_ind]
    aligned_data = data[col_ind]
    return aligned_noise, aligned_data


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    dataset = PTBXLWaveformDataset(
        data_dir=args.data_dir,
        split=args.split,
        folds=parse_fold_override(args.folds),
        sampling_rate=args.sampling_rate,
        leads=[args.lead],
        window_size=args.window_size,
        samples_per_record=args.samples_per_record,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    model = UNet1D(in_channels=1, dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cfm = TargetConditionalFlowMatcher(sigma=0.0)

    start_epoch = 0
    global_step = 0
    wandb_id = args.wandb_run_id

    if args.resume and os.path.exists(args.checkpoint):
        start_epoch, global_step, saved_id = load_checkpoint(args.checkpoint, model, optimizer, device)
        if wandb_id is None:
            wandb_id = saved_id

    use_wandb = (wandb is not None) and (not args.disable_wandb)
    run = None
    if use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            resume="allow",
            id=wandb_id,
            config={
                "data_dir": args.data_dir,
                "split": args.split,
                "folds": args.folds,
                "sampling_rate": args.sampling_rate,
                "lead": args.lead,
                "window_size": args.window_size,
                "batch_size": args.batch_size,
                "lr": args.lr,
            },
        )
        wandb.watch(model, log="all")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"PTB-XL Hungarian Epoch {epoch+1}/{args.epochs}")

        for data_batch in progress:
            data_cpu = data_batch.cpu()
            noise_cpu = torch.randn_like(data_cpu)
            matched_noise, matched_data = hungarian_align(noise_cpu, data_cpu)

            x0 = matched_noise.to(device, non_blocking=True)
            x1 = matched_data.to(device, non_blocking=True)
            t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1)
            vt = model(xt, t)
            loss = torch.mean((vt - ut) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            progress.set_postfix({"loss": loss.item()})
            if use_wandb:
                wandb.log({"train/loss": loss.item()}, step=global_step)

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")
        if use_wandb:
            wandb.log({"train/avg_loss": avg_loss, "epoch": epoch + 1}, step=global_step)

        if ((epoch + 1) % args.save_every == 0) or (epoch + 1 == args.epochs):
            save_checkpoint(args.checkpoint, model, optimizer, epoch, global_step, run.id if run else wandb_id)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)


