import argparse
import os
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from model import UNet1D
from ptbxl_dataset import PTBXLWaveformDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flow model on PTB-XL without Hungarian matching.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the extracted PTB-XL folder.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--folds", type=str, default=None, help="Comma-separated list of folds overriding --split.")
    parser.add_argument("--sampling-rate", type=int, default=500, choices=[100, 500])
    parser.add_argument("--lead", type=str, default="II", help="Lead to train on (e.g., II, V2).")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--samples-per-record", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ptbxl/latest.pt")
    parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint if it exists.")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="ptbxl-flow")
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
):
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
        pin_memory=(device.type == "cuda"),
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
        progress = tqdm(dataloader, desc=f"PTB-XL Epoch {epoch+1}/{args.epochs}")

        for batch in progress:
            # (B, 1, window)
            x1 = batch.to(device)
            x0 = torch.randn_like(x1)
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


