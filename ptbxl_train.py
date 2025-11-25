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

from dit_model import ECG_DiT_1D
from ptbxl_dataset import PTBXLWaveformDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flow model on PTB-XL without Hungarian matching.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the extracted PTB-XL folder.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--folds", type=str, default=None, help="Comma-separated list of folds overriding --split.")
    parser.add_argument("--sampling-rate", type=int, default=100, choices=[100, 500])
    parser.add_argument("--lead", type=str, default="II", help="Lead to train on (e.g., II, V2).")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--samples-per-record", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ptbxl/latest.pt")
    parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint if it exists.")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="flow-Arythmiz")
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--dit-hidden", type=int, default=512, help="DiT hidden size")
    parser.add_argument("--dit-depth", type=int, default=8, help="DiT depth (number of blocks)")
    parser.add_argument("--dit-heads", type=int, default=8, help="DiT attention heads")
    parser.add_argument("--dit-patch-size", type=int, default=16, help="DiT patch size")
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

    # Get number of classes from dataset
    num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else 1
    print(f"Using {num_classes} classes for conditioning")

    model = ECG_DiT_1D(
        input_size=args.window_size,
        patch_size=args.dit_patch_size,
        hidden_size=args.dit_hidden,
        depth=args.dit_depth,
        num_heads=args.dit_heads,
        num_classes=num_classes,
    ).to(device)
    model = torch.compile(model)
   
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3400)

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
                "dit_hidden": args.dit_hidden,
                "dit_depth": args.dit_depth,
                "dit_heads": args.dit_heads,
                "dit_patch_size": args.dit_patch_size,
                "num_classes": num_classes,
            },
        )
        wandb.watch(model, log="all")

    model.train()
    cfm = TargetConditionalFlowMatcher(sigma=0.05)
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"PTB-XL Epoch {epoch+1}/{args.epochs}")

        for batch_data in progress:
            # Handle tuple/list (signal, label) or just signal
            if isinstance(batch_data, (tuple, list)):
                if len(batch_data) >= 2:
                    x1, y = batch_data[:2]
                else:
                    x1 = batch_data[0]
                    y = torch.zeros(x1.shape[0], dtype=torch.long)
            else:
                x1 = batch_data
                y = torch.zeros(x1.shape[0], dtype=torch.long)

            x1 = x1.to(device)
            y = y.to(device, dtype=torch.long)
            
            # Ensure x1 is (B, 1, window) - take first channel if multi-channel
            if x1.dim() == 3 and x1.shape[1] > 1:
                x1 = x1[:, 0:1, :]  # Take first channel
            
            x0 = torch.randn_like(x1)
            t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                vt = model(xt, t, y)
            ut = ut.to(torch.bfloat16)
            loss = torch.mean((vt - ut) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            current_lr = optimizer.param_groups[0].get("lr", args.lr)
            epoch_loss += loss.item()
            global_step += 1
            progress.set_postfix({"loss": loss.item()})
            if use_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": current_lr,
                    },
                    step=global_step,
                )

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


