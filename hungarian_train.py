import argparse
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from model import UNet1D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flow matching using pre-computed Hungarian pairs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--matched-path", type=str, help="Single torch file containing both noise and data tensors.")
    group.add_argument("--paired-paths", nargs=2, metavar=("NOISE", "DATA"), help="Two torch files containing noise and data tensors respectively.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/hungarian/latest.pt")
    parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint-path if it exists.")
    parser.add_argument("--save-every", type=int, default=2000, help="Checkpoint frequency in epochs.")
    parser.add_argument("--wandb-project", type=str, default="flow-Arythmia")
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def build_loader(tensor: torch.Tensor, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    global_step = checkpoint.get("global_step", 0)
    print(f"Resumed from {path} at epoch {start_epoch}")
    return start_epoch, global_step


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, global_step: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )
    print(f"Checkpoint saved to {path}")


def load_paired_tensors(args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    if args.matched_path:
        payload = torch.load(args.matched_path, map_location="cpu")
        return payload["noise"].contiguous(), payload["data"].contiguous()

    noise_payload = torch.load(args.paired_paths[0], map_location="cpu")
    data_payload = torch.load(args.paired_paths[1], map_location="cpu")
    return noise_payload["tensor"].contiguous(), data_payload["tensor"].contiguous()


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    noise, data = load_paired_tensors(args)
    if noise.shape != data.shape:
        raise ValueError(f"Noise/data shape mismatch: {noise.shape} vs {data.shape}")

    pin_memory = device.type == "cuda"
    noise_loader = build_loader(noise, args.batch_size, args.num_workers, pin_memory)
    data_loader = build_loader(data, args.batch_size, args.num_workers, pin_memory)

    model = UNet1D(in_channels=1, dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cfm = TargetConditionalFlowMatcher(sigma=0.0)

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.checkpoint_path):
        start_epoch, global_step = load_checkpoint(args.checkpoint_path, model, optimizer, device)

    use_wandb = (wandb is not None) and (not args.disable_wandb)
    run = None
    if use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            config={
                "device": str(device),
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "matched_path": args.matched_path,
                "noise_path": args.paired_paths[0] if args.paired_paths else None,
                "data_path": args.paired_paths[1] if args.paired_paths else None,
            },
            resume="allow",
            id=args.wandb_run_id,
        )
        wandb.watch(model, log="all")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        progress = tqdm(zip(noise_loader, data_loader), desc=f"Hungarian Epoch {epoch+1}/{args.epochs}", total=len(noise_loader))

        for (noise_batch,), (data_batch,) in progress:
            x0 = noise_batch.to(device, non_blocking=pin_memory)
            x1 = data_batch.to(device, non_blocking=pin_memory)

            t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1)
            vt = model(xt, t)

            loss = torch.mean((vt - ut) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})
            if use_wandb:
                wandb.log({"train/loss": loss.item()}, step=global_step)
            global_step += 1

        avg_loss = epoch_loss / len(noise_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")
        if use_wandb:
            wandb.log({"train/avg_loss": avg_loss, "epoch": epoch + 1}, step=global_step)

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            save_checkpoint(args.checkpoint_path, model, optimizer, epoch, global_step)

    if run is not None:
        run.finish()

if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)

