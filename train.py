import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from dataset import MITBIH_Dataset
from dit_model import ECG_DiT_1D


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = torch.compile(model)
    #optimizer.load_state_dict(checkpoint["optimizer"])
    if "epoch" in checkpoint:
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0
    if "global_step" in checkpoint:
        global_step = checkpoint["global_step"]
    else:
        global_step = 0
    if "wandb_run_id" in checkpoint:
        wandb_id = checkpoint["wandb_run_id"]
    else:
        wandb_id = None
    #global_step = checkpoint.get("global_step", 0)
    #wandb_id = checkpoint.get("wandb_run_id")
    print(f"Resumed from checkpoint {path} at epoch {start_epoch}")
    return model, start_epoch, global_step, wandb_id


def train(
    resume_path=None,
    cache_path=None,
    wandb_project="flow-arythmia",
    wandb_run_id=None,
    disable_wandb=False,
    dit_hidden=256,
    dit_depth=6,
    dit_heads=8,
):
    # --- Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1024
    LR = 1e-4
    EPOCHS = 100000 # Increase for better fidelity
    WINDOW_SIZE = 256
    
    print(f"Training on {DEVICE}...")

    # --- Data ---
    # We load more records for training
    records = [str(i) for i in range(100, 125) if i not in [110]] # Some are missing in MIT-BIH
    dataset = MITBIH_Dataset(
        records=records,
        window_size=WINDOW_SIZE,
        samples_per_record=200,
        cache_path=cache_path,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # --- Model ---
    model = ECG_DiT_1D(
        input_size=WINDOW_SIZE,
        patch_size=16,
        hidden_size=dit_hidden,
        depth=dit_depth,
        num_heads=dit_heads,
        num_classes=1,
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # --- Flow Matching Wrapper ---
    # sigma=0.0 implies "Optimal Transport" paths (straight lines)
    cfm = TargetConditionalFlowMatcher(sigma=0.0)

    # --- Training Loop ---
    model.train()
    loss_history = []

    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")

    start_epoch = 0
    global_step = 0

    resume_target = resume_path
    if resume_target:
        model, start_epoch, global_step, saved_wandb_id = load_checkpoint(resume_target, model, optimizer, DEVICE)
        if wandb_run_id is None and saved_wandb_id:
            wandb_run_id = saved_wandb_id

    use_wandb = (wandb is not None) and (not disable_wandb)
    wandb_run = None
    if use_wandb:
        wandb_config = {
            "device": str(DEVICE),
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "window_size": WINDOW_SIZE,
            "samples_per_record": dataset.__dict__.get("samples_per_record", 200),
            "model": "ECG_DiT_1D",
            "dit_hidden": dit_hidden,
            "dit_depth": dit_depth,
            "dit_heads": dit_heads,
        }
        wandb_run = wandb.init(
            project=wandb_project,
            config=wandb_config,
            resume="allow",
            id=wandb_run_id,
        )
        wandb.watch(model, log="all")

    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            # batch is (B, 1, 256)
            x1 = batch.to(DEVICE)
            
            # Generate random noise (source distribution)
            x0 = torch.randn_like(x1).to(DEVICE)
            
            # Sample time t, intermediate samples xt, and target vector field ut
            # t: (B,)
            # xt: (B, 1, 256) (Noisy data at time t)
            # ut: (B, 1, 256) (The direction we want to flow)
            t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1)
            
            # Predict vector field with model
            y = torch.zeros(x1.shape[0], dtype=torch.long, device=DEVICE)
            vt = model(xt, t, y)
            
            # Loss is simply MSE between predicted direction and target direction
            loss = torch.mean((vt - ut) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            if use_wandb:
                wandb.log({"train/loss": loss.item()}, step=global_step)
            global_step += 1

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

        # Save Checkpoint every 2000 epochs
        if (epoch + 1) % 2000 == 0:
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "wandb_run_id": wandb_run.id if wandb_run else None,
                },
                checkpoint_path,
            )

    # Save Final Model
    torch.save(model.state_dict(), "ecg_flow_model_final.pth")
    save_checkpoint(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": EPOCHS - 1,
            "global_step": global_step,
            "wandb_run_id": wandb_run.id if wandb_run else None,
        },
        checkpoint_path,
    )
    print("Training Complete!")
    
    # Plot Loss
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig("training_loss.png")

    if use_wandb:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ECG Flow Model")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--cache", type=str, default=None, help="Optional dataset cache override path")
    parser.add_argument("--wandb-project", type=str, default="flow-arythmia", help="Weights & Biases project name")
    parser.add_argument("--wandb-run-id", type=str, default=None, help="Existing Weights & Biases run ID to resume")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--dit-hidden", type=int, default=256, help="Hidden size for DiT model")
    parser.add_argument("--dit-depth", type=int, default=6, help="Number of transformer blocks in DiT")
    parser.add_argument("--dit-heads", type=int, default=8, help="Number of attention heads in DiT")
    args = parser.parse_args()

    train(
        resume_path=args.resume,
        cache_path=args.cache,
        wandb_project=args.wandb_project,
        wandb_run_id=args.wandb_run_id,
        disable_wandb=args.no_wandb,
        dit_hidden=args.dit_hidden,
        dit_depth=args.dit_depth,
        dit_heads=args.dit_heads,
    )