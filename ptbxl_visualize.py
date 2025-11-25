import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from dit_model import ECG_DiT_1D
from ptbxl_dataset import PTBXLWaveformDataset

def rk4_step(model, x, t, dt, y):
    """
    Runge-Kutta 4 Solver for high-quality signal generation.
    """
    t_tensor = t * torch.ones(x.shape[0], device=x.device)
    
    # K1
    k1 = model(x, t_tensor, y)
    
    # K2
    t2 = (t + dt * 0.5) * torch.ones(x.shape[0], device=x.device)
    k2 = model(x + k1 * (0.5 * dt), t2, y)
    
    # K3
    k3 = model(x + k2 * (0.5 * dt), t2, y)
    
    # K4
    t4 = (t + dt) * torch.ones(x.shape[0], device=x.device)
    k4 = model(x + k3 * dt, t4, y)
    
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def generate_samples(args, device):
    print(f"Loading model from {args.checkpoint}...")
    
    # Initialize model with the same architecture as training
    model = ECG_DiT_1D(
        input_size=args.window_size,
        patch_size=args.patch_size,
        hidden_size=args.hidden,
        depth=args.depth,
        num_heads=args.heads,
        num_classes=args.num_classes
    ).to(device)
    model = torch.compile(model)
    
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Handle cases where checkpoint saves 'model' state dict or just state dict
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    model.eval()
    
    # Setup Generation
    num_samples = args.num_samples
    
    # Start from pure Gaussian Noise
    # Shape: (Batch, Channels, Time)
    # Using 1 channel since ptbxl_train.py defaults to single lead training
    x = torch.randn(num_samples, 1, args.window_size).to(device)
    
    # Class conditioning
    # If using classes, generate a mix, or specific class if requested
    if args.class_idx is not None:
        y = torch.ones(num_samples, dtype=torch.long).to(device) * args.class_idx
        print(f"Generating samples for Class Index {args.class_idx}")
    else:
        # Generate random classes
        y = torch.randint(0, args.num_classes, (num_samples,)).to(device)
        print(f"Generating samples for random classes (0-{args.num_classes-1})")

    # Solve ODE
    steps = args.steps
    dt = 1.0 / steps
    
    print(f"Sampling with RK4 ({steps} steps)...")
    
    with torch.no_grad():
        for i in range(steps):
            t = i * dt
            x = rk4_step(model, x, t, dt, y)
            
    return x.cpu().numpy(), y.cpu().numpy()

def load_real_samples(args, device):
    """
    Load real samples from the dataset matching the generated classes.
    """
    if args.data_dir is None:
        print("Warning: --data-dir not provided. Skipping real sample loading.")
        return None, None
    
    print(f"Loading real samples from {args.data_dir}...")
    try:
        dataset = PTBXLWaveformDataset(
            data_dir=args.data_dir,
            split=args.split,
            sampling_rate=args.sampling_rate,
            leads=[args.lead] if args.lead else None,
            window_size=args.window_size,
            samples_per_record=1,
            normalize=True,
            cache_dir=args.cache_dir,
            seed=args.seed,
        )
        
        # Get all samples and their labels
        all_signals = []
        all_labels = []
        for i in range(len(dataset)):
            signal, label = dataset[i]
            all_signals.append(signal.numpy())
            all_labels.append(label)
        
        all_signals = np.array(all_signals)  # Shape: (N, C, T)
        all_labels = np.array(all_labels)
        
        print(f"Loaded {len(all_signals)} real samples with {len(np.unique(all_labels))} unique classes")
        return all_signals, all_labels
        
    except Exception as e:
        print(f"Error loading real samples: {e}")
        return None, None

def plot_results(signals, labels, real_signals, real_labels, args):
    """
    Plots the generated signals alongside real samples of the same class.
    """
    num_generated = len(signals)
    num_real_per_class = args.num_real_samples if hasattr(args, 'num_real_samples') else 3
    
    # Scaling factor to make plot look like standard ECG (approximate)
    # Since model output is ~[-1, 1] (normalized), and QRS is often ~1.5mV
    SCALE_FACTOR = 1.5
    
    # Determine unique classes in generated samples
    unique_classes = np.unique(labels)
    
    # If we have real samples, match them by class
    has_real = real_signals is not None and real_labels is not None
    
    # Calculate total number of subplots
    # For each generated sample, show it + matching real samples
    total_plots = 0
    plot_layout = []
    
    for gen_idx, gen_label in enumerate(labels):
        plot_layout.append(('generated', gen_idx, gen_label))
        total_plots += 1
        
        if has_real:
            # Find real samples of the same class
            matching_real = np.where(real_labels == gen_label)[0]
            if len(matching_real) > 0:
                # Sample randomly from matching real samples
                np.random.seed(42 + gen_idx)  # Deterministic sampling
                selected = np.random.choice(
                    matching_real, 
                    size=min(num_real_per_class, len(matching_real)), 
                    replace=False
                )
                for real_idx in selected:
                    plot_layout.append(('real', real_idx, gen_label))
                    total_plots += 1
    
    # Create subplots: 2 columns (generated on left, real on right) or single column
    if has_real and num_real_per_class > 0:
        # Use 2 columns: generated samples on left, real on right
        fig, axs = plt.subplots(total_plots, 1, figsize=(14, 2 * total_plots), sharex=True)
        if total_plots == 1:
            axs = [axs]
    else:
        # Single column for generated only
        fig, axs = plt.subplots(num_generated, 1, figsize=(12, 2 * num_generated), sharex=True)
        if num_generated == 1:
            axs = [axs]
    
    plot_idx = 0
    for plot_type, idx, label in plot_layout:
        if plot_type == 'generated':
            signal = signals[idx, 0, :] * SCALE_FACTOR
            axs[plot_idx].plot(signal, color='blue', linewidth=0.8, label='Generated')
            axs[plot_idx].set_title(f"Generated Sample {idx+1} | Class: {label}", fontweight='bold')
        else:  # real
            signal = real_signals[idx, 0, :] * SCALE_FACTOR
            axs[plot_idx].plot(signal, color='green', linewidth=0.8, label='Real')
            axs[plot_idx].set_title(f"Real Sample (Class: {label})", style='italic')
        
        axs[plot_idx].grid(True, alpha=0.3, linestyle='--')
        axs[plot_idx].set_ylabel("Ampl (mV)")
        axs[plot_idx].axhline(0, color='red', alpha=0.1)
        axs[plot_idx].legend(loc='upper right', fontsize=8)
        plot_idx += 1

    axs[-1].set_xlabel("Time (samples)")
    plt.tight_layout()
    
    out_path = "ptbxl_generated.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PTB-XL Flow Model")
    
    # Architecture Args (Must match training!)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ptbxl/latest.pt", help="Path to model checkpoint")
    parser.add_argument("--window-size", type=int, default=1024, help="Window size used in training")
    parser.add_argument("--hidden", type=int, default=512, help="DiT hidden size")
    parser.add_argument("--depth", type=int, default=8, help="DiT depth")
    parser.add_argument("--heads", type=int, default=8, help="DiT heads")
    parser.add_argument("--patch-size", type=int, default=16, help="DiT patch size")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of classes model was trained with (check training logs)")
    
    # Generation Args
    parser.add_argument("--num-samples", type=int, default=5, help="How many signals to generate")
    parser.add_argument("--steps", type=int, default=50, help="Number of ODE steps (higher = better quality, slower)")
    parser.add_argument("--class-idx", type=int, default=None, help="Force specific class index (optional)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset Args (for loading real samples)
    parser.add_argument("--data-dir", type=str, default=None, help="Path to PTB-XL data directory (required for real samples)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"], help="Dataset split to load real samples from")
    parser.add_argument("--lead", type=str, default="II", help="Lead name (must match training)")
    parser.add_argument("--sampling-rate", type=int, default=100, choices=[100, 500], help="Sampling rate (must match training)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-real-samples", type=int, default=3, help="Number of real samples to show per generated sample")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Generate samples
    result = generate_samples(args, device)
    if result is not None:
        signals, labels = result
        
        # Load real samples if data_dir is provided
        real_signals, real_labels = None, None
        if args.data_dir is not None:
            real_signals, real_labels = load_real_samples(args, device)
        
        plot_results(signals, labels, real_signals, real_labels, args)