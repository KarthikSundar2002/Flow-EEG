import torch
import matplotlib.pyplot as plt
import numpy as np
from dit_model import ECG_DiT_1D
from dataset import MITBIH_Dataset

def generate_ecg(model, num_samples=5, device="cuda"):
    model.eval()
    window_size = 256
    
    # 1. Start with Gaussian Noise (t=0)
    x = torch.randn(num_samples, 1, window_size).to(device)
    
    # 2. Class labels (dummy, since num_classes=1 in training)
    y = torch.zeros(num_samples, dtype=torch.long).to(device)
    
    # 3. Solve ODE from t=0 to t=1 (Euler Method)
    # 100 steps is usually enough for high quality
    steps = 100
    dt = 1.0 / steps
    
    print("Generating synthetic signals...")
    with torch.no_grad():
        for i in range(steps):
            t = torch.ones(num_samples).to(device) * (i / steps)
            
            # Predict velocity (DiT requires class label y)
            v = model(x, t, y)
            
            # Update state
            x = x + v * dt
    
    return x.cpu().numpy()

def plot_comparison(real_data, fake_data):
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    
    # Plot Real
    for i in range(5):
        axs[0, i].plot(real_data[i][0])
        axs[0, i].set_title(f"Real Heartbeat {i+1}")
        axs[0, i].axis('off')
        
    # Plot Fake
    for i in range(5):
        axs[1, i].plot(fake_data[i][0], color='orange')
        axs[1, i].set_title(f"Synthetic Heartbeat {i+1}")
        axs[1, i].axis('off')
        
    plt.suptitle("Real vs Synthetic ECG Signals (Flow Matching)")
    plt.tight_layout()
    plt.savefig("ecg_results.png")
    print("Results saved to ecg_results.png")
    plt.show()

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model (DiT with same parameters as training)
    model = ECG_DiT_1D(
        input_size=256,
        patch_size=16,
        hidden_size=256,
        depth=6,
        num_heads=8,
        num_classes=1,
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load("checkpoints/latest.pt", map_location=DEVICE)["model"])
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model file not found! Please run train.py first.")
        exit()

    # Get some real samples for comparison
    ds = MITBIH_Dataset(records=['100'], samples_per_record=10)
    real_samples = [ds[i].numpy() for i in range(5)]
    
    # Generate fake samples
    fake_samples = generate_ecg(model, num_samples=5, device=DEVICE)
    
    # Plot
    plot_comparison(real_samples, fake_samples)