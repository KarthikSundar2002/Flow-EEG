import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def _group_norm(channels, max_groups=8):
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)

class SelfAttention1D(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.0):
        super().__init__()
        heads = min(num_heads, channels)
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels)
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C) for attention
        x_seq = x.permute(0, 2, 1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq, need_weights=False)
        x_seq = self.norm1(x_seq + attn_out)
        ff_out = self.ff(x_seq)
        x_seq = self.norm2(x_seq + ff_out)
        return x_seq.permute(0, 2, 1)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = _group_norm(out_channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = _group_norm(out_channels)
        
        # Shortcut connection if dims don't match
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + res)

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, dim=64):
        super().__init__()
        
        # Time embedding MLP
        # Need enough capacity to modulate the widest decoder tensor (6 * dim)
        self.time_dim = dim * 6
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(dim),
            nn.Linear(dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

        # Encoder
        self.init_conv = nn.Conv1d(in_channels, dim, 3, padding=1)
        self.down1 = ResidualBlock1D(dim, dim)
        self.attn1 = SelfAttention1D(dim)
        self.down2 = ResidualBlock1D(dim, 2*dim)
        self.attn2 = SelfAttention1D(2*dim)
        self.down3 = ResidualBlock1D(2*dim, 4*dim)
        self.attn3 = SelfAttention1D(4*dim)

        self.pool = nn.MaxPool1d(2)

        # Bottleneck
        self.mid = ResidualBlock1D(4*dim, 4*dim)
        self.mid_attn = SelfAttention1D(4*dim)

        # Decoder
        self.up3 = nn.ConvTranspose1d(4*dim, 2*dim, 2, stride=2)
        self.up_block3 = ResidualBlock1D(6*dim, 2*dim) # 2*dim upsample + 4*dim skip
        self.up_attn3 = SelfAttention1D(2*dim)

        self.up2 = nn.ConvTranspose1d(2*dim, dim, 2, stride=2)
        # 3*dim channels after concat (dim upsampled + 2*dim skip)
        self.up_block2 = ResidualBlock1D(3*dim, dim)
        self.up_attn2 = SelfAttention1D(dim)

        self.up1 = nn.ConvTranspose1d(dim, dim, 2, stride=2) # Restore original resolution
        self.final_block = ResidualBlock1D(2*dim, dim)
        self.up_attn1 = SelfAttention1D(dim)

        self.final_conv = nn.Conv1d(dim, 1, 3, padding=1)

    def forward(self, x, t):
        # x: (Batch, 1, Length)
        # t: (Batch,)
        
        # Time Embedding
        t_emb = self.time_mlp(t) # (B, Time_Dim)
        t_emb = t_emb.unsqueeze(-1) # (B, Time_Dim, 1) for broadcasting

        # Encoding
        x1 = self.init_conv(x)      # (B, dim, L)
        x1 = self.down1(x1 + t_emb[:, :x1.shape[1], :]) # Add time 
        x1 = self.attn1(x1)
        
        x2 = self.pool(x1)
        x2 = self.down2(x2 + t_emb[:, :x2.shape[1], :])
        x2 = self.attn2(x2)
        
        x3 = self.pool(x2)
        x3 = self.down3(x3 + t_emb[:, :x3.shape[1], :])
        x3 = self.attn3(x3)

        # Bottleneck
        x_mid = self.pool(x3)
        x_mid = self.mid(x_mid + t_emb[:, :x_mid.shape[1], :])
        x_mid = self.mid_attn(x_mid)

        # Decoding
        x_up3 = self.up3(x_mid)
        # Handle simple concat (ensure dims match if necessary, but fixed size helps)
        x_up3 = torch.cat([x_up3, x3], dim=1) 
        x_up3 = self.up_block3(x_up3 + t_emb[:, :x_up3.shape[1], :])
        x_up3 = self.up_attn3(x_up3)

        x_up2 = self.up2(x_up3)
        x_up2 = torch.cat([x_up2, x2], dim=1)
        x_up2 = self.up_block2(x_up2 + t_emb[:, :x_up2.shape[1], :])
        x_up2 = self.up_attn2(x_up2)

        x_up1 = self.up1(x_up2)
        x_up1 = torch.cat([x_up1, x1], dim=1) # Skip from x1
        x_up1 = self.final_block(x_up1 + t_emb[:, :x_up1.shape[1], :])
        x_up1 = self.up_attn1(x_up1)

        return self.final_conv(x_up1)