import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FlashSelfAttention(nn.Module):
    """
    Multi-head self-attention implemented with scaled_dot_product_attention,
    enabling FlashAttention kernels when available (PyTorch 2+ with CUDA).
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, n, 3, self.num_heads, self.head_dim).unbind(dim=2)
        q = q.transpose(1, 2)  # (B, heads, N, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)  # FlashAttention when backend supports it
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, n, c)
        return self.proj(attn_out)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :return: an (N, dim) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiTBlock(nn.Module):
    """
    A DiT block with Adaptive Layer Norm (AdaLN) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = FlashSelfAttention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        
        # AdaLN Modulation: Regresses shift (gamma) and scale (beta) from the condition embedding
        # We output 6 parameters: gamma1, beta1, alpha1, gamma2, beta2, alpha2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # x: (Batch, Seq_Len, Hidden)
        # c: (Batch, Hidden) - Condition embedding (Time + Class)
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # 1. Attention Block with AdaLN
        x_norm1 = self.norm1(x)
        # Modulate
        x_norm1 = x_norm1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out = self.attn(x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # 2. MLP Block with AdaLN
        x_norm2 = self.norm2(x)
        # Modulate
        x_norm2 = x_norm2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x

class ECG_DiT_1D(nn.Module):
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        hidden_size=256,
        depth=6,
        num_heads=8,
        num_classes=5  # MIT-BIH Classes (N, S, V, F, Q)
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = input_size // patch_size
        self.hidden_size = hidden_size

        # 1. Patch Embeddings (Conv1d acts as linear projection of patches)
        self.x_embedder = nn.Conv1d(1, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        # 2. Positional Embeddings (Learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # 3. Condition Embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Embedding(num_classes, hidden_size) # Class embedding

        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # 5. Final Layer (Depatchify)
        self.final_layer = FinalLayer(hidden_size, patch_size, 1)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize zero for the "gate" and "scale" to start as identity function
        # This makes training very stable at the beginning
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size * C) -> (N, C, L)
        """
        c = 1 # Output channels
        n, t, _ = x.shape
        x = x.reshape(n, t, self.patch_size, c)
        x = torch.einsum('ntpc->nctp', x)
        x = x.reshape(n, c, t * self.patch_size)
        return x

    def forward(self, x, t, y):
        """
        x: (Batch, 1, 256)
        t: (Batch,) Time steps
        y: (Batch,) Class labels (0-4)
        """
        # 1. Patchify input: (B, 1, 256) -> (B, Hidden, Num_Patches)
        x = self.x_embedder(x) 
        x = x.transpose(1, 2) # (B, Num_Patches, Hidden) for Transformer
        
        # 2. Add Position Embedding
        x = x + self.pos_embed

        # 3. Combine Time and Class embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb # Combine conditions

        # 4. Apply Transformer Blocks
        for block in self.blocks:
            x = block(x, c)

        # 5. Final Layer
        x = self.final_layer(x, c) # (B, Num_Patches, Patch_Size)
        
        # 6. Unpatchify to signal
        x = self.unpatchify(x) # (B, 1, 256)
        
        return x