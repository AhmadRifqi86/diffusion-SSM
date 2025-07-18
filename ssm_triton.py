import torch
import triton
import triton.language as tl

@triton.jit
def parallel_scan_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    # Inclusive scan (sequential within block, parallel across blocks)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(BLOCK):
        if i == 0:
            acc[i] = x[i]
        else:
            acc[i] = acc[i-1] + x[i]
    tl.store(Y_ptr + offs, acc, mask=mask)

def parallel_scan(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous(), "Input must be contiguous"
    assert x.dtype == torch.float32, "Only float32 supported for now"
    N = x.numel()
    y = torch.empty_like(x)
    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(N, BLOCK),)
    parallel_scan_kernel[grid](x, y, N, BLOCK)
    return y

class SSM(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (batch, seq, hidden)
        x = self.linear(x)
        b, s, h = x.shape
        x_flat = x.reshape(-1)
        y_flat = parallel_scan(x_flat)
        y = y_flat.reshape(b, s, h)
        return y 


class VAEEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.mu_head = nn.Conv2d(512, latent_dim, 1)
        self.logvar_head = nn.Conv2d(512, latent_dim, 1)
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=512, out_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.decoder(x)

class CLIPEncoder(nn.Module):
    def __init__(self, vocab_size=50000, embed_dim=512, max_len=77):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(max_len, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=8, dim_feedforward=2048),
            num_layers=6
        )
        self.ln_final = nn.LayerNorm(embed_dim)
        
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) + self.pos_embedding[:seq_len]
        x = x.transpose(0, 1)  # (seq_len, batch, embed_dim)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch, seq_len, embed_dim)
        x = self.ln_final(x)
        return x.mean(dim=1)  # Global average pooling