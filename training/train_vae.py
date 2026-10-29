import os, csv, math
from fractions import Fraction
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Config
# =========================
ROOT = r"/Users/hayakzan/Desktop/JUCE/NEW PROJ/data"       # dataset root 
MANIFEST = os.path.join(ROOT, "manifest.csv")
PRE_ROLL_SEC = 0.20                                         # each file has 0.2s lead-in
STEADY_CHUNK_SEC = 0.5                                      # chunk length 
CYCLE_LEN = 1024                                            # single-cycle table length
LATENT_DIM = 3
BATCH = 256
EPOCHS = 30 
LR = 1e-3                                                   # learning rate = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Utilities
# =========================
# stereo files arrive as a 2D array (samples x chans), so average them, leave mono files as is 
def to_mono(x):
    return x if x.ndim == 1 else x.mean(axis=1)

# turn decimal into fraction with denom at most 1000. resample_poly below needs whole numbers
def rational_approx(alpha, max_den=1000):
    frac = Fraction(alpha).limit_denominator(max_den)
    return frac.numerator, frac.denominator

# every example has to be the same size (1024 samples). Down/upsample as needed
# factor = 1024/samples per cycle
def resample_factor(x, factor):
    up, down = rational_approx(factor)
    return resample_poly(x, up, down) # up & down are numerator and denominator. 
                                     # The resampling is a ratio denoted by them


# eps avoids dividing by zero on silence.
def zero_dc_and_norm(x, target=0.9, eps=1e-9):
    x = x - x.mean() # remove DC, 
    peak = np.max(np.abs(x)) + eps #then scale so the highest peak is 0.9. eps helps avoid division by zero
    return x * (target / peak)

def positive_zero_cross_shift(cycle):
    """Return circular shift so sample 0 is a +slope zero crossing (linear interp)."""
    x = cycle
    idx = None
    for i in range(len(x)-1):
        if x[i] < 0 and x[i+1] >= 0:
            # frac between i and i+1
            di = x[i+1] - x[i]
            if abs(di) < 1e-12:
                continue
            s = -x[i] / di
            idx = i + s
            break
    if idx is None:
        return cycle  # fallback
    # fractional circular shift using linear interpolation
    k = int(math.floor(idx))
    frac = idx - k
    rolled = np.roll(x, -k)
    if frac > 1e-9:
        # linear between samples
        nxt = np.roll(rolled, -1)
        rolled = (1-frac)*rolled + frac*nxt
    return rolled

def extract_cycles_phase_locked(x, fs, f0, pre_roll=0.2, chunk_sec=0.5, cycle_len=1024):
    """Return a list of cleaned single cycles (np.float32, length cycle_len)."""
    start = int(round(pre_roll * fs))
    x = x[start:]
    # choose a steady chunk
    N = int(round(chunk_sec * fs))
    if len(x) < N + 8:
        N = len(x) - 8
    chunk = x[:N].astype(np.float64)

    # number of cycles K in chunk, and resample to make it exactly K periods
    K = max(1, int(round(chunk_sec * f0)))
    alpha = K / (chunk_sec * f0)                      # ≈ 1.0
    chunk_r = resample_factor(chunk, alpha)

    # split into K cycles (evenly), then each → 1024, align, normalize
    N2 = len(chunk_r)
    Lc = N2 / K
    out = []
    for k in range(K):
        a = k * Lc
        b = (k + 1) * Lc
        # fractional slice by linear interpolation onto integer grid
        idx = np.linspace(a, b, int(round(Lc)), endpoint=False)
        i0 = np.floor(idx).astype(int)
        frac = idx - i0
        i1 = np.clip(i0 + 1, 0, len(chunk_r)-1)
        seg = (1-frac) * chunk_r[i0] + frac * chunk_r[i1]

        # resample to fixed cycle_len (FFT method is fine here)
        seg = resample_factor(seg, cycle_len / len(seg))
        seg = positive_zero_cross_shift(seg)
        seg = zero_dc_and_norm(seg).astype(np.float32)
        out.append(seg)
    return out

# =========================
# Dataset
# =========================
WAVE_TO_ONEHOT = {"triangle": (1,0,0), "saw": (0,1,0), "pulse": (0,0,1)}

class CyclesDataset(Dataset):
    def __init__(self, manifest_csv, root, include_noise=False):
        df = pd.read_csv(manifest_csv)
        if not include_noise:
            df = df[df["wave"].str.lower().isin(["triangle","saw","pulse"])]
        df = df.reset_index(drop=True)
        self.rows = df.to_dict("records")
        self.root = root
        self.cache = []  # list of (cycle, cond)
        self._build_cache()

    def _build_cache(self):
        for r in self.rows:
            path = os.path.join(self.root, r["relpath"])
            if not os.path.exists(path): 
                continue
            x, fs = sf.read(path, always_2d=False)
            x = to_mono(x)
            cycles = extract_cycles_phase_locked(
                x, fs, r["freq_hz"], pre_roll=PRE_ROLL_SEC,
                chunk_sec=STEADY_CHUNK_SEC, cycle_len=CYCLE_LEN
            )
            # conditioning vector
            pw = 0.0 if (pd.isna(r.get("pw_norm")) or r.get("pw_norm")=="" ) else float(r["pw_norm"])
            vel = float(r["velocity"]) / 127.0
            lf0 = math.log(max(1e-6, float(r["freq_hz"])))
            wave_oh = np.array(WAVE_TO_ONEHOT[r["wave"].lower()], dtype=np.float32)
            cond = np.concatenate([[pw, lf0, vel], wave_oh]).astype(np.float32)

            for c in cycles:
                self.cache.append((c, cond))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        c, cond = self.cache[idx]
        # (1, 1024) tensor, and cond vector
        return torch.from_numpy(c[None, :]), torch.from_numpy(cond)

# =========================
# Model (conditional Conv1D VAE)
# =========================
class Encoder(nn.Module):
    def __init__(self, cond_dim, zdim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=2, padding=4), nn.LeakyReLU(0.2, True),
            nn.Conv1d(32,64, 9, stride=2, padding=4), nn.LeakyReLU(0.2, True),
            nn.Conv1d(64,128,9, stride=2, padding=4), nn.LeakyReLU(0.2, True),
            nn.Conv1d(128,128,9, stride=2, padding=4), nn.LeakyReLU(0.2, True),
        )  # 1024 -> 64
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128 + cond_dim, 128)
        self.mu = nn.Linear(128, zdim)
        self.logvar = nn.Linear(128, zdim)

    def forward(self, x, c):
        h = self.conv(x)
        h = self.gap(h).squeeze(-1)              # (B,128)
        h = torch.cat([h, c], dim=1)             # (B,128+cond)
        h = torch.relu(self.fc(h))
        return self.mu(h), self.logvar(h)

class Decoder(nn.Module):
    def __init__(self, cond_dim, zdim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(zdim + cond_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 128*64),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128,128,9,stride=2,padding=4,output_padding=1), nn.ReLU(True),
            nn.ConvTranspose1d(128,64, 9,stride=2,padding=4,output_padding=1), nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, 9,stride=2,padding=4,output_padding=1), nn.ReLU(True),
            nn.ConvTranspose1d(32, 1,  9,stride=2,padding=4,output_padding=1),
            nn.Tanh(),
        )

    def forward(self, z, c):
        h = torch.cat([z, c], dim=1)
        h = self.fc(h)
        h = h.view(h.size(0), 128, 64)
        x = self.deconv(h)
        return x

class VAE(nn.Module):
    def __init__(self, cond_dim=6, zdim=LATENT_DIM):
        super().__init__()
        self.enc = Encoder(cond_dim, zdim)
        self.dec = Decoder(cond_dim, zdim)

    def forward(self, x, c):
        mu, logvar = self.enc(x, c)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        xhat = self.dec(z, c)
        return xhat, mu, logvar

def mrstft_loss(xhat, x):
    # lightweight 2-scale STFT L1
    def mag(x, n_fft):
        win = torch.hann_window(n_fft, device=x.device)
        X = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=n_fft//4,
                       window=win, return_complex=True)
        return torch.abs(X)
    L = torch.mean(torch.abs(mag(xhat,256)-mag(x,256))) + \
        torch.mean(torch.abs(mag(xhat,512)-mag(x,512)))
    return L

def loop_continuity_loss(x):
    x0 = x[:,:,0]
    x1 = x[:,:,-1]
    dx0 = x[:,:,1]-x[:,:,0]
    dx1 = x[:,:,-1]-x[:,:,-2]
    return ((x0-x1)**2 + (dx0-dx1)**2).mean()

# =========================
# Train
# =========================
def train():
    ds = CyclesDataset(MANIFEST, ROOT, include_noise=False)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)
    cond_dim = 3 + 3  # [pw, logf0, vel] + wave one-hot
    net = VAE(cond_dim=cond_dim, zdim=LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    print(f"Samples (cycles): {len(ds)}  |  Device: {DEVICE}")

    for epoch in range(1, EPOCHS+1):
        net.train()
        tot = 0.0
        for x, c in dl:
            x = x.to(DEVICE)
            c = c.to(DEVICE)
            xhat, mu, logvar = net(x, c)

            rec = 0.7*mrstft_loss(xhat, x) + 0.3*torch.mean(torch.abs(xhat-x))
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loop = 0.05 * loop_continuity_loss(xhat)

            loss = rec + kl + loop
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"epoch {epoch:02d}  loss {tot/len(dl):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(net.state_dict(), "checkpoints/sid_vae.pt")
    print("Saved -> checkpoints/sid_vae.pt")

if __name__ == "__main__":
    train()
