#!/usr/bin/env python3
"""
## Validating the Latent Morph

To verify that the decoder is learning a smooth, continuous manifold rather than acting as a discrete waveform crossfader, `tools/inspect_model.py` evaluates interpolation behavior across steps:

* **Nonlinearity ($\text{nl}$):** Computes the relative geometric distance between the decoded output and a linear baseline in sample space:
  $$ \text{nl} = \frac{\|d(t) - ((1-t)d_0 + td_1)\|}{\|d_1 - d_0\|} $$
  A value of `0` indicates a trivial linear crossfade; non-zero values confirm the decoder is generating continuous hybrid timbres in-between.
* **Spectral Envelope Distance:** Repeats the metric on phase-insensitive FFT harmonic magnitudes to track timbral progression.
* **Step-Size Analysis:** Measures inter-step Euclidean distances ($\Delta t$) to detect latent space discontinuities or abrupt structural "jumps" across the morph trajectory.

Run the diagnostic against your weights with:
```bash
python tools/inspect_model.py Models/sid_vae_decoder.onnx --a tri --b saw --plot morph.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

DEFAULT_ORDER = ["pw_norm", "log_f0", "vel01", "wave_tri", "wave_saw", "wave_pulse"]
WAVES = {"tri": 0, "saw": 1, "pulse": 2}


def load_meta(path):
    meta = {"latent_dim": 3, "cond_dim": 6, "cond_order": DEFAULT_ORDER}
    if path and Path(path).exists():
        meta.update(json.loads(Path(path).read_text()))
    return meta


def build_cond(meta, pw, f0, vel, weights):
    vals = {
        "pw_norm": pw,
        "vel01": vel,
        "wave_tri": weights[0],
        "wave_saw": weights[1],
        "wave_pulse": weights[2],
    }
    lf0 = math.log(max(1.0, f0))
    if "log_f0_mean" in meta and "log_f0_std" in meta and meta["log_f0_std"]:
        lf0 = (lf0 - meta["log_f0_mean"]) / meta["log_f0_std"]
    vals["log_f0"] = lf0
    unknown = [k for k in meta["cond_order"] if k not in vals]
    return np.array([vals.get(k, 0.0) for k in meta["cond_order"]], dtype=np.float32), unknown


class Decoder:
    def __init__(self, path, latent_dim, cond_dim):
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inputs = self.sess.get_inputs()
        self.L, self.C = latent_dim, cond_dim
        self.out_name = self.sess.get_outputs()[0].name

        def per_item(shape):
            dims = [d for d in shape[1:]] if len(shape) > 1 else list(shape)
            if any(not isinstance(d, int) or d <= 0 for d in dims):
                return None
            return int(np.prod(dims)) if dims else None

        if len(self.inputs) == 2:
            sizes = [per_item(i.shape) for i in self.inputs]
            names = [i.name.lower() for i in self.inputs]
            zi = 0
            if latent_dim != cond_dim and None not in sizes:
                zi = 0 if sizes[0] == latent_dim else 1
            elif ("cond" in names[0] or "label" in names[0]) and not ("cond" in names[1] or "label" in names[1]):
                zi = 1
            self.z_in, self.c_in = self.inputs[zi], self.inputs[1 - zi]
            self.layout = "two"
        elif len(self.inputs) == 1:
            self.layout = "concat"
        else:
            raise SystemExit(f"expected 1 or 2 inputs, got {len(self.inputs)}")

    def describe(self):
        ins = ", ".join(f"{i.name}{list(i.shape)}:{i.type}" for i in self.inputs)
        outs = ", ".join(f"{o.name}{list(o.shape)}:{o.type}" for o in self.sess.get_outputs())
        return f"inputs: {ins}\noutputs: {outs}\nlayout: {self.layout}"

    def __call__(self, z, cond):
        z = np.asarray(z, np.float32).reshape(1, -1)
        cond = np.asarray(cond, np.float32).reshape(1, -1)
        if self.layout == "two":
            feeds = {self.z_in.name: z, self.c_in.name: cond}
        else:
            feeds = {self.inputs[0].name: np.concatenate([z, cond], axis=1)}
        return self.sess.run([self.out_name], feeds)[0].reshape(-1)


def harmonic_mags(x, n_harm=64):
    spec = np.abs(np.fft.rfft(x - x.mean())) / len(x)
    return spec[1 : n_harm + 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--meta", default=None, help="sid_vae_meta.json (default: next to model, then one level up)")
    ap.add_argument("--a", default="tri", choices=WAVES)
    ap.add_argument("--b", default="saw", choices=WAVES)
    ap.add_argument("--f0", type=float, default=220.0)
    ap.add_argument("--pw", type=float, default=0.5)
    ap.add_argument("--vel", type=float, default=1.0)
    ap.add_argument("--z", type=float, nargs="*", default=None, help="latent vector (default zeros)")
    ap.add_argument("--steps", type=int, default=21)
    ap.add_argument("--plot", default=None, help="save a PNG of the morph (needs matplotlib)")
    args = ap.parse_args()

    model = Path(args.model)
    meta_path = args.meta
    if meta_path is None:
        for cand in (model.parent / "sid_vae_meta.json", model.parent.parent / "sid_vae_meta.json"):
            if cand.exists():
                meta_path = cand
                break
    meta = load_meta(meta_path)
    print(f"meta: {meta_path or '(defaults)'} -> latent_dim={meta['latent_dim']} cond_dim={meta['cond_dim']}")
    print(f"cond_order: {meta['cond_order']}")

    dec = Decoder(model, meta["latent_dim"], meta["cond_dim"])
    print(dec.describe())

    z = np.zeros(meta["latent_dim"], np.float32)
    if args.z:
        z[: len(args.z)] = args.z[: meta["latent_dim"]]

    ts = np.linspace(0.0, 1.0, args.steps)
    a, b = WAVES[args.a], WAVES[args.b]
    tables = []
    for t in ts:
        w = [0.0, 0.0, 0.0]
        w[a] += 1.0 - t
        w[b] += t
        cond, unknown = build_cond(meta, args.pw, args.f0, args.vel, w)
        tables.append(dec(z, cond))
    if unknown:
        print(f"WARNING: cond names the plugin doesn't fill (set to 0): {unknown}")

    tables = np.stack(tables)
    print(f"output length: {tables.shape[1]} samples, range [{tables.min():.3f}, {tables.max():.3f}]")
    if not np.isfinite(tables).all():
        sys.exit("model produced NaN/Inf")

    d0, d1 = tables[0], tables[-1]
    span = np.linalg.norm(d1 - d0) + 1e-12
    h0, h1 = harmonic_mags(d0), harmonic_mags(d1)
    hspan = np.linalg.norm(h1 - h0) + 1e-12

    print(f"\nmorph {args.a} -> {args.b} at f0={args.f0} Hz, pw={args.pw}, z={z.tolist()}")
    print("   t    nonlinearity  spectral")
    for t, d in zip(ts, tables):
        lin = (1 - t) * d0 + t * d1
        hlin = (1 - t) * h0 + t * h1
        nl = np.linalg.norm(d - lin) / span
        sp = np.linalg.norm(harmonic_mags(d) - hlin) / hspan
        print(f"  {t:4.2f}  {nl:10.3f}  {sp:9.3f}")

    steps = np.linalg.norm(np.diff(tables, axis=0), axis=1)
    med = np.median(steps) + 1e-12
    worst = int(np.argmax(steps))
    print(f"\nstep size: median {med:.4f}, max {steps.max():.4f} "
          f"({steps.max() / med:.1f}x median, between t={ts[worst]:.2f} and t={ts[worst + 1]:.2f})")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            sys.exit("matplotlib not installed; pip install matplotlib")
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        idx = np.linspace(0, len(ts) - 1, 6).astype(int)
        x = np.arange(tables.shape[1]) / tables.shape[1]
        for i in idx:
            axes[0].plot(x, tables[i], label=f"t={ts[i]:.2f}", lw=1.2)
        axes[0].set_title(f"decoded cycle, {args.a} -> {args.b}")
        axes[0].legend(fontsize=8)
        axes[1].plot(ts[1:], steps, marker="o")
        axes[1].set_title("step size between neighbouring morph positions")
        axes[1].set_xlabel("t")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"plot saved to {args.plot}")


if __name__ == "__main__":
    main()
