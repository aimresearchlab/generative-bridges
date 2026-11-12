"""Does a Gaussian bridge actually waste path mass on non-physical states?

The claim we would want to make about non-Gaussian drivers is that the data
lives on a constrained set and a Gaussian bridge leaves it, spending capacity
and sampling steps in states the physics forbids. That claim is worth nothing
unless the constraint BINDS: if a Gaussian path rarely leaves the feasible set,
or if projecting back costs nothing, there is no story.

This measures it on real speech, for three representations with three different
supports:

    linear magnitude   support [0, inf)      a subordinator (gamma) bridge fits
    ratio mask         support [0, 1]        a Jacobi / Beta bridge fits
    phase              support [-pi, pi)     circular; a wrapped bridge fits

For each, the Gaussian bridge between the true endpoints is
    xi_s = (1-s) x + s y + gamma sqrt(s(1-s)) eps
and we report how much of its path mass falls outside the feasible set, how far
outside, and what projecting back costs against a reference the task cares
about. CPU only.
"""

from __future__ import annotations

import argparse, json, math, os

import torch
import torchaudio


def load_speech(root: str, n_clips: int = 24, sr: int = 16000):
    """A small real-speech sample.

    YESNO is ~4 MB of real recorded speech. Its wavs are plain 16-bit PCM, so
    they are read with the standard library rather than pulling in a codec
    backend for four megabytes of audio.
    """
    import glob, wave

    torchaudio.datasets.YESNO(root=root, download=True)   # fetch + extract only
    files = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    if not files:
        raise RuntimeError(f"no wavs found under {root}")
    out = []
    for f in files[:n_clips]:
        with wave.open(f, "rb") as wf:
            fs, nch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        if sw != 2:
            continue
        w = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
        if nch > 1:
            w = w.view(-1, nch).mean(1)
        if fs != sr:
            w = torchaudio.functional.resample(w, fs, sr)
        out.append(w)
    return out


def stft(x, n_fft=512, hop=128):
    return torch.stft(x, n_fft=n_fft, hop_length=hop,
                      window=torch.hann_window(n_fft), return_complex=True)


def bridge_path(x, y, gamma, n_s=19, seed=0):
    """Gaussian bridge states at n_s interior times, given the two endpoints."""
    g = torch.Generator().manual_seed(seed)
    ss = torch.linspace(0, 1, n_s + 2)[1:-1]
    return ss, torch.stack([
        (1 - s) * x + s * y + gamma * math.sqrt(float(s * (1 - s)))
        * torch.randn(x.shape, generator=g)
        for s in ss])


def report(name, states, lo=None, hi=None, circular=False, scale=1.0):
    """Fraction of entries outside the feasible set, and how far outside."""
    n = states.numel()
    if circular:
        out = (states < -math.pi) | (states >= math.pi)
        excess = torch.where(out, states.abs() - math.pi, torch.zeros_like(states))
    else:
        below = torch.zeros_like(states) if lo is None else (lo - states).clamp(min=0)
        above = torch.zeros_like(states) if hi is None else (states - hi).clamp(min=0)
        excess = below + above
        out = excess > 0
    frac = float(out.float().mean())
    viol = excess[out]
    return {
        "representation": name,
        "frac_outside": frac,
        "mean_excess_rel": float(viol.mean() / scale) if viol.numel() else 0.0,
        "max_excess_rel": float(viol.max() / scale) if viol.numel() else 0.0,
        "n_entries": n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./data/speech")
    p.add_argument("--gamma", type=float, default=0.3162277660168379)
    p.add_argument("--snr_db", type=float, default=5.0)
    p.add_argument("--n_clips", type=int, default=24)
    p.add_argument("--out", default="runs/speech_premise.json")
    a = p.parse_args()

    torch.manual_seed(0)
    os.makedirs(a.root, exist_ok=True)
    clips = load_speech(a.root, a.n_clips)
    print(f"{len(clips)} real speech clips, "
          f"{sum(c.numel() for c in clips)/16000:.1f} s total\n")

    mags_c, mags_n, masks, phases = [], [], [], []
    for w in clips:
        Sc = stft(w)
        noise = torch.randn_like(w) * (w.pow(2).mean().sqrt()
                                       * 10 ** (-a.snr_db / 20))
        Sn = stft(w + noise)
        mags_c.append(Sc.abs()); mags_n.append(Sn.abs())
        masks.append((Sc.abs() / Sn.abs().clamp(min=1e-8)).clamp(0, 1))
        phases.append(torch.angle(Sc))

    Mc = torch.cat([m.flatten() for m in mags_c])
    Mn = torch.cat([m.flatten() for m in mags_n])
    Mk = torch.cat([m.flatten() for m in masks])
    Ph = torch.cat([m.flatten() for m in phases])

    rows = []

    # 1. linear magnitude, noisy -> clean.  Support [0, inf).
    scale = float(Mn.std())
    Mn_n, Mc_n = Mn / scale, Mc / scale          # unit-variance, as a model would see
    _, st = bridge_path(Mn_n, Mc_n, a.gamma)
    rows.append(report("linear magnitude (noisy->clean)", st, lo=0.0, scale=1.0))

    # 2. ratio mask, all-ones -> ideal mask.  Support [0, 1].
    _, st = bridge_path(torch.ones_like(Mk), Mk, a.gamma)
    rows.append(report("ratio mask (ones->ideal)", st, lo=0.0, hi=1.0, scale=1.0))

    # 3. phase, random -> true.  Support [-pi, pi), circular.
    rnd = (torch.rand(Ph.shape) * 2 - 1) * math.pi
    _, st = bridge_path(rnd, Ph, a.gamma * math.pi)
    rows.append(report("phase (random->true)", st, circular=True, scale=math.pi))

    # 4. control: log-magnitude, which is what most speech work uses. Unconstrained.
    Lc, Ln = torch.log(Mc.clamp(min=1e-5)), torch.log(Mn.clamp(min=1e-5))
    sc = float(Ln.std())
    _, st = bridge_path(Ln / sc, Lc / sc, a.gamma)
    rows.append(report("log magnitude (control, unconstrained)", st,
                       lo=None, hi=None, scale=1.0))

    print(f"{'representation':<40}{'% outside':>11}{'mean excess':>14}{'max excess':>13}")
    print("-" * 78)
    for r in rows:
        print(f"{r['representation']:<40}{100*r['frac_outside']:>10.2f}%"
              f"{r['mean_excess_rel']:>14.4f}{r['max_excess_rel']:>13.3f}")
    print("\n(excess is relative to the representation's own scale: 1 sd for")
    print(" magnitude, the full [0,1] range for masks, pi for phase)")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"gamma": a.gamma, "snr_db": a.snr_db, "rows": rows},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
