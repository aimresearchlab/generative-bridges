# Response to the experimental review of 17 October 2026

Checked every item against the configs, result JSONs and code on `bridge-reimpl`.
Verdicts below are evidence-backed; where I could not close an item I say so.

The review is accurate and well-hedged. Two findings are wrong in the manuscript and
must be fixed, one is worse than reported, one resolves favourably from data we already
had, and several are documentation rather than defects. Nothing found so far changes a
conclusion.

## Verdicts

| # | Finding | Verdict | Action |
| --- | --- | --- | --- |
| 1a | Optimiser not matched | **Confirmed** | Code fixed; manuscript sentence must change |
| 1b | Improved-DDPM VLB 1000x light | **Confirmed** | Decision needed |
| 2 | Objective ablation confounds sampler | **Confirmed for the quoted pair**; underlying claim survives | Renumber 42.14 -> 38.93; figure fixed |
| 3 | Resume reuse hazard | **Confirmed in code; results clean** | Guard added |
| 4 | Architecture sentence | **Confirmed** | One-word manuscript fix |
| 5 | Velocity-error units | Accepted in principle, not independently verified | Notation fix |
| 6 | DDIM clipping convention | Accepted; affects an explanation, not a number | Low priority |
| 7 | Table/prose contradictions | **2 of 3 confirmed** | Manuscript fixes |
| 8 | FID floor unsupported | **Confirmed, and the number is also wrong** | Measured; manuscript fix |
| 9 | Variance restoration disabled on last step | **Confirmed** | Decision needed |
| 10 | Diagnostics alignment | Accepted | Low priority |

---

## Must fix in the manuscript (zero compute, all verified)

**(7a) The Flow Matching gap claim is false.** `experiments.tex:177` says "the largest
difference in either direction is $1.24$ FID, at twenty evaluations". From the n=50,000
table the differences are:

| NFE | 1 | 2 | 5 | 10 | 20 | 50 | 100 | 200 | 1000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flow − bridge | **−13.77** | +5.25 | −2.35 | +0.20 | +1.24 | +0.60 | +0.18 | −0.06 | −0.80 |

1.24 is the largest gap *from ten evaluations onward*. At one evaluation it is 13.77.
The list of which count each leads at, in the preceding sentence, is correct.

**(7c) The ancestral-sampler claim is false.** `experiments.tex:165` says ancestral is
"the stronger of its two at every evaluation count measured". At two evaluations DDIM is
363.91 against ancestral 364.42, and at one they tie at 364.62. True from five onward.

**(4) The architecture sentence is wrong for CIFAR-10.** `experiments.tex:32` says
"$64$ base channels". `build_unet` uses 64 for MNIST and **128** for CIFAR-10. The stated
parameter counts (7.28M / 52.54M) are correct and I verified them against the
checkpoints, so only the channel figure is wrong.

**(1a) The matched-optimiser claim is false.** `experiments.tex:14-18`: "Every model
shares one network architecture, one optimiser ... so no reported difference is
attributable to architecture or to training setup." In fact:

| trainer | weight decay |
| --- | --- |
| `bridges/train.py` | **0.0** (explicit) |
| `bridges/flowmatch.py` | 0.01 (AdamW default, never passed) |
| `bridges/improved_ddpm.py` | 0.01 (AdamW default, never passed) |
| `bridges/ddpm_baseline.py` | 0.01 (AdamW default, never passed) |

The three baselines were trained with decay the bridge did not get. Weight decay is
mildly harmful on diffusion UNets, so the direction is against us: the baselines were
slightly handicapped. It does not overturn anything — improved denoising still beats the
bridge on CIFAR-10 at every count from twenty up — but the sentence as written is not
true and the code is now public on this branch.

**(8) The FID floor is both unmeasured and about 4x too low.** The body text is honest
("therefore sit near 0.15"), but two table captions say "**measured** floor $0.15$". I
re-ran `calibrate_fid` on both datasets:

```
cifar10 (50,000 train, 10,000 test)      mnist (60,000 train, 10,000 test)
  n= 2,000   holdout 14.599                n= 2,000   holdout 5.190
  n= 5,000   holdout  5.848                n= 5,000   holdout 3.113
  n=10,000   holdout  3.152                n=10,000   holdout 0.984
```

These match the body text exactly, so the measured part is sound. The test split caps n
at 10,000, so n=50,000 cannot be measured this way — the review is right about that. But
the extrapolation is also wrong. On CIFAR-10 the floor follows `20470 * n^-0.955` with a
maximum log residual of 0.028 — essentially `C/n`, with `n x floor` constant to within 8%
across the three points. That gives **0.67 at n=50,000**, not 0.15.

On MNIST the same fit gives 0.23 but the points do not follow a power law (`n x floor`
ranges 9,840 to 15,565, log residual 0.29), so **no defensible extrapolation exists for
MNIST** and the caption should not quote one.

This does not touch the resolution claims. The floor is a common offset on every row; the
"roughly seven standard deviations" statement rests on the simulation-seed spread
(0.05-0.09 at fifty evaluations, giving 0.60/0.09 = 6.7), which is unaffected. Evidence
saved to `runs/fid_floors.json`.

---

## Resolved from existing data, favourably

**(2) The quoted pair does confound the sampler; the effect itself does not.**
The appendix pair is `bridge`-sampled 42.14 against `euler`-sampled 23.23, so it varies
objective and sampler together -- the review is right. Matched on euler at the same
n=2000 the endpoint model gives **38.93**, so the honest number is 38.93 -> 23.23.
The effect is robust under matching: The `objsampler_n10k.json` pair
holds the sampler fixed. CIFAR-10, n=10,000, same gamma, time scaling and protocol:

| sampler | NFE | endpoint | drift | effect of the objective alone |
| --- | --- | --- | --- | --- |
| euler | 10 | 39.35 | 20.87 | **−18.48** |
| euler | 50 | 28.52 | 11.86 | **−16.66** |
| euler_det | 10 | 69.76 | 111.67 | +41.91 |

Under the sampler actually used throughout, the objective correction is worth 16-18 FID
on its own — same direction and comparable magnitude to the quoted 42.14 -> 23.23 at
n=2,000. The claim stands.

The figure is a fair criticism though: `samples_figure.py` loads `bridge_0050.png` for
endpoint and `euler_0050.png` for drift, so the *illustration* varies both. Either
regenerate with a matched sampler or state in the caption that both differ.

The `euler_det` sign flip is worth noting somewhere — the drift objective is markedly
worse under the deterministic map — but it is a separate observation, not a problem.

**(3) The hazard is real; the published numbers are not affected.** `evaluate.py` keyed
completed records on `(sampler, n_steps)` alone, so pointing an existing output path at a
different checkpoint would silently inherit stale cells. I audited every
`fid_n50k*.json`: **12 files, 139 records, all `n_samples=50000` and `ckpt_step=40000`.**
No mixed provenance anywhere in the published tables.

---

## Fixed in code on this branch

- `--weight_decay` added to `flowmatch`, `improved_ddpm` and `ddpm_baseline`, defaulting
  to **0.0** to match `train.py` instead of inheriting AdamW's 0.01. Recorded in each
  run's `config.json`.
- `evaluate.py` now refuses to resume a results file whose records were measured under a
  different `ckpt_step`, `n_samples` or `eta`, naming the mismatch, instead of silently
  skipping those cells. Verified: blocks the mismatched case, allows legitimate resume.
- `runs/fid_floors.json` records the floor measurements and fits.
- The 20 per-run `config.json` manifests are now tracked (pushed earlier today), which
  answers the review's closing request for configs and a provenance trail.

---

## Decisions needed

**(1b) Improved-DDPM VLB weight.** `training_loss` samples one timestep and returns
`l_simple + lam * l_vlb` with `lam=1e-3`, with no rescaling by `T`. The reference applies
coefficient 1 to the single sampled term, which is what `lambda=0.001` on the *full*
T-term sum reduces to at T=1000. Ours is therefore **1000x light**, so the learned
reverse variance is effectively untrained. It also trains the VLB term with `clip=True`
where the reference uses `clip_denoised=False`.

Against fixing it: this baseline already beats us on CIFAR-10 (6.04 vs 9.83 at fifty) and
is far better than the submitted version (5.60 vs 29.16 at 1000), so it is not a crippled
baseline and the deviation did not buy us anything. Options are (i) disclose and keep,
(ii) retrain improved DDPM on both datasets with the reference weighting. Retraining is
~3.9 h on CIFAR-10 plus ~1 h on MNIST, so it is affordable; the risk is that the baseline
gets *better* and the parity story weakens further.

**(1a) Do we retrain the baselines with matched weight decay?** MNIST is where the bridge
leads, so that is where the handicap matters most, and MNIST is cheap (~1 h per model,
three baselines). CIFAR-10 is ~3.9 h per model. My suggestion: retrain the MNIST
baselines, disclose for CIFAR-10, since the CIFAR-10 conclusion does not depend on it.

**(9) Variance restoration.** `bridge.py:282` forces `eta=0` on the last `bridge` step
regardless of `var_profile`, so the final-step correction never fires. Either describe the
implemented variant honestly or apply the full correction and re-sample. Re-sampling only,
no retraining.

---

## Not worth doing now

Items 5, 6 and 10 are all real but none changes a reported number. Item 5 is a notation
convention (define `v_s = T u_t` and label consistently). Item 6 affects the *explanation*
of DDIM degradation at high step counts, not the measurement. Item 10 asks the 1-D
diagnostic to match the production sampler's output rule, which is worth doing before that
diagnostic is cited as validating the production path, but not before.

---

## One disagreement

The review's framing of item 8 as "if 0.15 is an estimate, label it accordingly"
understates it. The body text already labels it as an inference; the error is that the
inference is numerically wrong by a factor of four and two captions call it measured.
Relabelling alone would preserve a wrong number.
