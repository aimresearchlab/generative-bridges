# Response to the rebuttal numerical and prose review (20 October)

24 October 2026 — reply to `rebuttal-edit-self-review-2026-10-20.txt`

Berkan, thanks. I checked all twelve items against `paper/sections`, `paper/tables`,
`runs/fid_floors.json`, `runs/metrics.log`, the other run records and the figure
PDFs. Eleven are real mismatches between the response and the manuscript. Item 7
isn't: the response and the appendix quote profiles from two different datasets,
so swapping in the appendix numbers would introduce an error. Below is the
corrected wording for each paragraph, ready to paste into the response source
(which isn't in this repo, so I haven't applied it). Text not quoted here stays as
you have it.

*Update, same day:* I've now also done the three checks I'd first left open (density,
"largest effect", "factor of three"). The results are under "Checks on the
unchanged claims" below, and items 2, 7, 9 and 11 are revised to match.

## Verification summary

| Item | Response said | Source of truth | Verdict |
|---|---|---|---|
| 1, 3, 8 | only process and sampler differ | `experiments.tex:18`, `appendix_experiments.tex:301`: weight decay 0 vs 1e-2 | Fix, short exception |
| 2, 9 | 42.14 → 23.23 | `appendix_experiments.tex:65`: 38.93 → 23.23, both Euler–Maruyama, n=2000. 42.14 was the exact-posterior-sampled entry (errata item 7) | Fix |
| 4 | most expensive to FID ≤10 | `tab:cost`: FID 10 is 47 / 65 / 23, FID 9 is 151 / 157 / 27 | Fix |
| 5 | nine step counts for every method | `tab:cifar`: DDIM has dashes from 20 evaluations | Fix |
| 6 | floors at four sample counts | `runs/fid_floors.json`: three counts (2000, 5000, 10000) per dataset; no fourth exists | Fix, minimal |
| 7 | 0.24565 / 0.25315 / 41 | 0.24565 and 41 are right for CIFAR-10 (`runs/variance_restore.json`); the appendix's 0.21346 / 0.21201 / 21 is MNIST (`runs/diagnose_timecond.json`); 0.25315 is the wrong statistic | Keep CIFAR-10, fix the comparator, name the dataset |
| 10 | one to two hundred evaluations | figure labels: CIFAR-10 2, 5, 10, 50, 200; MNIST 2, 5, 10, 50, 1000 | Fix, per dataset |
| 11 | table reports density | `tab:coverage` has no density column (density *was* computed: `runs/metrics.log`) | Fix, plus manuscript follow-up |
| 12 | leads below twenty | `tab:cifar`: at 1 evaluation 385.98 vs 364.62; leads at 2, 5, 10 | Fix |

On item 6 you asked whether a fourth floor measurement might exist somewhere else. It
doesn't: `runs/fid_floors.json` is the only floor record (tracked in `9bfa630`) and it
holds three counts per dataset. I agree the expanded AFTER says too much, so the fix
below only deletes the count.

## Corrected paragraphs

### Item 1: opening description

> The evaluation was rebuilt from first principles: one architecture, one
> optimiser, one EMA decay, one batch size, one training budget and one FID
> implementation across every row. Apart from weight decay (zero for the
> random-bridge runs, AdamW's default $10^{-2}$ for the baselines), only the
> generative process and its sampler differ.

### Item 2: first objective-ablation comparison

> Regressing the drift instead lowers CIFAR-10 FID at $50$ network evaluations from
> $38.93$ to $23.23$ at the same $\gamma$, sampler, architecture and budget, the
> largest effect of any training choice. It also explains the step-count behaviour the
> referees asked about: refining the sampling grid spends more time integrating
> through exactly the region the uniform objective discounted, which is why
> additional steps made the original configuration worse.

I dropped "$n=2000$" because the appendix already gives it. "Largest single effect
measured" is narrowed to "largest effect of any training choice", because deleting
the diffusion term at sampling time costs more (see the checks below).

### Item 3: AE 3, matched baseline conditions

> Conditional flow matching is added at matched architecture, optimiser, EMA decay,
> batch size, training budget, data split, evaluation code and hardware, apart from
> the weight decay noted above (Tables~\ref{tab:cifar} and~\ref{tab:mnist}).
> Rectified flow with the linear path is the same objective and is not reported
> separately. DDIM is reported for both diffusion baselines, and improved denoising
> diffusion with both of its samplers, the stronger one used for the comparison on
> each dataset.

### Item 4: AE 4, cost at matched quality

> At equal quality the bridge is the cheapest route to FID $15$ and above
> ($10$ evaluations against improved denoising's $12$ at FID $20$). At FID $10$ and
> below it needs fewer evaluations than flow matching but more than improved
> denoising ($47$ against $65$ and $23$ at FID $10$). The advantage is a factor of
> $1.2$ in evaluations at FID $20$, not an order of magnitude, and
> Section~\ref{sec:gaussclaims} states it at that size.

"At FID 10 and below" only refers to the targets the table reports (10 and 9), and
both of those follow the same ordering.

### Item 5: AE 5, sampling-budget coverage

> Sensitivity is reported for the diffusion scale over five separately trained
> values of $\gamma$ (Figure~\ref{fig:gamma}), for the sampler over three schemes,
> and for the discretisation grid over up to nine step counts from $1$ to $1000$.
> The terminal time and the diffusion scale are shown to be one parameter rather
> than two, since training and sampling depend on $\sigma$ and $T$ only through
> $\gamma^2=\sigma^2T$, which removes one of the sensitivity axes the point asks for
> and collapses two reported tuning decisions into one.

### Item 6: AE 6, FID floor

> [...] so the grayscale and colour pipelines differ only in the input.
> Achievable floors are measured rather than assumed, using the comparator a table
> entry uses.

The only change is deleting "at four sample counts". There are no new numbers or
caveats; the 0.67 extrapolation and the MNIST limit are already in the table
captions.

### Item 7: R1.2, variance profile

Please don't apply the AFTER for this item. The two sets of numbers come from
different measurements:

- The paragraph is about the variance-restoring experiment: coverage 0.602 → 0.639.
  That experiment ran on the CIFAR-10 velocity network, and `variance_restore.py`
  measures its own profile on that network: 41 grid points, $v(0)=0.24565$, and it
  decreases monotonically. Those numbers are in `runs/variance_restore.json`, so
  0.24565 and 41 are correct here.
- The appendix's 0.21346 / 0.21201 / 21 come from the MNIST endpoint network
  (`runs/diagnose_timecond.json`, plotted by `theory_figure.py`). Quoting them in
  this paragraph would attach an MNIST profile to a CIFAR-10 result.
- The one real error is the comparator. 0.25315 is the variance pooled over every
  pixel of the CIFAR-10 train split (`real.var()`), which also counts the spread
  between pixel means. Proposition~\ref{propconditionalvarianceprocess} needs the
  mean per-pixel variance, which is what the MNIST appendix uses. On the same
  convention (test split, first 4096 images) CIFAR-10 gives $0.24603$, so the
  ratio is $0.998$ rather than $0.970$. On the train split it is $0.24792$ (ratio
  $0.991$).
- "Zero at $s=1$" holds on CIFAR-10 by construction, not by measurement: the
  velocity parametrisation returns $\xi_1 = Y$ at $s=1$ (`bridge.py:191`). The
  measured terminal check is the MNIST endpoint network's $9\times10^{-5}$ in the
  appendix.

The last sentence becomes:

> The profile estimated on the same CIFAR-10 network also meets the proposition's
> initial condition, $0.24565$ against a mean per-pixel variance of $0.24603$ at
> $s=0$, and decreases at all $41$ grid points; the terminal zero, which the
> velocity parametrisation imposes by construction, is measured on the
> endpoint-trained MNIST network at $9\times10^{-5}$ (Appendix~\ref{app:variance}).

### Item 8: R1.5, architecture and training control

The last sentence becomes:

> Apart from the weight decay noted above, the only differences between rows are
> the generative process and the sampler.

### Item 9: R1.5, repeated ablation

> Four components are ablated separately: the training objective, endpoint against
> drift at matched $\gamma$ and sampler ($38.93$ against $23.23$ FID at fifty
> evaluations, the largest effect of any training choice); the sampler, over three
> schemes; the diffusion scale, over five separately trained values; and the
> discarded posterior variance, on and off. The objective and the sampler are also
> crossed, which shows that at ten evaluations the unweighted objective understates
> the cost of deleting the diffusion term by a factor of three.

The last sentence only holds if the 2×2 is added to the appendix; see the checks
below.

### Item 10: R1.7, sample-image budgets

> Sample grids at matched budget are in Appendix~\ref{app:samples}, including a
> direct comparison of the two training objectives at matched $\gamma$ and budget,
> and samples against sampling budget from two to two hundred evaluations on
> CIFAR-10 and two to one thousand on MNIST. Timing is in
> Section~\ref{sec:gausscost}.

### Item 11: R1.7, coverage-table metrics

> Mode coverage is quantified in two independent ways. On CIFAR-10,
> Table~\ref{tab:coverage} reports precision and recall, the outlier-robust
> coverage, and KID, for six sampler rows at three step counts. [rest unchanged]

This version is only needed if density stays out of the table. If we add the density
column instead (recommended, see the checks below), the original "the outlier-robust
density and coverage" becomes correct and this paragraph stays as it was.

### Item 12: R1.8, CIFAR-10 range

> [...] the bridge leads improved denoising diffusion at every step count on MNIST
> and from two to ten evaluations on CIFAR-10 [...]

## Checks on the unchanged claims

### Density (item 11)

All 18 rows of `tab:coverage` have a density value in `runs/metrics.log` (same
run, same k=3 Inception features). So the main text (`experiments.tex:247`) isn't
wrong; the table just leaves the column out. I'd add the column rather than cut the
claim, since it also answers the reviewer's coverage question directly. Values, in
table row order:

| Model / simulation | 10 | 50 | 200 |
|---|---|---|---|
| Random-bridge, velocity / Euler–Maruyama | 0.734 | 0.764 | 0.808 |
| Flow Matching / Euler | 0.654 | 0.794 | 0.813 |
| Improved denoising / ancestral | 0.740 | 0.921 | 0.912 |
| Improved denoising / DDIM | 0.467 | 0.704 | 0.766 |
| Simple denoising / ancestral | 0.402 | 0.692 | 0.798 |
| Random-bridge, endpoint / exact posterior | 0.457 | 0.595 | 0.597 |

Density can exceed 1 and isn't bounded like coverage; none of these do. The caption
needs one clause saying so.

### "Largest single effect measured" (items 2, 9): false as written

CIFAR-10, γ = 0.3162, 50 evaluations, n = 2000 (`fid_sanity.json` per run), the same
protocol as the 38.93 → 23.23 pair:

| Component varied | Comparison | ΔFID |
|---|---|---|
| **Sampler: delete the diffusion term** | endpoint, Euler–Maruyama 38.93 vs deterministic 81.02 | **42.1** |
| Objective | endpoint 38.93 vs velocity 23.23, both Euler–Maruyama | 15.7 |
| Sampler: exact posterior vs Euler–Maruyama | velocity 28.53 vs 23.23; endpoint 42.14 vs 38.93 | 5.3; 3.2 |
| Posterior variance restored | exact posterior 17.70 → 14.28 (velocity, n = 10k) | 3.4 |
| Diffusion scale γ ∈ {0.05, 0.1, 0.2, 0.3162} | endpoint, Euler–Maruyama: 38.93–42.29 | 3.4 |
| Diffusion scale γ ∈ {0.05, 0.3162} | velocity, Euler–Maruyama: 23.34 vs 23.23 | 0.1 |

The 2×2 at n = 10k gives the same order: deleting the diffusion term costs 37.8
(endpoint, 50 evaluations) and 90.8 (velocity, 10), against 16.7 for the objective.
The objective is the largest *training* choice, but not the largest effect measured,
and a referee can see the deterministic-map numbers in Figure~\ref{fig:gamma}. The
corrected items 2 and 9 above say "the largest effect of any training choice".

### "Factor of three" (item 9): holds at ten evaluations only, and isn't in the paper

From `runs/*/objsampler_n10k.json` (CIFAR-10, n = 10k), the cost of deleting the
diffusion term:

| Objective | 10 evaluations | 50 evaluations |
|---|---|---|
| endpoint (unweighted) | 69.76 − 39.35 = 30.41 | 66.37 − 28.52 = 37.85 |
| velocity | 111.67 − 20.87 = 90.80 | not run |

The ratio at 10 evaluations is 2.99, so the "factor of three" is right there. At 50
evaluations there is no number: the velocity/deterministic cell is missing from
`objsampler_n10k.json` and `runs/2x2.log` ends before it (the queue never ran it), so
the claim can't be stated in general. Item 9 now says "at ten evaluations".

The bigger problem is that the 2×2 crossing doesn't appear anywhere in the
manuscript, so "the objective and the sampler are also crossed" points the referee
at nothing. We should either add a four-cell table to `app:weighting` (the 10-evaluation
cells above, plus the 50-evaluation cell once the missing run is done) or delete the
sentence from the response. I'd add it: it's the evidence for the claim that the
unweighted objective hides how much the diffusion term matters.

## Remaining actions

1. Paste the corrected paragraphs into the response source. Please push the source
   here as well, so the response and the manuscript are versioned together.
2. Item 7 needs a manuscript follow-up too. `app:variance` never says its profile
   is MNIST, and it says the variance-restoring experiment reads its estimate "off
   the profile of Figure~\ref{fig:variance}", which isn't true: that run measured its
   own CIFAR-10 profile. One sentence naming the dataset of each fixes both.
3. Add the density column to `tab:coverage` (values above).
4. Add the 2×2 table to `app:weighting`, or drop the crossing sentence from item 9.
   Getting the missing 50-evaluation cell means one evaluation run on the existing
   velocity checkpoint: `--samplers euler_det --steps 50`.

Stefano
