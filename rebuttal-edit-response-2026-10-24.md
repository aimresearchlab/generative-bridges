# Response to the rebuttal numerical and prose review (20 October)

24 October 2026 — reply to `rebuttal-edit-self-review-2026-10-20.txt`

Berkan, thanks. I checked all twelve items against `paper/sections`, `paper/tables`,
`runs/fid_floors.json`, `runs/metrics.log` and the figure PDFs. All twelve are real
mismatches between the response and the manuscript. Below is the corrected wording
for each paragraph, ready to paste into the response source (which isn't in this
repo, so I haven't applied it). Text not quoted here stays as you have it.

## Verification summary

| Item | Response said | Source of truth | Verdict |
|---|---|---|---|
| 1, 3, 8 | only process and sampler differ | `experiments.tex:18`, `appendix_experiments.tex:301`: weight decay 0 vs 1e-2 | Fix, short exception |
| 2, 9 | 42.14 → 23.23 | `appendix_experiments.tex:65`: 38.93 → 23.23, both Euler–Maruyama, n=2000. 42.14 was the exact-posterior-sampled entry (errata item 7) | Fix |
| 4 | most expensive to FID ≤10 | `tab:cost`: FID 10 is 47 / 65 / 23, FID 9 is 151 / 157 / 27 | Fix |
| 5 | nine step counts for every method | `tab:cifar`: DDIM has dashes from 20 evaluations | Fix |
| 6 | floors at four sample counts | `runs/fid_floors.json`: three counts (2000, 5000, 10000) per dataset; no fourth exists | Fix, minimal |
| 7 | 0.24565 / 0.25315 / 41 | `appendix_experiments.tex:226`: 0.21346 / 0.21201 / 21 | Fix |
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
> $38.93$ to $23.23$ at the same $\gamma$, sampler, architecture and budget, and is the
> largest single effect measured. It also explains the step-count behaviour the
> referees asked about: refining the sampling grid spends more time integrating
> through exactly the region the uniform objective discounted, which is why
> additional steps made the original configuration worse.

I dropped "$n=2000$" because the appendix already gives it. "Largest single effect
measured" stays as it was; see the open points.

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

The last sentence becomes:

> The estimated profile also meets the proposition's two boundary conditions:
> $0.21346$ against $\mathrm{Var}[Y]=0.21201$ at $s=0$, zero at $s=1$, and
> decreasing at all $21$ grid points.

### Item 8: R1.5, architecture and training control

The last sentence becomes:

> Apart from the weight decay noted above, the only differences between rows are
> the generative process and the sampler.

### Item 9: R1.5, repeated ablation

> Four components are ablated separately: the training objective, endpoint against
> drift at matched $\gamma$ and sampler ($38.93$ against $23.23$ FID at fifty
> evaluations, the largest effect measured); the sampler, over three schemes; the
> diffusion scale, over five separately trained values; and the discarded posterior
> variance, on and off. The objective and the sampler are also crossed, which shows
> the unweighted objective understates the cost of deleting the diffusion term by a
> factor of three.

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

### Item 12: R1.8, CIFAR-10 range

> [...] the bridge leads improved denoising diffusion at every step count on MNIST
> and from two to ten evaluations on CIFAR-10 [...]

## Open points

1. **Manuscript follow-up from item 11.** `experiments.tex:247` says Appendix
   `app:coverage` reports "the outlier-robust density and coverage", but
   `tab:coverage` has no density column. Density was computed for the bridge rows
   (`runs/metrics.log`). We either add the column (if every row has it) or change
   the main text to "coverage". I'll check whether all rows have density and fix
   the manuscript separately, so that the response and the paper match.
2. **"Largest single effect measured" (items 2, 9)** and **"factor of three"
   (item 9)** are unchanged. You're right that correcting 42.14 doesn't re-validate
   them. I'll check both against the γ sweep and the 2×2 runs before the response
   goes to Overleaf. If either doesn't hold, I'll soften it.
3. Once these paragraphs are in your local response source, please push it here as
   well, so the response and the manuscript are versioned together.

Stefano
