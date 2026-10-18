# Manuscript sections, for review alongside the code

The two experiments subsections as corrected on 17-18 October 2026, with their tables and
the one figure that was regenerated. They are here so the corrections can be read next to
the code and run outputs they were checked against.

```
sections/experiments.tex            main experiments subsection
sections/appendix_experiments.tex   appendix
tables/cifar.tex, tables/mnist.tex  table bodies, generated from the evaluation output
figures/samples_objective.pdf       regenerated: both panels now use the same sampler
CHANGES-2026-10-17.md               the ten edits, with find/replace text
experiments.diff                    the same changes as a unified diff
appendix_experiments.diff
```

**These are not the live manuscript.** Overleaf is ahead at `13de22f` with a consistency
and bibliography pass that is not reflected here, so these files are a base of 11 October
plus the ten corrections. Apply the corrections to Overleaf rather than uploading these
over it — `CHANGES-2026-10-17.md` gives the find/replace text for each.

`main_Final.tex` is deliberately absent. So is anything quoting the referee letter.

## What the corrections were

Ten edits, all fixing statements that contradicted the tables or the code. No new
experiments were run; every replacement number came from data already on disk.

The substantive ones: the optimiser was not shared across models (the bridge trains at
zero weight decay, the baselines at AdamW's default $10^{-2}$); the FID floor quoted at
$n=50\,000$ was neither measured nor correct, and is now an extrapolation of $0.67$ rather
than $0.15$; and the objective ablation quoted $42.14 \to 23.23$ across two different
samplers, where the sampler-matched pair is $38.93 \to 23.23$.

The rest are range and arithmetic corrections: four claims of the form "below twenty
evaluations" that fail at a single evaluation, three MNIST numbers that disagreed with
`tables/mnist.tex`, the CIFAR-10 base-channel count, and two figure captions.

Three further findings are disclosed rather than fixed, because correcting them needs
retraining or re-sampling. They are listed in `CHANGES-2026-10-17.md` and in
`../errata-2026-10-17.md`, with suggested wording.

## Provenance

`../review-response-2026-10-17.md` records what was checked against what. In short: all
139 records across the twelve `fid_n50k*.json` files are `n_samples=50000`,
`ckpt_step=40000`; the measured floor triple reproduces exactly; both parameter counts
match the checkpoints; and the $\gamma$ ablation was already sampler-matched.
