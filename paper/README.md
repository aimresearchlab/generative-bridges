# Experiments section and appendix — complete, ready for Overleaf

Everything needed to bring the experiments subsection and its appendix into
`main_Final.tex`. Self-contained: all seven figures, both tables, both sections, the new
bibliography entries, and the instructions.

```
sections/experiments.tex            main experiments subsection
sections/appendix_experiments.tex   appendix A.1-A.5
tables/cifar.tex                    CIFAR-10 table body, generated from the evaluation output
tables/mnist.tex                    MNIST table body, generated from the evaluation output
figures/velocity_error.pdf          A.1
figures/gamma_sweep.pdf             A.3   (regenerated 18 Oct)
figures/conditional_variance.pdf    A.4   (regenerated 18 Oct)
figures/samples_objective.pdf       A.5   (regenerated 17 Oct)
figures/samples_steps.pdf           A.5
figures/samples_models.pdf          A.5
figures/samples_mnist.pdf           A.5
bibitems_to_add.tex                 7 new \bibitem entries
README_OVERLEAF.md                  the three edits to make in main_Final.tex
CHANGES-2026-10-17.md               the twelve corrections, with find/replace text
experiments.diff                    the same changes as unified diffs
appendix_experiments.diff
```

No new packages are required: only `booktabs` and `graphicx`, both already loaded.

## How to bring it in

`README_OVERLEAF.md` has the three edits to `main_Final.tex`: replace the
`[TO BE REWRITTEN]` placeholder with `\input{sections/experiments}`, add
`\input{sections/appendix_experiments}` before the bibliography, and paste the seven
`\bibitem` entries.

**Then read `CHANGES-2026-10-17.md` before uploading the `.tex` files.** Overleaf is ahead
at `13de22f` with a consistency and bibliography pass that is *not* in these sections, so
uploading them wholesale would revert it. The twelve corrections are listed there with
find/replace text so they can be applied to the live document instead. The `.diff` files
show the same changes if that is easier to read.

**The figures have no such conflict — upload all seven.** Three changed on 17-18 October
and the other four are unchanged since 11 October.

## What the corrections were

Twelve edits, all fixing statements that contradicted the tables or the code. **No new
experiments were run**; every replacement number came from data already on disk.

The substantive ones: the optimiser was not shared across models (the bridge trains at
zero weight decay, the baselines at AdamW's default $10^{-2}$); the FID floor quoted at
$n=50\,000$ was neither measured nor correct, and is now an extrapolation of $0.67$ rather
than $0.15$; the objective ablation quoted $42.14 \to 23.23$ across two different
samplers, where the sampler-matched pair is $38.93 \to 23.23$; and two figures had
investigational prose rendered into the image, one of it critiquing our own manuscript.

The rest are range and arithmetic corrections: four claims of the form "below twenty
evaluations" that fail at a single evaluation, three MNIST numbers that disagreed with
`tables/mnist.tex`, a second wrong floor ($0.70$ where the measurement is $0.98$), the
CIFAR-10 base-channel count, and four figure captions.

Three further findings are disclosed rather than fixed, because correcting them would
need retraining or re-sampling. They are in `CHANGES-2026-10-17.md` and in
`../errata-2026-10-17.md`, with suggested wording.

## A note on figures

Matplotlib renders figure text as vector paths, so `pdftotext` returns zero characters for
some of these files and no text-level scan can inspect them. The two figures that carried
investigational prose were only found by rasterising and looking. If a figure is
regenerated, look at the PNG before shipping it.

Convention: keep text in figures to the minimum that makes the axes readable — titles,
axis labels, legend, units. Everything explanatory goes in the LaTeX caption. All seven
figure generators now take `--bare` for this.

## Provenance

`../review-response-2026-10-17.md` records what was checked against what: all 139 records
across the twelve `fid_n50k*.json` files are `n_samples=50000`, `ckpt_step=40000`; the
measured floor triple reproduces exactly; both parameter counts match the checkpoints; and
the $\gamma$ ablation was already sampler-matched.

`main_Final.tex` is deliberately absent, as is anything quoting the referee letter.
