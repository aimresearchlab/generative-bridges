# Experiments section for `main_Final.tex` — drop-in files

Everything here is written in the notation of the current manuscript: time runs
on `[0,T]`, the bridge state is `\xib^{\boldsymbol{x}}_t`, the conditional
expectation is `\boldsymbol{Y}^{\boldsymbol{x}}_t`, the network is
`f^*(\cdot;\boldsymbol{\Theta}^*)`, the diffusion scale is `\boldsymbol{\sigma}`,
and simulation cost is quoted as network evaluations per sample, equal to the
grid size `m` of Algorithm 2. No new packages are required: the files use only
`booktabs` and `graphicx`, both already loaded.

## Files

```
sections/experiments.tex            main experiments subsection
sections/appendix_experiments.tex   appendix (weighting, coverage, gamma, Prop. variance, samples)
tables/cifar.tex                    CIFAR-10 table body, generated from the evaluation output
tables/mnist.tex                    MNIST table body, generated from the evaluation output
figures/*.pdf                       7 figures
bibitems_to_add.tex                 7 new \bibitem entries
```

## The three edits in `main_Final.tex`

**1. Replace the placeholder.** Find

```latex
\subsubsection{Gaussian Random-Bridge Transport Numerical Experiments}
[TO BE REWRITTEN]
```

and replace the `[TO BE REWRITTEN]` line with

```latex
\input{sections/experiments}
```

**2. Add the appendix.** After `\section{Conclusion}` and its paragraphs, and
before `\section*{Data Availability}`, insert

```latex
\appendix
\input{sections/appendix_experiments}
```

**3. Add the references.** Open `bibitems_to_add.tex` and paste its seven
`\bibitem` lines inside the existing `thebibliography` environment. The keys
(`fid2017`, `cifar2009`, `prec2019`, `dens2020`, `binkowski2018`, `snr2024`,
`adamw2019`) do not clash with any key already in the paper.

That is all. No preamble change is needed.

## Uploading to Overleaf

Upload `overleaf-experiments.zip` through **Menu -> Upload Project** if you are
starting a fresh project, or drag the `sections/`, `tables/` and `figures/`
folders into an existing project through the file-tree upload button. Overleaf
preserves the folder structure, and the `\input` and `\includegraphics` paths
above are relative to the project root, so they resolve without modification.

## Points elsewhere in the manuscript that the measurements now bear on

These are flagged rather than changed, since they sit in sections owned by the
theory revision.

- **Algorithm 1.** The results are obtained by regressing the velocity field
  `\boldsymbol{u}` of eq. (continuousvelocity) rather than
  `\boldsymbol{Y}^{\boldsymbol{x}}_t`, which is the same objective carrying the
  weight `(T-t)^{-2}`. The experiments section derives this and reports what it
  is worth (CIFAR-10 FID at 50 evaluations: 42.14 against 23.23). If the
  reweighting is to be adopted as the recommended training rule, the loss in
  eq. (optimizationlossfunction) and the corresponding line of Algorithm 1 need
  one change each; the experiments text currently presents it as a property of
  the objective established empirically, which works either way.

- **Abstract, Introduction, Conclusion.** The present wording is "highly
  competitive" and "matching widely used denoising diffusions and Flow Matching
  baselines". The measurements support parity with Flow Matching throughout and
  an advantage over improved denoising diffusion below twenty network
  evaluations on CIFAR-10 and at every count on MNIST, but improved denoising
  leads on CIFAR-10 above twenty. A phrasing consistent with the tables would be
  "competitive in the few-evaluation regime", with the trade-off stated rather
  than the aggregate.

- **Section 1, paragraph on distillation.** The experiments section states
  explicitly that distilled samplers are outside the comparison by construction,
  being a compression of a trained teacher rather than a transport, which
  strengthens the existing sentence deferring them to future work.

## Provenance

Both table bodies are generated directly from the evaluation output rather than
transcribed. Every figure quoted in the prose was checked against the stored
results; see the reproduction instructions on the `bridge-reimpl` branch of the
code repository.
