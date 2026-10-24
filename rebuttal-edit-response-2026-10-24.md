# Reply to the rebuttal review (24 Oct)

Berkan, checked against the paper and `runs/`. Apply these; no new runs.

## Response edits

1. **Items 1, 3, 8:** add "apart from weight decay (zero for the random-bridge runs,
   $10^{-2}$ for the baselines)" in item 1, and "apart from the weight decay noted
   above" in 3 and 8.
2. **Items 2, 9:** $42.14 \to 38.93$, say "same sampler", and replace "largest
   (single) effect measured" with "largest effect of any training choice".
   Deleting the diffusion term costs more (38.93 → 81.02).
3. **Item 9:** delete the sentence "The objective and the sampler are also
   crossed…". The paper doesn't report that comparison.
4. **Item 4:** "At FID $10$ and below it needs fewer evaluations than flow matching
   but more than improved denoising ($47$ against $65$ and $23$ at FID $10$)."
5. **Item 5:** "over up to nine step counts from $1$ to $1000$" (drop "for every
   method").
6. **Item 6:** delete "at four sample counts". Only three exist.
7. **Item 7: don't use your AFTER.** This paragraph is the CIFAR-10 experiment, and
   0.24565 / 41 are its correct numbers. The appendix ones are MNIST. Replace the
   last sentence with: "The profile estimated on the same CIFAR-10 network matches
   the mean per-pixel variance at $s=0$ ($0.24565$ against $0.24603$) and decreases
   at all $41$ grid points." (0.25315 was the pooled variance, the wrong quantity.)
8. **Item 10:** "from two to two hundred evaluations on CIFAR-10 and two to one
   thousand on MNIST".
9. **Item 11:** drop "density".
10. **Item 12:** "from two to ten evaluations on CIFAR-10".

## Manuscript edits

- `experiments.tex:247`: "the outlier-robust density and coverage" → "the
  outlier-robust coverage".
- `app:variance`: say that Figure~\ref{fig:variance} is MNIST, and that the
  variance-restoring run uses the same estimate measured on its own CIFAR-10
  network.

Please push the response source here when done.
