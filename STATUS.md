# Where the CIFAR-10 sweep stands

Paused on 2026-10-01 with the GPU handed back. Nothing is lost: every
`(sampler, n_steps)` entry is written to its JSON as it completes, and
`bridges.evaluate` skips entries already present, so resuming continues from
here.

## Complete, at n = 50,000

| row | steps measured |
| --- | --- |
| Bridge, gamma = 0.3162, drift objective, Euler-Maruyama | 1, 2, 5, 10, 20, 50, 100, 200, 1000 |
| Flow matching | 1, 2, 5, 10, 20, 50, 100, 200, 1000 |
| Improved denoising diffusion, ancestral | 1, 2, 5, 10, 20, 50, 100, 200, 1000 |

These three carry the comparison and `runs/tables_cifar10.md` is current for
them.

## Outstanding

| row | missing | cost |
| --- | --- | --- |
| Improved denoising diffusion + DDIM | 1000 steps only | about 8 h |
| Simple denoising diffusion, linear, ancestral | the whole row | about 8 h |
| Seed replicates at n = 50,000 | none run | 8 h per replicate |
| MNIST Euler rows | re-measurement after the evaluation-count change | about 6 h |

Neither of the first two carries a claim; they complete the row set of the
submitted table.

The MNIST Euler and deterministic-Euler entries were measured when the sampler
took one network evaluation after the integration loop to read the endpoint,
while the record wrote `nfe = n_steps`. The sampler now reuses the estimate
already computed at the last grid point, matching how the exact-posterior
sampler was always counted. The CIFAR-10 table is measured after that change;
the MNIST table is not, and its Euler rows are to be re-run before use. The
exact-posterior and baseline rows are unaffected.

## Resuming

```bash
./run_cifar_sweeps.sh      # continues at the first unmeasured entry
```
