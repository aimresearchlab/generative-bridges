# Experimental review: open issues for Stefano

Reviewed code: `bridge-reimpl` at `4f89bf9` (17 October 2026). Manuscript references below are to the integrated Overleaf files: `main_Final.tex`, `sections/experiments.tex`, `sections/appendix_experiments.tex`, and `tables/`.

These are the substantive open findings from the experimental review. The approved manuscript consistency and bibliography fixes are already on Overleaf at `13de22f` and are excluded from the action list. No implementation fixes or new image experiments have been made during this review.

The observations below describe the checked-in code and displayed manuscript. The exact revisions and settings used for the reported runs were unavailable during review. Please check these against your existing configs, logs, JSONs, and checkpoint metadata first: a difference in today's code does not establish that a historical result is wrong. Existing records may resolve several items without any rerun.

## Where to start

| Item | Why it matters | Cheapest first step | New training? |
| --- | --- | --- | --- |
| 1. Improved-DDPM training and optimizer matching | Validity of the baseline comparison | Check actual training revision and settings | Potentially, if affected weights support a retained reference-equivalent comparison |
| 2. Objective ablation | Whether the loss benefit is isolated | Inspect existing objective-by-sampler results | No; missing comparisons need sampling from existing weights |
| 3. Result identity and resume logic | Whether table cells belong to the intended runs | Audit JSON provenance and one-step entries | No |
| 4. Algorithms versus evaluated method | Reproducibility of the stated method | Check checkpoint targets and sampler conventions | Usually no |
| 5. Velocity-error units | Interpretation of a central diagnostic | Reconcile normalized and physical time | No |
| 6. DDIM clipping convention | Baseline sampling and explanation of degradation | Compare the actual sampler with the reference | No; changed results need sampling |
| 7. Remaining table/prose contradictions | Accuracy of empirical summaries | Check the existing CIFAR table | No |
| 8. FID floor and uncertainty | Support for resolved-difference claims | Locate calibration logs and seed JSONs | No |
| 9. Variance restoration | Agreement between the described correction and code | Check the final-step convention | No; changed results need sampling |
| 10. Analytical diagnostics | Whether the diagnostics test the claimed behavior | Align the toy sampler and diagnostic interpretation | No image-model training |

## 1. Improved-DDPM training and optimizer matching

In [bridges/improved_ddpm.py](bridges/improved_ddpm.py), `training_loss` samples one timestep uniformly, averages its VLB contribution, and multiplies it by `lam=0.001` by default. It also calls `p_mean_logvar` with its default `clip=True` during VLB training.

The [OpenAI training implementation](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py#L654-L687) uses coefficient 1 on the sampled VLB term for `MSE`, or `num_timesteps/1000` for `RESCALED_MSE`; both are 1 at 1000 timesteps. It disables denoised clipping for this term. Thus the branch default is 1000 times smaller in VLB coefficient at that timestep count and also differs in training clipping. This establishes an implementation difference, not a measured FID effect.

Separately, [bridge training](bridges/train.py) explicitly defaults AdamW weight decay to 0. [Flow Matching](bridges/flowmatch.py), [simple DDPM](bridges/ddpm_baseline.py), and improved DDPM omit that argument, inheriting the optimizer's default of 0.01. That conflicts with an assertion that only the generative process and sampler differ, if those settings were used.

**Next step:** check the code revision, VLB weight, clipping, and optimizer settings used for the actual checkpoints. If the runs already used the intended implementation, record that recipe and reconcile the branch. If they used these differences, decide whether the comparison should retain those settings with explicit disclosure or use corrected training. Reevaluation cannot repair a training objective in existing weights; reference-equivalent retraining and any optimizer-matched arms would be the potentially expensive part.

## 2. The loss ablation's figure changes the sampler too

The manuscript attributes the CIFAR-10 FID improvement from 42.14 to 23.23 at 50 evaluations and 2000 samples to the objective correction. However, [bridges/samples_figure.py](bridges/samples_figure.py) loads `bridge_0050.png` for the endpoint objective and `euler_0050.png` for the drift objective. The illustrated comparison therefore changes sampler as well as objective. The JSONs are needed to establish whether the quoted FIDs do too.

**Next step:** inspect the existing `objsampler_n10k.json` files from [run_2x2.sh](run_2x2.sh), which compares both checkpoints with Euler and deterministic Euler at 10 and 50 evaluations. Compare objectives under the same sampler, gamma, time conditioning, sample count, and metric protocol. Keep the 10,000-sample comparison separate from the quoted 2000-sample values and headline 50,000-sample table.

Also identify the checkpoint behind the `flow` series in `mse_by_s_cifar.json` and `mse_by_s_mnist.json`. If this is a zero-noise Flow Matching model, the velocity-error comparison changes the training path distribution as well as the objective. Its generating recipe was not found in the branch.

**Compute:** existing outputs may suffice. Otherwise use existing checkpoints for a small matched comparison; no new training is required to compare the available models.

## 3. Audit result identity before rerunning table cells

In the current [simple DDPM sampler](bridges/ddpm_baseline.py), one-step DDIM and ancestral sampling return the same clipped endpoint for identical weights and initial noise. Yet the MNIST table reports 418.50 versus 377.15 for cosine and 418.36 versus 282.19 for linear. Current bridge Euler and deterministic Euler likewise return the same one-step prediction for identical inputs, while their table entries are 346.30 and 346.05.

Different seeds or historical implementations can produce different measurements, so these observations do not select a correct number. They do make the one-step column a useful place to check checkpoint identity, seeds, batching, precision, EMA/raw choice, sampler revision, and network-evaluation accounting.

There is a concrete reuse hazard in [bridges/evaluate.py](bridges/evaluate.py): completed records are keyed only by `(sampler, n_steps)`. Reusing an output path with a different checkpoint, seed, sample count, or other settings silently skips those cells. A CPU check during review confirmed reuse of a 100-sample record when requesting 50,000 samples from a different model/seed. This demonstrates the hazard, not that the paper's files were affected.

**Next step:** map each published row to its original JSON and command, including whether it predates the sampler/NFE fixes. Make resume validation reject incompatible experiment settings and record the checkpoint and evaluator revision. Preserve original outputs and use fresh paths for checks.

**Compute:** the audit and code fix need no GPU work. If provenance remains unresolved, start with selected one-evaluation cells from existing weights; expand only to affected entries.

## 4. Align the algorithms with the method actually evaluated

Algorithm 1 describes endpoint prediction with uniform time on `[0,T)`. The experimental text says all subsequent results use velocity regression with truncation at `0.999 T`. [bridges/bridge.py](bridges/bridge.py) supports both targets and feeds the network `t_scale * s`, where `s=t/T` and the default scale is 1000. Run names/scripts suggest an endpoint-trained MNIST checkpoint and drift-trained CIFAR checkpoints, but filenames alone cannot establish their actual settings.

Algorithm 2 returns the simulated state on a grid ending at `T-epsilon`. The implementation's Euler variants normally return the endpoint estimate computed in the last loop iteration, rather than the final updated state. This uses the stated number of network calls but is a different output convention.

**Next step:** use checkpoint args and commands to state the target, truncation, time scaling, grid, and returned quantity for each reported experiment. Update the algorithm or identify the experimental variant explicitly. Also check the architecture sentence: it says 64 base channels, while [build_unet](bridges/unet.py) uses 64 for MNIST and 128 for CIFAR-10.

**Compute:** documenting the existing recipe needs none. Changing the sampler output requires reevaluation; replacing a training target requires training only if that replacement is actually desired.

## 5. Velocity-error numbers use normalized-time scaling

The paper defines physical-time velocity as `(E[Y | xi_t] - xi_t)/(T-t)`. In the appendix weighting table, endpoint MSE 0.20779 at `s=t/T=0.05` becomes approximately 0.230, which is division by `(1-s)^2`. For the stated `T=0.1`, division by `(T-t)^2` instead gives approximately 23.0.

**Next step:** define the normalized field `v_s = T u_t` and label the existing diagnostics consistently, or convert the errors to physical-time units. Check the accompanying noise-variance expression as well: the displayed physical-time target implies `sigma^2 t/[T(T-t)]`, while the subsequent prose omits `1/T`.

**Compute:** notation and a common unit conversion require no experiments. Ratios between estimators and the FID results are unchanged by a common conversion.

## 6. DDIM after clipping differs from the reference

Both [simple DDPM](bridges/ddpm_baseline.py) and [improved DDPM](bridges/improved_ddpm.py) retain the network's raw epsilon after clipping the reconstructed `x0`. The [OpenAI DDIM sampler](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py#L440-L481) rederives epsilon from the processed `x0`. The simple sampler's `eta=1` mean can consequently differ from a direct respaced ancestral posterior when clipping is active.

**Next step:** check which convention produced the reported rows and whether reference-sampler comparisons already exist. The difference is relevant to the explanation of large-step-count DDIM degradation; the present review does not establish its measured contribution or isolate the noise schedule as the cause.

**Compute:** compare formulas and toy cases on CPU first. Quantifying the difference uses existing checkpoints and new sampling, without retraining.

## 7. Remaining CIFAR summaries contradict the table

These were left for author consideration, rather than changed in the editorial pass:

- The largest absolute bridge/Flow Matching gap across the full table is 13.77 FID at one evaluation, not 1.24. The 1.24 statement holds from 10 evaluations onward; the gaps at two and five evaluations are also larger than 1.24.
- The summary that the bridge leads Flow Matching below 20 evaluations and trails above 50 needs dataset-specific ranges. On CIFAR it trails at one and five, and still leads at 100.
- Improved ancestral diffusion is not better than its DDIM variant at every measured CIFAR count: at two evaluations, DDIM is 363.91 versus ancestral 364.42, and at one they tie.

**Next step:** restrict the summaries to the ranges supported by the displayed tables. This preserves the measured entries and their specific advantages. No experiments are needed.

## 8. Locate the evidence for the FID floor and resolved differences

The paper calls 0.15 a measured CIFAR floor at 50,000 samples, but [bridges/calibrate_fid.py](bridges/calibrate_fid.py) caps its holdout arm at the 10,000-image test split. Its disjoint arm cannot form two disjoint 50,000-image sets from the training split. The supplied scripts pass 0.15 as a constant, so they do not establish that measurement.

The uncertainty discussion also treats three sampling seeds at 10,000 samples as an upper bound for 50,000-sample uncertainty and describes a gap as roughly seven standard deviations. That needs the actual seed records and uncertainty of the difference, including covariance if paired. Sampling-seed spread does not measure training-seed variation.

**Next step:** attach the calibration procedure/log and seed JSONs. If 0.15 is an estimate, label it accordingly. Check resolved-difference claims against the available uncertainty evidence. Likewise, a shared FID rise from 200 to 1000 evaluations is an observation; attributing it to stochastic simulation needs additional evidence.

**Compute:** existing records and appropriately scoped wording may suffice. A new floor calibration uses real-data feature extraction; it does not require model training. Repeated long sampling sweeps are unnecessary merely to report the observed trend.

## 9. Variance restoration is disabled on the final bridge step

In [bridges/bridge.py](bridges/bridge.py), `sample` forces `eta=0` on the last `bridge` step, including when a positive `var_profile` is supplied. Consequently, the extra endpoint-uncertainty term added by `step` does not produce final-step noise. A one-step constant-predictor check with a positive variance profile still returns zero variance.

**Next step:** decide whether the intended method deliberately uses a partial restoration and describe that exception, or apply and evaluate the full correction. Existing measured gains can remain measurements of the implemented variant; they should be tied to that variant.

**Compute:** clarifying the method needs none. Changing the correction requires sampling from existing weights, not training.

## 10. Two diagnostics need alignment with their claims

In [bridges/sanity_1d.py](bridges/sanity_1d.py), Euler takes an additional conditional-mean evaluation after the loop. Production Euler returns the prediction from the last loop iteration. The diagnostic therefore tests a different output rule and call count. Align these before using the analytical check to validate the production sampler; this is a small CPU calculation.

The conditional-variance plot generated by [bridges/theory_figure.py](bridges/theory_figure.py) measures batch-averaged prediction error. For a fitted predictor, that error includes both expected conditional variance and conditional-mean approximation error. A decreasing average is consistent with the theorem but does not directly verify its conditional supermartingale inequality or full matrix-order claim. Check the norm/dimension normalization and describe the plot as the diagnostic it actually computes. Existing plots may suffice; this does not inherently require new image experiments.

## Minimal handoff and next pass

For each retained table/figure, identify its checkpoint, training revision/settings, evaluation command/revision, and raw JSON. The useful fields are target, gamma, time scale, architecture, optimizer settings, training budget, EMA/raw choice, sampler/clipping/output convention, sample count, seed, precision, and generation batch size. Small configs and result JSONs plus a checkpoint-location manifest would make this review reproducible without committing model binaries.

Resolve items from existing records first. If sampling is needed, select the affected cells or matched ablation rather than automatically repeating the 1-to-1000-evaluation sweep. Decide on new training only after checking whether the reported checkpoints actually used the training differences in item 1.

The review included source inspection and small CPU checks; it did not independently reproduce the image-model FIDs or certify the full mathematical theory. Each finding should be closed against the actual run evidence.
