# Eval reproducibility investigation

**Status: PAUSED. Do not run `scripts/reevaluate_test_timing.py` on any more
checkpoints until this is resolved** — it overwrites `test_metrics.json` and
`test_predictions/` in place, and the numbers it produces don't match what
was previously recorded for at least one checkpoint (see below). Running it
across all ~39 checkpoints while this is unexplained risks silently shifting
every ladder/range-breakdown comparison this project's decisions are based
on.

If you're picking this up in a new session: read this whole file before
touching `src/training/trainer.py`, `scripts/reevaluate_test_timing.py`, or
any `checkpoints/*/test_metrics.json` file.

## Why this started

The user asked to re-run evaluation on all completed training runs to
capture per-protein inference time, specifically to compare model inference
speed against APBS solve time (APBS timing already exists per-protein in
each protein's metadata JSON as `time_apbs_sec` — confirmed in
`src/electrostatics/run_apbs.py:185`, no new instrumentation needed on that
side).

## What was built

1. **`src/training/trainer.py`** — `evaluate_test()` modified to time each
   protein's forward pass (`torch.cuda.synchronize()` before/after,
   `time.perf_counter()`), with a one-batch CUDA warmup before the timed
   loop starts (kernel compilation overhead would otherwise skew the first
   protein). Adds `inference_time_s` per protein and
   `total_inference_time_s` / `mean_inference_time_s` to the global metrics
   dict. `git diff src/training/trainer.py` for the exact change — should
   be purely additive (timing instrumentation only), but see Step 1 below,
   this is now itself a suspect.

2. **`scripts/reevaluate_test_timing.py`** (new file) — standalone
   re-evaluation script. Loads `best_model.pt` from a given checkpoint dir,
   reconstructs the exact architecture from the checkpoint's own saved
   `model_config`/`feature_spec` (not from `config.yaml`, which may have
   changed since), reloads the test split, re-runs `evaluate_test`, and
   overwrites `test_metrics.json` + `test_predictions/*.npz` in place.
   Single GPU, no DDP (re-evaluation doesn't need it).

## The discrepancy

Ran `scripts/reevaluate_test_timing.py checkpoints/phase_a/attention_pw05`
(this checkpoint is Sweep A's decided winner and has been cited as the
baseline reference in dozens of comparisons throughout this project — see
`notebooks/decisions/07_loss_function_sweep.ipynb` and every later
phase_b/c/d/e notebook and analysis).

- **Previously recorded / repeatedly cited value:** test Pearson r =
  **0.8926**, RMSE = **2.5889**.
- **Fresh re-evaluation (twice, separate processes):** r = **0.8898**,
  RMSE = **2.660**, both times, consistently.

This is real signal, not noise — see what's already been ruled out below.
**`checkpoints/phase_a/attention_pw05/test_metrics.json` and
`test_predictions/*.npz` have already been overwritten with the new
(0.8898) numbers** — the original files are gone, can't be directly
recovered from disk. No other checkpoint has been touched.

## What's already been ruled out

- **Not bf16 precision.** Forced fp32 (`bf16=False`) in a fresh standalone
  script → r=0.8897, essentially identical to the bf16 fresh result
  (0.8898). If bf16 rounding/kernel-selection were the cause, fp32 should
  have landed much closer to 0.8926. It didn't.
- **Not run-to-run randomness.** Two separate fresh-process invocations of
  `reevaluate_test_timing.py` gave 0.8898 both times (RMSE differed only in
  the 4th decimal). Whatever's causing this is systematic, not stochastic.
- **Not a state_dict loading bug (probably).** `model.load_state_dict()`
  uses `strict=True` by default and did not raise — every key in the
  freshly-constructed model matched the checkpoint's saved state dict.
  `model_config`/`feature_spec` read from the checkpoint were also
  manually inspected and look correct (4/4/4 rounds, agg=multi,
  use_element_embedding=True, features off, esp_mean=-0.3557,
  esp_std=4.143 — all consistent with what Sweep A used).
- **Not a stale/misremembered figure, most likely** — r=0.8926 wasn't a
  one-off citation, it was repeated consistently across many separate
  tool-call results and tables over a long session (Sweep A's own
  `test_metrics.json` read at the time, later "range breakdown" tables,
  the "table of all baseline/phase D/phase E models" a few turns before
  this investigation started). A single miscite wouldn't reproduce
  identically across that many independent reads. (Not proven, though —
  see Step 4 below.)
- **CORRECTION (2026-08-21): the original log survives after all.** This
  entry originally claimed `nohup_phase_a.out` "no longer exists in
  `/home/student/thesis/`" — that check looked in the wrong directory
  (the parent of this repo, not inside `ProteinStructure_ESP_Thesis/`
  itself). The file was actually sitting untracked in the repo the whole
  time and has now been preserved as `sweep_a_original_run.txt` at the
  repo root. It directly confirms the original run really did print
  `RMSE: 2.5889 kT/e`, `Pearson r: 0.8926` for `attention_pw05` — so
  **0.8926/2.5889 is not a stale/misremembered citation; it is exactly
  what `07_train.py` printed during the real Sweep A run.** This
  strengthens point 4's conclusion (repetition across citations was real,
  not copy-paste drift) and narrows the discrepancy to something in the
  re-evaluation path itself (Step 1/2/3 above), not the original number.
  Investigation is still paused, by user decision — this correction is
  recorded for whoever resumes it next, not acted on further here.

## Next steps, in priority order

1. **Check whether my own `trainer.py` edit caused this, independent of
   the DDP-vs-standalone question.** This is the highest-priority check
   and hasn't been done yet — the timing change was made *immediately*
   before the discrepancy was discovered, so it's a live suspect despite
   looking purely additive. `git stash` the `trainer.py` change (or
   manually diff back to the pre-timing version), re-run evaluation on
   `attention_pw05` (or better, a checkpoint that hasn't been touched yet —
   see the note on backups below) using the *original* unmodified
   `evaluate_test`, standalone/single-GPU, and see if it reproduces 0.8926
   or still gives ~0.8898. If it still gives ~0.8898, the timing edit is
   exonerated and the cause is something about standalone/single-GPU
   reload itself (go to step 2). If it gives 0.8926, the bug is in the
   timing edit itself — re-examine the warmup block and the
   `torch.cuda.synchronize()` placement very closely; possible that
   calling `iter(loader)` twice (once for warmup, once for the real loop)
   has a side effect worth isolating by removing just the warmup block and
   re-testing.

2. **Test the DDP-vs-standalone hypothesis directly.** The original
   evaluation ran inside the same DDP training process right after
   `trainer.fit()`, using `raw_model = model.module` on rank 0's GPU. My
   script builds a fresh model directly, no DDP wrapper, `torch.device("cuda")`
   (defaults to cuda:0). To test whether DDP context itself matters: launch
   a minimal `torchrun --nproc_per_node=2` script that loads
   `attention_pw05/best_model.pt`, wraps in DDP the same way `07_train.py`
   does (including the `init_sync=False` + manual broadcast workaround —
   see `07_train.py` lines ~310-322), and evaluates on rank 0 only, exactly
   mirroring the original code path. If this reproduces 0.8926, the cause
   is something about DDP wrapping or dual-GPU context specifically
   (possible but would be surprising for an eval-only forward pass in
   `no_grad()` — DDP shouldn't change forward-pass numerics, only
   gradient sync — so this is a lower-probability hypothesis than it might
   seem, worth testing mainly to rule out).

3. **Check `compute_esp_stats` for any non-determinism.** The original
   flow computes `esp_mean, esp_std = compute_esp_stats(train_ds)` fresh
   each training run (`src/data/transform.py`) and saves the result into
   the checkpoint. My script reads the *saved* values from the checkpoint
   directly rather than recomputing — confirmed these match
   (esp_mean=-0.3557, esp_std=4.143, read directly from
   `best_model.pt`). This should mean normalization is identical either
   way, but double check `compute_esp_stats`'s implementation doesn't
   subsample or otherwise introduce any run-dependent randomness that
   could make the *originally saved* mean/std itself not fully
   representative — this seems unlikely but hasn't been directly
   inspected yet.

4. **Set up a clean before/after comparison on an untouched checkpoint.**
   `attention_pw05` was overwritten before a careful diff was taken, which
   was a process mistake — don't repeat it. Before re-running
   `reevaluate_test_timing.py` on any other checkpoint: first `cp -r` its
   `test_metrics.json` and `test_predictions/` to a backup location (e.g.
   `/tmp` or a scratch dir), *then* run the re-evaluation, then diff old
   vs new per-protein predictions directly (not just the aggregate r/rmse)
   to see whether the shift is uniform across all proteins (consistent
   with a systematic numerical/precision effect) or concentrated in a few
   proteins (would point somewhere else entirely, e.g. a data-loading
   discrepancy for specific proteins).

5. **Once the mechanism is understood and fixed (or confirmed benign):**
   decide whether to treat fresh re-evaluation as the new ground truth
   project-wide (requires re-running on all ~39 phase_a-e checkpoints,
   updating every notebook table and the numbers cited in this session's
   decisions) or whether the original in-training numbers remain
   authoritative and re-evaluation should only be used for the new
   inference-timing data specifically (in which case, consider whether
   `evaluate_test` should report *both* the timing and the r/rmse/mae, but
   only the timing fields get merged into the existing `test_metrics.json`
   rather than overwriting the metrics fields too).

## Which checkpoints have been touched

Only **`checkpoints/phase_a/attention_pw05`** — `test_metrics.json` and
`test_predictions/*.npz` overwritten with the 0.8898/2.660 fresh-eval
numbers (twice, both times consistent). Every other checkpoint across
phase_a through phase_e is untouched and still has its original
(non-timing) `test_metrics.json`.

## Relevant numbers for reference

Checkpoint inspected directly (`best_model.pt`):
```
esp_mean: -0.3556644320487976
esp_std: 4.143270492553711
epoch: 120
val_loss: 0.5270394484202067
val_pearson_r: 0.8914906609918654
model_config: {'hidden_dim': 256, 'n_rbf': 16, 'n_heads': 4,
  'n_bond_radial_rounds': 4, 'n_aq_rounds': 4, 'n_qq_rounds': 4,
  'agg': 'multi', 'use_element_embedding': True,
  'use_residue_embedding': True, 'use_bond_edges': True,
  'use_radial_edges': True}
feature_spec: {'query_curvature': False, 'query_normal': False}
```
n_proteins=110 in both old and new test evaluations (test set size itself
is not in question).

## Critical review of this investigation (2026-07-31)

Re-read this doc plus `git diff src/training/trainer.py`, `git diff
pipelines/07_train.py`, `src/training/loss.py`, `src/data/dataset.py`,
`src/data/transform.py`, and `src/models/*.py` end to end. Findings below;
tooling fix (isolated output dir + seeding) applied to
`scripts/reevaluate_test_timing.py` — see its module docstring.

**1. Random seeds — confirmed gap, now fixed, but probably not the root
   cause.** `scripts/reevaluate_test_timing.py` never called
   `random.seed`/`np.random.seed`/`torch.manual_seed`/
   `torch.cuda.manual_seed_all`, unlike `pipelines/07_train.py` (lines
   ~297–305). However: `src/models/{egnn,attention_espn,distance_espn}.py`
   contain no `nn.Dropout`, no `torch.rand*`, no `torch.bernoulli` —
   nothing stochastic in the forward pass. `model.eval()` is called before
   inference. The test split comes from a persisted `split_manifest.json`
   (`load_split_manifest`), not a fresh random split. The transform applied
   to `test_ds` is `NormalizeESP` only — deterministic; `RandomRotation`
   exists in `src/data/transform.py` but is never wired into the eval path
   in either the original or reevaluate code. `DataLoader(num_workers=0)`
   by default, so no worker-process ordering nondeterminism either. Given
   all of that, seeding could not have caused the 0.8926→0.8898 shift —
   but it's now set anyway (`--seed`, default 42, matching `07_train.py`)
   as a zero-cost defensive measure, in case a future change adds
   augmentation or workers to this path.

**2. The biggest gap: Next-step 1 (isolate the trainer.py edit) was never
   actually done.** The "fp32 test" that's cited under "already ruled out"
   only toggled `bf16=False` on the *new* `evaluate_test` — it still ran
   with the new warmup block and the new `torch.autocast(..., enabled=False)`
   / `pred.float()` wrapping. It did not run the pre-diff `evaluate_test`
   (`git show HEAD:src/training/trainer.py`, no warmup, no autocast, no
   `.float()` call at all). These are not the same test. Before going to
   Step 2/3, actually do Step 1 as written: `git stash` (or a worktree
   copy of pre-diff `trainer.py`), reload a checkpoint through the
   *original* `evaluate_test`, and compare. This is the cheapest
   experiment available and hasn't been run yet despite being marked
   priority 1.

**3. The doc's diff description undersells what actually changed.** Item 1
   under "What was built" describes the `trainer.py` edit as "should be
   purely additive (timing instrumentation only)." The actual diff also
   adds: always-on EMA shadow weights (`_ema_state`, decayed every
   optimizer step, swapped in for `val_epoch` and for what gets written to
   `best_model.pt`), and `torch.autocast(bf16=...)` wrapped around both
   `train_epoch` and `val_epoch`. Those two are unrelated to the timing
   goal and unrelated to *this specific* checkpoint's discrepancy (EMA/
   autocast-in-training only affect weights produced by *future* training
   runs — `attention_pw05`'s `best_model.pt` was already on disk and is
   only ever read, never regenerated, by `reevaluate_test_timing.py`). But
   the "purely additive" framing is inaccurate, and inaccurate framing in
   this doc is exactly the kind of thing that causes a future session to
   under-scope its own re-check. Worth a corrected one-line description if
   this doc is edited again.

   Separately, `src/training/loss.py`'s `ESPLoss` docstring now claims
   per-protein MSE averaging is "the standard as of the EMA/protein-weighted
   baseline adopted in Sweep A, not a configurable option" — but Sweep A
   (`attention_pw05`, trained Jul 27) predates this EMA code, which is
   still uncommitted. `git diff pipelines/07_train.py` also shows the old
   `--protein-weighted` CLI flag and the `protein_weighted=` kwarg to
   `ESPLoss` were silently dropped in the same diff. This doesn't affect
   `evaluate_test`'s r/rmse (those are computed directly from
   predictions, not through `loss_fn`) so it's not a suspect for the
   discrepancy — but the docstring is asserting project history that
   doesn't match what's on disk, which is worth fixing so it doesn't get
   taken at face value later.

**4. "Not a stale/misremembered figure" is weaker evidence than the doc
   treats it as.** The reasoning is that 0.8926 appeared consistently
   across many separate tool-call results and notebook tables. But
   Jupyter notebooks persist stale cell *outputs* on disk — a later
   notebook's "table of all baseline/phase D/phase E models" could have
   been built by reading an earlier notebook's already-rendered output or
   copying a number forward, rather than by re-reading
   `test_metrics.json` fresh each time. Repetition across citations is not
   the same as repetition across independent re-executions. If this
   becomes relevant again, grep the citing notebooks
   (`07_loss_function_sweep.ipynb` and later phase_b–e notebooks) for
   whether each 0.8926 occurrence is a freshly-executed read of
   `test_metrics.json` (or a value derived from one in the same kernel
   session) versus a hardcoded/copy-pasted number in a markdown or
   pre-computed table cell.

**5. The data loss is permanent, not just inconvenient.** `checkpoints/`
   lives at `/home/student/thesis/checkpoints`, entirely outside this git
   repository (`/home/student/thesis/ProteinStructure_ESP_Thesis`). It was
   never version-controlled, so the original `attention_pw05/
   test_metrics.json` and `test_predictions/*.npz` are not recoverable via
   `git`, full stop — worth being explicit about this so nobody spends
   time trying `git checkout` / reflog on that path.

**6. Tooling fix applied.** `scripts/reevaluate_test_timing.py` now:
   - defaults to writing `test_metrics.json` + `test_predictions/` under
     `model_eval/reeval_test_timing/<checkpoint_name>/` instead of into the
     source checkpoint directory (`checkpoint_dir`/`predictions_dir` passed
     to `evaluate_test` point at the new location; `best_model.pt` is only
     ever read).
   - requires an explicit `--in-place` flag to restore the old
     overwrite-the-checkpoint-dir behavior, so a repeat of the
     `attention_pw05` incident requires deliberate opt-in, not the default
     code path.
   - sets seeds via `--seed` (default 42) per point 1 above.

   This directly enables Step 4 from "Next steps" (clean before/after
   comparison) without a manual `cp -r` step, and makes it safe to run
   this script across all ~39 checkpoints to gather timing data without
   risking another silent metrics overwrite while the r/rmse discrepancy
   is still unexplained.
