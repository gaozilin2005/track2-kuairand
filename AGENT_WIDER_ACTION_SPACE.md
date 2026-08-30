# Widening the action space: does architecture choice + ensembling change the ceiling?

`agent_loop.py` (the submitted run) tunes loss functions and hyperparameters over a
single Factorization Machine — a narrow, reliable action space. This document covers a
separate experiment, `agent_loop_ensemble.py`, testing a specific, evidence-grounded
question: the *entire* hand-driven investigation (`RUN_LOG.md`, ~20 independent
single-model levers) converged on the same ~0.601–0.602 ceiling every time; the only
thing that broke past it was heterogeneous ensembling, because it combines models with
different error patterns rather than making any one model individually stronger. Can an
autonomous agent discover that same principle itself, using its own trained models and
its own measurements — not told our finding?

Two runs were needed to answer this honestly.

## Run 1: correct reasoning, cut off one action short

Action space: `train_member` (fm_watchtime / fm_quantile / deepfm / finalmlp /
lightgcn — BST excluded, ~2450s/epoch on this machine's CPU per `RUN_LOG.md`, would
consume the whole wall-clock budget on one training job), `check_correlation` (measure
real pairwise rank correlation between two of its own trained members' validation
predictions), `combine` (exhaustive valid-selected fusion search over everything
trained so far). Members train into a dedicated `agent_scores/`, never touching the
hand-driven track's cached artifacts.

With `patience_n=3` (matching `agent_loop.py`'s default), it: trained `fm_watchtime`
(valid 0.6071) → trained `lightgcn` as "the most structurally distinct member
available" (valid 0.5661) → **measured** their correlation itself (0.6162, moderate) →
combined them (0.6018 — *worse* than fm_watchtime alone) → correctly diagnosed why:
*"decorrelation alone isn't sufficient when one member is too weak individually; it
drags the fusion down more than its diversity helps"* → pivoted to `finalmlp` (valid
0.6066) as a member with "a real shot at both individual strength and structural
diversity, not diversity alone." Then the convergence rule fired — the run had gone 3
scored iterations without exceeding the very first iteration's already-strong score,
even though the most promising next step (measure finalmlp's correlation, then combine)
hadn't happened yet.

Checking directly on its own cached artifacts (no new training, no hint) confirmed the
harness cut it off one action early: `{finalmlp, fm_watchtime}` z-score fusion scores
valid 0.6072 / test 0.6021 — a real, if modest, win over solo fm_watchtime. **This is a
harness-design lesson, not a reasoning failure**: a fixed N=3 patience window fits a
single-shot action space (every iteration is a complete experiment) but not a
multi-step one (train → correlate → combine are separate steps toward one payoff).

## Run 2: fixed the harness (patience_n=6), fresh `agent_scores/`, no manual help

Same action space, same model (Sonnet), fresh scores directory, one change: enough
patience to let a multi-step plan actually complete. 11 iterations, converged via the
same code-enforced rule, $0.918 total cost, 913s wall-clock, **1 manual intervention
(launch), 0 after**.

The full arc, and the parts worth reading closely:

1. **fm_watchtime** (0.6071) → **lightgcn** (0.5661, reasoned as "most structurally
   distinct") → **measured** correlation = 0.6162 → **combined**: 0.6018, worse than
   solo — reproduces run 1 exactly (deterministic given the same seed/config).
2. Correctly generalized from the failure: *"the next member needs a real shot at both
   individual strength and structural diversity, not diversity alone."* Ruled out
   `fm_quantile`/`deepfm` from architecture reasoning alone ("likely high correlation,
   low diversity payoff") and trained **finalmlp** (0.6066).
3. **Measured** fm_watchtime↔finalmlp correlation: **0.9216** — high. Its own stated
   hypothesis going in was *"I expect... lower rank correlation than the lightgcn
   pair"* — the measurement contradicted that prediction outright. It did not get
   stuck defending it: the very next iteration's reasoning starts from the correct,
   updated fact ("nearly redundant, rank corr 0.9216") and proceeds anyway, since
   `combine`'s exhaustive search costs little to run even when a pairing looks
   unpromising. **This is the single most important moment in either run for judging
   real reasoning versus pattern-matching** — a system merely restating expected
   patterns would have skipped or hedged; this one updated cleanly and moved on.
4. **Combine** (all 3 members): found `{finalmlp, fm_watchtime}` z-score, valid
   **0.6072** (new best) / test **0.6021** — the exact result run 1 was one step away
   from, now reached autonomously, with a fairer window.
5. Measured finalmlp↔lightgcn correlation (0.5911) as a cheap check before deciding
   whether a 4th member was worth it, correctly reasoning that the previous combine's
   own exclusion of lightgcn was itself informative ("the search... excluded lightgcn —
   consistent with lightgcn's weak solo score... outweighing its moderate diversity").
6. Trained **deepfm** (0.6058) specifically because it "keeps the FM bilinear term...
   while adding a parallel DNN branch — a real structural addition, not just a
   different auxiliary loss." Measured its correlation with fm_watchtime: **0.9227**
   (this time correctly predicted as similar to finalmlp's, having learned from the
   earlier surprise). Ran combine with all 4 members.
7. **Final result: `{deepfm, fm_watchtime}` raw fusion, valid 0.6073, test 0.6022** —
   the run's best, and where it converged.

## The result (single seed — see the stability check below before trusting this)

| | valid primary | test GAUC | test nDCG@5 | test primary |
|---|---|---|---|---|
| Submitted single-model agent (`agent_loop.py`, 5-seed mean) | 0.6066 | 0.6686 | 0.5326 | 0.6006 |
| **This run's discovered ensemble** (`{deepfm, fm_watchtime}`, raw) | **0.6073** | **0.6708** | **0.5336** | **0.6022** |
| Hand-driven ensemble (`RUN_LOG.md`, includes BST) | 0.6091 | 0.6724 | 0.5344 | 0.6034 |

At the point this document was first written, we reported `score_dataset = +0.0076`,
beating the submitted single-model result by +0.0016 — **on a single seed per
member.** The single biggest gap in that claim was exactly what it looked like: no
variance estimate. We closed it directly rather than leave it as a caveat.

## The 5-seed stability check — the "win" doesn't hold up, and here's why

Added a `--seed` override to `ablation_hetero_ensemble.py`'s `--member` dispatch
(previously hardcoded per member type; defaults unchanged, no existing hand-driven
result affected) and trained `deepfm` + `fm_watchtime` at 4 more seeds:

| | test primary (5-seed) |
|---|---|
| deepfm alone | 0.6007 ± 0.0006 |
| fm_watchtime alone | 0.6017 ± 0.0004 |
| **{deepfm, fm_watchtime} ensemble** | **0.6019 ± 0.0002** |

**Ensemble vs. fm_watchtime alone: +0.0002, 0.98 SE — not significant.** The seed-0
result that looked like a real gain (0.6022 vs. fm_watchtime's 0.6020 that seed) was a
lucky draw: fm_watchtime's own score varies 0.6011–0.6022 across seeds, and combining
it with the weaker, also-noisy deepfm doesn't reliably add anything beyond what
fm_watchtime alone already provides. This is the exact "high correlation with the FM
family, no real ensemble value" finding the hand-driven work already established for
DeepFM/FinalMLP (`RUN_LOG.md`, 0.973 correlation) — now independently reconfirmed by
a completely different, autonomous process.

**A second, more consequential finding fell out of this check.** `fm_watchtime` alone
(0.6017 ± 0.0004) — which matches `baseline.py`'s own long-established default exactly
— is *better* than the submitted single-model agent's own discovered config
(`pairwise_multitask`, k=32, 0.6006 ± 0.0005), and the ensemble's real advantage
(+0.0013, 5.34 SE, genuine) over the submitted result comes entirely from containing
fm_watchtime, not from ensembling. The likely cause: `agent_loop.py`'s run committed
to `k=32` in iteration 1 and never revisited it — its "watchtime looks flat" conclusion
at iteration 2 was tested only at that one, arbitrary, never-revisited k. That's a real
limitation in the single-model agent's own search, not a property of watch-time
auxiliary training — worth documenting as such rather than smoothing over.

## Verdict

**Not promoting `submission_pure_agent_ensemble.csv`.** The gain that motivated it
doesn't clear this project's own significance bar once properly measured. We are also
**not** retroactively substituting the known-good default config for the submitted
single-model result — doing so would repeat, in reverse, the exact mistake avoided
earlier (substituting human knowledge for what the agent's own process actually
converged on). The submitted result stands as `agent_loop.py` converged to it, with
this limitation now documented honestly rather than concealed by a since-retracted
"win."

What the wider action space genuinely demonstrated, independent of the promotion
question: correct diagnostic reasoning throughout (including a clean self-correction
after a wrong correlation prediction), a real harness-design lesson (fixed patience
windows need to fit multi-step action spaces), and — via this stability check — a
second, independently-found confirmation that architectural diversity within the FM
family doesn't buy real ensemble value, plus a concrete, fixable weakness (hyperparameter
anchoring) in the single-model agent's own search discipline.

Reproduce: `rm -rf agent_scores && python3 agent_loop_ensemble.py --max_iterations 15
--max_wallclock_s 3600 --patience_n 6 --device cpu`. Stability check:
`python3 ablation_hetero_ensemble.py --member deepfm --seed <1..4> --scores_dir
stability_scores/seed<N>` (repeat for `fm_watchtime`), then fuse and evaluate.
