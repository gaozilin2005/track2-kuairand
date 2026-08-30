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

## The result

| | valid primary | test GAUC | test nDCG@5 | test primary |
|---|---|---|---|---|
| Submitted single-model agent (`agent_loop.py`, 5-seed mean) | 0.6066 | 0.6686 | 0.5326 | 0.6006 |
| **This run's discovered ensemble** (`{deepfm, fm_watchtime}`, raw) | **0.6073** | **0.6708** | **0.5336** | **0.6022** |
| Hand-driven ensemble (`RUN_LOG.md`, includes BST) | 0.6091 | 0.6724 | 0.5344 | 0.6034 |

Per the scoring formula: `score_dataset = mean(delta(GAUC), delta(nDCG@5)) =
mean(0.0098, 0.0054) = +0.0076` — **beats the currently-submitted single-model
result's +0.0060 by +0.0016**, and comes within 0.0012 of the hand-driven ensemble
despite never having access to BST (excluded for CPU wall-clock reasons) — the one
member that gave the hand-driven work its largest jump. That gap is consistent with,
not contradictory to, everything found here: BST is a genuinely different mechanism
(sequence order), and this run's only two useful contributors (finalmlp, deepfm) both
correlate above 0.92 with fm_watchtime — real but modest variance-reduction gains, not
the bias-reduction gain a truly decorrelated-but-competitive member would give.

**Caveat, stated as plainly as the hand-driven ensemble entry states its own version of
this**: this is a single point estimate (one seed per member), not a 5-seed mean.
Submission file: `submission_pure_agent_ensemble.csv`, format/alignment-validated on
all 170,588 test rows.

## Verdict

Widening the action space to include architecture choice and ensembling **did** move
the ceiling — modestly, autonomously, and via genuinely correct reasoning including a
clean self-correction after a wrong prediction. Whether to promote
`submission_pure_agent_ensemble.csv` over the currently-submitted single-model result
is a decision worth making deliberately rather than silently: it's a stronger number
produced by a wider, more expensive, and (with only 2 runs behind it) less
battle-tested action space than the one already submitted.

Reproduce: `rm -rf agent_scores && python3 agent_loop_ensemble.py --max_iterations 15
--max_wallclock_s 3600 --patience_n 6 --device cpu`
