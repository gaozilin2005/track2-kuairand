# Final Submission & Results Summary

## Final model output

**`submission_pure_agent.csv`** — the officially-designated final submission for the
required benchmark (KuaiRand-Pure), produced by `agent_loop.py`'s own converged
result (Run 3, iteration 8: `pairwise_watchtime`, `k=4`, `lr=0.001`, `aux_weight=1`),
trained with early stopping on validation, scored once on test. Format/alignment
validated against all 170,588 test rows via `submit.py`'s checker.

We did not submit the hand-driven ensemble (`submission_pure_ens.csv`, test primary
0.6034) as the official result, even though it scores higher, because it was produced
by human-guided research (`RUN_LOG.md`), not by the autonomous agent being judged —
see `PROJECT_DESCRIPTION.md` and `AGENT_VS_MANUAL.md` for why we keep the two tracks
distinct. It remains in the repository as a secondary reference, clearly labeled.

**Bonus benchmarks:** `agent_loop.py` was not pointed at KuaiRand-1K/27K in this
submission — its action space and prompt are scoped to KuaiRand-Pure only. The
hand-driven track separately ran both (see `RUN_LOG.md`), but since those runs weren't
produced by the autonomous agent either, we aren't submitting them as bonus results for
*this* deliverable.

## Three agent runs exist — here's why the third is the one submitted

Each run fixed a specific, diagnosed problem in the one before it, rather than being
independent attempts:

| | Run 1 (Haiku, ungrounded) | Run 2 (Sonnet, grounded) | **Run 3 (Sonnet, grounded + 2 harness fixes — submitted)** |
|---|---|---|---|
| Hypothesis quality | Generic docstring restatement | Grounded, cites facts/assumptions | Same grounding, plus explicit stale-dimension tracking |
| Iterations to convergence | 5 | 4 | 8 |
| Best config | `pairwise_combined`, k=16 | `pairwise_multitask`, k=32 | `pairwise_watchtime`, k=4, lr=0.001 |
| Test primary (5-seed) | 0.6012 ± 0.0005 | 0.6006 ± 0.0005 | **0.6016 ± 0.0003** |

Run 2's grounding improved reasoning quality dramatically but left the score
unchanged, because it never revisited `k`/`lr` after iteration 1 — a hyperparameter-
anchoring failure diagnosed via a stability check on an unrelated experiment
(`AGENT_WIDER_ACTION_SPACE.md`). Run 3 adds two harness fixes: (1) an explicit
parameter-coverage summary in every prompt plus a required `dimension_check` field,
forcing the model to notice and address stale dimensions instead of drifting toward
whichever axis it already has the most reasoning material for; (2) a shortlist
confirmation step — the harness tracks the top 3 distinct configs by single-seed
valid score and runs the *full* 5-seed confirmation on all of them, not just the
nominal best, because once the search got good its top candidates clustered within
0.0003 of each other — tighter than this project's own single-seed noise floor of
0.0004, meaning a single-seed "winner" pick was close to a coin flip. Both fixes are
documented and tested in `RUN_AND_ITERATION_LOG.md`, including the specific moment
the shortlist step caught a real ranking flip (the single-seed nominee was not the
5-seed winner).

Run 3's result is a genuine, significant improvement over Run 2 (+0.0010, 3.83
standard errors) and lands statistically indistinguishable from the hand-driven
track's own long-established single-model default (0.6017 ± 0.0004, −0.45 SE) — after
two harness fixes, not a policy or prompt change to the model's task, the autonomous
agent's own converged result now matches four days of literature-guided human
research's single-model best.

## Results table — required benchmark (KuaiRand-Pure)

Validation-best checkpoint (Run 3, iteration 8), evaluated once on test:

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official baseline (test) | 0.6610 | 0.5282 | 0.5946 |
| **Agent's converged result — validation** | 0.6754 | 0.5384 | 0.6069 |
| **Agent's converged result — test** | **0.6701** ¹ | **0.5331** ¹ | **0.6016** ¹ |
| **Δ vs. baseline (test, per scoring formula)** | **+0.0091** | **+0.0049** | **+0.0070** |

¹ 5-seed mean (± 0.0003–0.0005) from the shortlist-winning configuration's final
confirmation. The single seed-0 run underlying `submission_pure_agent.csv` scores
GAUC 0.6701 / nDCG@5 0.5329 / primary 0.6015, within the same spread.

`score_dataset = mean(delta(GAUC), delta(nDCG@5)) = mean(0.0091, 0.0049) = +0.0070`.
Exceeds zero, clearing the Feasibility & Practicality quality gate.

## Resource usage to reach the converged result

| Metric | Value |
|---|---|
| Iterations used | **8** (of the 50-iteration cap) |
| Total tokens (agent's LLM calls) | **22,748** (28 input + 22,720 output) |
| Total LLM cost | $0.885 (informational — see billing note below) |
| Agent wall-clock (search loop, to convergence) | **815s** (~14 min) |
| Agent wall-clock (incl. shortlist confirmation: 3 configs × 5 seeds) | ~1,900s (~32 min, estimated from per-run timing) |
| GPU-hours | **0** — single-core CPU throughout |
| Manual interventions | **1** (launching the script; 0 afterward) |

Convergence was triggered by the code-enforced rule (ε=0.002 over the last N=5
iterations), not the iteration cap or wall-clock ceiling — 42 iterations and ample
wall-clock budget remained unused when the run stopped itself. Two calls failed and
recovered automatically mid-run (empty API responses, exit 1) — logged and routed
around without human intervention, visible directly in `AGENT_LOG.md` iterations 1
and 6 (of the raw, unrenumbered log).

**Note on LLM billing:** the agent's proposal calls authenticate through the same
Claude Code login used for this development session (no separate `ANTHROPIC_API_KEY`
was configured), so the cost figures above are notional API-equivalent costs drawn
from that account's usage, not necessarily separately itemized charges.

Reproduce end to end: `python3 agent_loop.py --max_iterations 20 --max_wallclock_s
2400 --final_seeds 5 --shortlist_k 3` (defaults to `sonnet`, `patience_n=5`; will
converge well before the 20-iteration soft cap, per the behavior above).
