# Final Submission & Results Summary

## Final model output

**`submission_pure_agent.csv`** — the officially-designated final submission for the
required benchmark (KuaiRand-Pure), produced by `agent_loop.py`'s own converged
result (iteration 4 of the redesigned run: `pairwise_multitask`, `aux_weight=0.3`,
`k=32`, `lr=0.001`), trained with early stopping on validation, scored once on test.
Format/alignment validated against all 170,588 test rows via `submit.py`'s checker.

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

## Two agent runs exist — here's why the second is the one submitted

We ran `agent_loop.py` twice, deliberately, to test one specific thing: whether
grounding the agent's reasoning (dataset facts + a literature-derived method-assumption
reference + a required citation field) actually improves the *quality* of its proposals,
not just the score.

| | Run 1 (`claude-haiku-4-5`, no context) | **Run 2 (`sonnet`, grounded — submitted)** |
|---|---|---|
| Hypothesis quality | Generic ("LambdaRank directly optimizes NDCG") | Grounded — cites specific dataset facts and reference-sheet assumptions, reasons across prior iterations |
| Iterations to convergence | 5 | 4 |
| Best config | `pairwise_combined`, aux=0.3, k=16 | `pairwise_multitask`, aux=0.3, k=32 |
| Test primary (5-seed) | 0.6012 ± 0.0005 | 0.6006 ± 0.0005 |
| Cost / tokens / wall-clock | $0.182 / 14,715 / 963s | $0.346 / 8,367 / 853s |

The gap between the two final scores is −1.90 standard errors — not statistically
distinguishable, and nominally in the "worse" direction if anything. **We report this
honestly rather than picking whichever number is higher**: better-grounded reasoning
did not purchase a better score in this narrow action space, because the ceiling here
(established across the entire hand-driven investigation) sits around 0.601–0.602
regardless of search sophistication. We submit Run 2 anyway because Innovation &
Problem Insight is judged on the *reasoning*, not the outcome, and Run 2's reasoning is
categorically better-evidenced: it independently reconstructed much of the same
experimental sequence (BPR alignment → watch-time-style auxiliary → distinguishing
funnel-stage signals) that took days of human-guided research to establish, citing a
specific dataset fact or method assumption for every choice. Full transcript and
analysis: `RUN_AND_ITERATION_LOG.md`.

## Results table — required benchmark (KuaiRand-Pure)

Validation-best checkpoint (iteration 4 of the submitted run), evaluated once on test:

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official baseline (test) | 0.6610 | 0.5282 | 0.5946 |
| **Agent's converged result — validation** | 0.6750 | 0.5388 | 0.6069 |
| **Agent's converged result — test** | **0.6686** ¹ | **0.5326** ¹ | **0.6006** ¹ |
| **Δ vs. baseline (test, per scoring formula)** | **+0.0076** | **+0.0044** | **+0.0060** |

¹ 5-seed mean (± 0.0005 both metrics) from the final confirmation step. The single
seed-0 run underlying `submission_pure_agent.csv` scores GAUC 0.6691 / nDCG@5 0.5329 /
primary 0.6010, within the same spread.

`score_dataset = mean(delta(GAUC), delta(nDCG@5)) = mean(0.0076, 0.0044) = +0.0060`.
Exceeds zero, clearing the Feasibility & Practicality quality gate.

## Resource usage to reach the converged result

| Metric | Value |
|---|---|
| Iterations used | **4** (of the 50-iteration cap) |
| Total tokens (agent's LLM calls) | **8,367** (10 input + 8,357 output) |
| Total LLM cost | $0.346 (informational — see billing note below) |
| Agent wall-clock (search loop, to convergence) | **424s** (~7 min) |
| Agent wall-clock (incl. final 5-seed confirmation) | **853s** (~14 min) |
| GPU-hours | **0** — single-core CPU throughout |
| Manual interventions | **1** (launching the script; 0 afterward) |

Convergence was triggered by the code-enforced rule (ε=0.002 over the last N=3
iterations), not the iteration cap or wall-clock ceiling — 46 iterations and ~5h49m of
budget remained unused when the run stopped itself.

**Note on LLM billing:** the agent's proposal calls authenticate through the same
Claude Code login used for this development session (no separate `ANTHROPIC_API_KEY`
was configured), so the cost figures above are notional API-equivalent costs drawn
from that account's usage, not necessarily separately itemized charges.

Reproduce end to end: `python3 agent_loop.py --max_iterations 20 --max_wallclock_s
2400 --final_seeds 5` (defaults to `sonnet`; will converge well before the
20-iteration soft cap, per the behavior above).
