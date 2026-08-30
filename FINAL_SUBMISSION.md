# Final Submission & Results Summary

## Final model output

**`submission_pure_agent.csv`** — the officially-designated final submission for the
required benchmark (KuaiRand-Pure), produced by `agent_loop.py`'s own converged
result (iteration 2: `pairwise_combined`, `aux_weight=0.3`, `k=16`, `lr=0.001`), trained
with early stopping on validation, scored once on test. Format/alignment validated
against all 170,588 test rows via `submit.py`'s checker.

We did not submit the hand-driven ensemble (`submission_pure_ens.csv`, test primary
0.6034) as the official result, even though it scores higher, because it was produced
by human-guided research (`RUN_LOG.md`), not by the autonomous agent being judged —
see `PROJECT_DESCRIPTION.md` and `AGENT_VS_MANUAL.md` for why we keep the two tracks
distinct rather than presenting the better number as if the agent found it. It's
included in the repository as a secondary reference for anyone interested in the
project's overall ceiling, clearly labeled as such.

**Bonus benchmarks:** `agent_loop.py` was not pointed at KuaiRand-1K/27K in this
submission — its action space and prompt are scoped to KuaiRand-Pure only. The
hand-driven track separately ran both bonus datasets (KuaiRand-1K: valid 0.6208/test
0.6194; KuaiRand-27K: valid 0.6158/test 0.6057 — see `RUN_LOG.md`), but since those runs
weren't produced by the autonomous agent either, we aren't submitting them as bonus
results for *this* deliverable. Pointing `agent_loop.py` at the bonus datasets (it would
need `--data_dir` and the `data_large.py` columnar loader wired in) is a clear, scoped
next step we didn't do given time constraints — noted honestly here rather than
silently claimed.

## Results table — required benchmark (KuaiRand-Pure)

Validation-best checkpoint (iteration 2 of the agent's converged run), evaluated once
on test:

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official baseline (test) | 0.6610 | 0.5282 | 0.5946 |
| **Agent's converged result — validation** | 0.6753 | 0.5389 | 0.6071 |
| **Agent's converged result — test** | **0.6694** ¹ | **0.5329** ¹ | **0.6012** ¹ |
| **Δ vs. baseline (test, per scoring formula)** | **+0.0084** | **+0.0047** | **+0.0066** |

¹ Test row shown as the 5-seed mean (± 0.0005 on both metrics) from the final
confirmation step, matching this project's standard reporting convention. The single
seed-0 run underlying `submission_pure_agent.csv` itself scores GAUC 0.6702 / nDCG@5
0.5337 / primary 0.6019 — within the same 5-seed spread.

Per the competition's scoring formula, `score_dataset = mean(delta(GAUC), delta(nDCG@5))
= mean(0.0084, 0.0047) = +0.0066`. This exceeds zero, clearing the Feasibility &
Practicality quality gate (hidden-test primary must exceed the official baseline).

## Resource usage to reach the converged result

| Metric | Value |
|---|---|
| Iterations used | **5** (of the 50-iteration cap) |
| Total tokens (agent's LLM calls) | **14,715** (50 input + 14,665 output) |
| Total LLM cost | $0.182 (informational — see note on billing below) |
| Agent wall-clock (search loop, to convergence) | **537s** (~9 min) |
| Agent wall-clock (incl. final 5-seed confirmation) | **963s** (~16 min) |
| GPU-hours | **0** — the agent's entire run is single-core CPU |
| Manual interventions | **1** (launching the script; 0 afterward) |

Convergence was triggered by the code-enforced rule (ε=0.002 improvement over the last
N=3 iterations), not the iteration cap or wall-clock ceiling — the run had 45 more
iterations and ~5h51m of budget remaining when it stopped itself.

**Note on LLM billing:** the agent's proposal calls authenticate through the same
Claude Code login used for this development session (no separate `ANTHROPIC_API_KEY`
was configured), so the $0.182 figure is a notional API-equivalent cost drawn from
that account's usage, not necessarily a literal separate charge — see the discussion in
this project's development log if exact separate billing matters for your reporting.

Reproduce end to end: `python3 agent_loop.py --max_iterations 20 --max_wallclock_s
2400 --final_seeds 5` (will converge well before the 20-iteration soft cap used here,
per the behavior above).
