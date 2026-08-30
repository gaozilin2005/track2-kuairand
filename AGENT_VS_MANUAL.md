# Autonomous agent run vs. hand-driven exploration — an honest comparison

This file exists because the two are genuinely different deliverables, produced by
different mechanisms, and should not be conflated:

- **`RUN_LOG.md`** — a human (via Claude Code, driven interactively) reading papers,
  forming hypotheses, and deciding what to try next across four days and ~35
  configurations. This is *research done by hand*, using an LLM as a tool.
- **`AGENT_LOG.md`** — a Python driver (`agent_loop.py`) that itself calls Claude
  (headless, via the local Claude Code binary in `--print` mode — real, separately
  metered API calls) in a loop: propose a config → run it → read the result → decide
  what's next → check a fixed convergence rule in code. Once started, no human chose
  any hypothesis, judged any result, or decided when to stop. **This is the actual
  "autonomous ML research agent" deliverable the competition asks for.**

`agent_loop.py` was never shown `RUN_LOG.md`. Its only history is its own log. Any
overlap between what it found and what the hand-driven session found is independent
confirmation, not memorization.

## What the agent did, in full (5 iterations, converged on its own)

| It. | Loss tried | Key hyperparams | Valid primary | Test primary |
|---|---|---|---|---|
| 1 | lambdarank | k=16, lr=0.001, aux=0 | 0.6007 | 0.5937 |
| 2 | **pairwise_combined** | aux_weight=0.3 | **0.6071** (best) | 0.6019 |
| 3 | pairwise_combined | aux_weight=0.5 | 0.6071 (no gain) | 0.6012 |
| 4 | pairwise_dns | dns_n=5 | 0.6022 | 0.5979 |
| 5 | pairwise_adt | adt_beta=0.5 | 0.6071 (no gain) | 0.6007 |

Stopped itself via the competition's own rule: 3 consecutive iterations without
improving validation primary by more than ε=0.002 (all three of iterations 3–5 landed
within 0.0000 of iteration 2's 0.6071). No human called this — it's a Python
`if` statement checked after every iteration.

**Final 5-seed confirmation of its best config** (`pairwise_combined`, aux_weight=0.3):
valid **0.6072 ± 0.0002**, test **0.6012 ± 0.0005**.

## Resource cost (the literal Feasibility deliverable)

- Wall-clock: **963s (16m03s) total** — 537s for the 5-iteration search loop, ~426s
  for the final 5-seed confirmation.
- LLM cost: **$0.182**, 50 input + 14,665 output tokens, `claude-haiku-4-5`, 5 proposal
  calls.
- Manual interventions: **1** — starting the script. Nothing about which hypothesis to
  try, whether a result counted as an improvement, or when to stop was decided by a
  human after that.

## The honest comparison — not just "it beat the baseline"

The printed run summary says the agent's result **beats the official baseline
(0.5946) by +0.0066**. That's true but overstates the agent's own contribution if left
unqualified: its action space (loss function + hyperparameters, via `baseline.py`'s
existing CLI) operates on top of a pipeline that *already* has `prior_exposure` and
`author_recency` baked in as permanent fields (`data.py`'s default `encode()`) — a
feature-engineering win the hand-driven session found and adopted, not something this
agent run touched or gets credit for discovering.

The fairer comparison is against the same fixed pipeline's own **loss-only** baseline —
BPR with no auxiliary task, on the 7-field pipeline, which the hand-driven session
established at **0.6008**:

| Comparison | Δ | Significance |
|---|---|---|
| Agent's result vs. **raw official baseline** (0.5946) | +0.0066 | includes feature gains the agent didn't find |
| Agent's result vs. **fair baseline** (BPR + 7 fields, no aux, 0.6008) | **+0.0004** | 1.40 SE — the agent's *own* genuine contribution |
| Agent's result vs. **hand-driven single-model best** (`pairwise_watchtime`, 0.6017) | −0.0005 | −1.75 SE — statistically indistinguishable |

Read plainly: **within the narrow slice of the problem it was allowed to search — loss
function and its hyperparameters — the agent independently found essentially the same
thing the hand-driven session found**, that combining the main ranking loss with a
watch-time-based auxiliary signal helps by a small, real amount (its own genuine gain,
+0.0004, is the same order of magnitude as the hand-driven session's own +0.0009 for
the analogous single-objective watch-time task). It converged on a slightly different
specific configuration (`pairwise_combined`, which also folds in `is_click`, vs. the
hand-driven session's `pairwise_watchtime` alone) that lands at a statistically
indistinguishable final score.

**What it did *not* do**, worth stating plainly rather than glossing over: it never
tried plain `pairwise` (BPR with no auxiliary task at all) as a sanity checkpoint in
this run — it jumped straight from `lambdarank` to more elaborate multi-task losses.
A cheap "try the simplest lever first" heuristic wasn't something this particular
5-iteration run happened to explore, which is a fair critique of this run's search
strategy, not of the mechanism itself.

## The actual finding

A fully autonomous loop, given a narrow but real action space, a fixed convergence
rule, and zero human intervention after launch, spent 16 minutes and $0.18 to
independently arrive at a result matching — within noise — four days of human-guided,
literature-heavy exploration's single-model best. That's the evidence this mechanism
works, not a claim that 5 iterations of a Haiku-tier model replaces the broader search
(architecture changes, ensembling, bonus datasets) the hand-driven session also did —
those remain outside this run's action space by design (see `agent_loop.py`'s
docstring for that scope decision).

Reproduce: `python3 agent_loop.py --max_iterations 20 --max_wallclock_s 2400
--final_seeds 5` (will very likely converge well before 20 iterations, per the ε/N
rule above).
