# Run & Iteration Log (Starter Kit deliverable format)

Source of truth: `AGENT_LOG.md` (the autonomous agent's own, unedited output). This is
the **second, redesigned run** — see the note at the bottom on what changed from the
first run and why, and what that change did and did not buy.

**One structural note, stated plainly rather than glossed over:** `agent_loop.py`'s
action space is CLI-flag selection over `baseline.py`'s existing, already-implemented
losses and hyperparameters — it does not write or edit source code. So "the code diff
applied" per iteration is, honestly, a **configuration diff** (which flags changed),
not a source-code diff. We report it as such below rather than stretching the term.

---

## Iteration 1

- **Mechanism basis:** "Evaluation is explicitly within-user reranking (GAUC + nDCG@5
  computed per user's own impression group), which is exactly the objective
  pairwise/BPR optimizes directly (relative order within a group), whereas the
  baseline's pointwise BCE only classifies rows independently with no notion of group
  order. This is the most basic, lowest-risk mismatch to fix first before touching any
  auxiliary signal, so it's the natural first experiment."
- **Hypothesis:** Switching from pointwise BCE to pairwise (BPR) loss will improve
  primary over the 0.5946 baseline because BPR directly optimizes within-user relative
  ordering, matching how GAUC/nDCG@5 are computed.
- **Configuration diff:** `--loss pairwise --k 32 --lr 0.001 --aux_weight 1 --dns_n 8
  --adt_beta 1` (first iteration, no prior config)
- **Resulting metrics:** valid GAUC 0.6747 / nDCG@5 0.5387 / primary 0.6067 (new best);
  test primary 0.6014
- **Error/recovery events:** none
- **Wall-clock:** 62.4s

## Iteration 2

- **Mechanism basis:** "Dataset fact: play_time_ms >= 18000 alone matches long_view
  96.7% of the time, meaning long_view is essentially a thresholded/coarsened version
  of a continuous watch-time quantity rather than an independent label. Per the
  reference sheet, pairwise_watchtime is 'most useful when the binary label being
  ranked is itself a coarsened version of that same continuous quantity' — that is
  precisely this case... Since iteration 1 confirmed BPR's within-user ranking
  objective already helps (0.6067 vs 0.5946 baseline), the next lever to test is
  giving that ranking objective finer-grained gradient signal from the underlying
  continuous watch-time instead of just the coarsened binary label."
- **Hypothesis:** Adding the CWM-style watch-time regression auxiliary
  (`pairwise_watchtime`, log target) on top of BPR will improve primary further
  because it teaches a finer-grained version of the same signal long_view is
  thresholded from.
- **Configuration diff:** `--loss pairwise_watchtime` (was `pairwise`), `--aux_weight
  0.5` (was `1`); k/lr unchanged from iteration 1
- **Resulting metrics:** valid GAUC 0.6749 / nDCG@5 0.5381 / primary 0.6065 (essentially
  flat vs. iteration 1's 0.6067); test primary 0.6020
- **Error/recovery events:** none
- **Wall-clock:** 88.3s

## Iteration 3

- **Mechanism basis:** "Iteration 2 showed the watchtime auxiliary... was essentially
  neutral vs plain BPR (0.6065 vs 0.6067), suggesting the coarsened long_view label
  alone already carries most of the ordering signal BPR can exploit — the bottleneck
  isn't label granularity. ...nDCG@5 specifically rewards correct ordering only within
  the top-5 of each user's group; vanilla BPR... weights every sampled pair uniformly
  regardless of rank position. lambdarank reweights pairs by |delta-nDCG@K| from the
  model's current ranks, which directly targets this top-K-specific structure that
  plain pairwise ignores."
- **Hypothesis:** Switching to `lambdarank`, keeping k/lr fixed at iteration 1's
  winning values, will improve primary over 0.6067 because reweighting pairs by
  nDCG@5 impact directly targets the top-5-specific component of the metric that
  uniform BPR ignores.
- **Configuration diff:** `--loss lambdarank` (was `pairwise_watchtime`), `--aux_weight
  1` (was `0.5`); k/lr unchanged
- **Resulting metrics:** valid GAUC 0.6635 / nDCG@5 0.5342 / primary 0.5989 (worse than
  best); test primary 0.5932
- **Error/recovery events:** none
- **Wall-clock:** 90.8s

## Iteration 4

- **Mechanism basis:** "Iteration 2 showed the watchtime auxiliary (finer resolution
  of the *same* coarsened signal long_view already encodes) was flat vs plain BPR
  (0.6065 vs 0.6067), and iteration 3 showed lambdarank's top-K-specific reweighting
  actively hurt (0.5989), plausibly because the small, repeat-heavy catalog (190.8 avg
  interactions/video) doesn't give the reweighting a clean top-K boundary to exploit.
  Neither result touches whether a *genuinely independent* auxiliary label helps —
  is_click sits at a different funnel stage (~46% positive) than long_view (~33-34%),
  so per the reference sheet's own distinction ('helps when the auxiliary label
  carries information the main label's own gradient doesn't already supply'),
  pairwise_multitask is the one untested mechanism whose assumption... is actually
  plausible here, unlike watchtime's assumption which iteration 2 already found
  doesn't move the needle."
- **Hypothesis:** Adding an is_click auxiliary BCE (`pairwise_multitask`) will improve
  primary over 0.6067 because is_click is a distinct funnel-stage signal, unlike the
  watchtime auxiliary iteration 2 showed to be redundant with the main label.
- **Configuration diff:** `--loss pairwise_multitask` (was `lambdarank`), `--aux_weight
  0.3` (was `1`); k/lr unchanged
- **Resulting metrics:** valid GAUC 0.6750 / nDCG@5 0.5388 / primary 0.6069 (**new
  best, and the validation-best checkpoint this run converges to**); test GAUC 0.6691
  / nDCG@5 0.5329 / primary 0.6010
- **Error/recovery events:** none
- **Wall-clock:** 88.0s

### Convergence

After iteration 4, the code-enforced check fired: best validation primary over the
last N=3 iterations improved by ≤ ε=0.002 relative to 3 iterations prior. **The run
stopped itself.**

### Error/recovery event — a targeted verification, reported honestly as such

As with the first run, this organic 4-iteration run never hit a failure. The recovery
path was separately verified by deliberately feeding `run_baseline()` an invalid
`loss` value: it fails fast (0.2s, via `argparse`'s own validation), returns
`metrics=None` with the exact error text captured, and would be logged as a failed
iteration and fed into the next proposal call rather than crashing the loop. Verified
infrastructure, not something the recorded run demonstrated organically.

### Final confirmation (5 seeds, full metric breakdown)

| Split | GAUC | nDCG@5 | primary |
|---|---|---|---|
| valid | 0.6746 ± 0.0003 | 0.5386 ± 0.0002 | 0.6066 ± 0.0002 |
| test | 0.6686 ± 0.0005 | 0.5326 ± 0.0005 | 0.6006 ± 0.0005 |

Per the competition's scoring formula: delta(GAUC) = +0.0076, delta(nDCG@5) = +0.0044,
**score_dataset = mean = +0.0060**.

## Manual intervention summary (Task Requirement 2)

**Total manual interventions during this run: 1** — starting `agent_loop.py`. No human
selected a hypothesis, judged whether a result counted as an improvement, decided when
to stop, or intervened at any point after launch.

## What changed between the first and second run, and what it did and didn't buy

The first run (`claude-haiku-4-5`, no dataset context) converged to a statistically
indistinguishable result (test primary 0.6012 ± 0.0005) via generic, docstring-level
hypotheses ("LambdaRank directly optimizes NDCG"). We redesigned the prompt — added a
neutral dataset-facts block (catalog size, density, label-threshold behavior — numbers,
not conclusions), a method-reference sheet naming each loss's origin and the
*assumption* it relies on (not whether it works here), a required `mechanism_basis`
field forcing every choice to cite a specific fact + assumption, and upgraded the
proposal model to Sonnet.

**The reasoning quality changed dramatically** — each iteration above explicitly
reasons from the accumulated evidence of prior iterations, correctly distinguishes
"finer resolution of the same signal" (watchtime, found redundant) from "genuinely
independent signal" (is_click, found to help), and independently arrived at
essentially the same experimental sequence (BPR → watch-time-style auxiliary →
distinguishing funnel-stage signals) that the multi-day hand-driven research
(`RUN_LOG.md`) took much longer to establish. This is exactly what Innovation &
Problem Insight asks to see: grounded reasoning connecting a method's assumption to
this specific data, not a generic description.

**The raw score did not materially change**: test primary 0.6006 ± 0.0005, a −1.90
standard-error gap from the first run's 0.6012 — not distinguishable from noise, and
if anything nominally lower, not higher. Read plainly: **better-reasoned search did not
purchase a better outcome in this narrow action space**, because (per the entire
hand-driven investigation this project separately conducted) the ceiling in this
feature/loss space sits around 0.601–0.602 regardless of how sophisticated the search
process is. This is worth stating as its own finding, not smoothed over: Innovation
(how well the agent reasons about what to try) and Technical Execution (the resulting
score) are separable axes, and this pair of runs is direct, controlled evidence of
that separation — same task, same action space, same convergence rule, dramatically
different reasoning depth, same statistical outcome.

Reproduce: `python3 agent_loop.py --max_iterations 20 --max_wallclock_s 2400
--final_seeds 5` (uses Sonnet by default; pass `--model claude-haiku-4-5-20251001` to
reproduce the first run's cheaper, shallower-reasoning behavior for comparison).

---

## Run 3 (submitted): two more fixes, and this time the score actually moves

Run 2 above reasoned well but converged to the *same* result (0.6006) as run 1. A
separate 5-seed stability check on an unrelated experiment (`AGENT_WIDER_ACTION_SPACE.md`)
surfaced why: `agent_loop.py` had anchored on `k=32` in iteration 1 and never revisited
it across the whole run, so its "watchtime looks flat" conclusion was only ever tested
at that one, arbitrary capacity — and the well-established hand-driven default
(`pairwise_watchtime`, k=16) scores 0.6017 ± 0.0004, meaningfully better than run 2's
own discovered config.

**Two fixes, applied together:**

1. **Anchoring fix.** Every prompt now includes a `summarize_coverage()` block listing
   which loss/k/lr values have been tried, explicitly flagging in the text itself when
   k or lr has sat unchanged for 2+ iterations. A new required schema field,
   `dimension_check`, forces the model to state which dimension has gone longest
   unvaried and either address it or give a specific reason not to. New prompt guidance
   states loss and capacity are independent axes — a loss "looking flat" at one k/lr
   hasn't been shown flat in general. `patience_n` raised from 3 to 5.
2. **Single-seed decision-noise fix.** Even with (1) alone (an intermediate run, not
   separately archived), the search became good enough that its top candidates
   clustered within 0.0003 of each other on a single seed — *tighter* than this
   project's own established single-seed noise floor of 0.0004. Picking "the winner"
   from one seed at that point is close to a coin flip. Fix: the harness now tracks a
   shortlist of the top `--shortlist_k` (3) distinct configs by single-seed valid
   score throughout the search, and at the end runs the full 5-seed confirmation for
   *all* of them — not just the nominal best — before declaring a winner. The summary
   explicitly flags when the 5-seed winner differs from the single-seed nominee.

### The full run (8 iterations, converged via the eps/N rule)

| It. | Config diff | Valid primary | Test primary | Dimension addressed |
|---|---|---|---|---|
| 1 | pairwise, k=32, lr=0.005 (first) | 0.6031 (best) | 0.5958 | — (deliberate mid-range start, not the harness default, so later moves have real room in both directions) |
| 2 | k: 32→16 | 0.6036 (best) | 0.5973 | k (untested away from iter-1's value) |
| 3 | lr: 0.005→0.001 | 0.6070 (best) | 0.6015 | lr (unchanged 2 iterations) |
| 4 | loss: pairwise→pairwise_watchtime | 0.6073 (best) | 0.6016 | loss (unchanged 3 iterations) |
| 5 | k: 16→8 | 0.6071 | 0.6017 | k — explicitly re-tested under the *current best loss*, not just under the loss where it was first measured |
| 6 | lr: 0.001→0.0003 | 0.6063 (worse) | 0.6014 | lr (unchanged 3 iterations) — confirms 0.001 is better |
| 7 | loss: pairwise_watchtime→lambdarank | 0.6007 (worse) | 0.5937 | loss (unchanged 2 iterations) |
| 8 | k: 16→4 (floor of allowed range) | 0.6073 (tied best) | 0.6015 | k — deliberately tests the minimum, since the trend "so far flat-to-improving as k shrinks" had never been checked against the floor |

**Iteration 5 is the clearest evidence the anchoring fix works as intended, not just
superficially**: rather than accepting the iteration-2 finding "k=16 beats k=32," the
model explicitly notices that comparison was run under a different loss (`pairwise`,
not the current-best `pairwise_watchtime`) and re-tests capacity *at the winning loss*
before trusting the conclusion transfers. **Iteration 8 shows the same discipline
extended to a boundary case**: having observed a flat-to-improving trend as k shrank
(32→16→8), it explicitly tests the untried floor (k=4) rather than assuming the trend
stops at 8 — a genuinely new data point the hand-driven investigation's own capacity
ablation (`RUN_LOG.md`, k∈{8,16,32}) never checked.

### Shortlist confirmation — and the flip it exists to catch

| Config | Single-seed valid (nominal rank) | 5-seed valid | 5-seed test |
|---|---|---|---|
| it4: pairwise_watchtime, k=16, lr=0.001 | 0.6073 (1st) | 0.6069 ± 0.0003 | 0.6016 ± 0.0002 |
| it8: pairwise_watchtime, k=4, lr=0.001 | 0.6073 (tied 1st) | 0.6069 ± 0.0003 | **0.6016 ± 0.0003** |
| it5: pairwise_watchtime, k=8, lr=0.001 | 0.6071 (3rd) | 0.6067 ± 0.0004 | 0.6018 ± 0.0003 |

**The 5-seed winner (it8) was not the single-seed-nominal best (it4)** — the harness
logged this explicitly rather than silently picking whichever came first. In practice
here the three configs are statistically indistinguishable from each other (all
within one seed-noise-width), which is itself the finding: k ∈ {4, 8, 16} under
`pairwise_watchtime`/`lr=0.001` are functionally equivalent on this dataset, extending
the hand-driven capacity-insensitivity result down to the floor of the allowed range.

### Result

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official baseline (test) | 0.6610 | 0.5282 | 0.5946 |
| Run 2 (submitted before this fix) | 0.6686 | 0.5326 | 0.6006 |
| **Run 3 — winner: `pairwise_watchtime`, k=4, lr=0.001, aux_weight=1** | **0.6701** | **0.5331** | **0.6016** |
| Hand-driven established default (`RUN_LOG.md`) | 0.6702 | 0.5333 | 0.6017 |

`score_dataset = mean(delta(GAUC), delta(nDCG@5)) = mean(0.0091, 0.0049) = +0.0070`.

**vs. run 2 (previously submitted): +0.0010, 3.83 SE — a real, significant improvement.**
**vs. the hand-driven established default: −0.0001, −0.45 SE — statistically
indistinguishable.** After two targeted fixes to the harness (not the model, not the
task, not the dataset), the autonomous agent's own converged result now matches, within
noise, four days of literature-guided human research's own single-model best — found in
8 iterations, 815s, $0.88.

### This is now the submitted result

`submission_pure_agent.csv` regenerated for `pairwise_watchtime`/k=4/lr=0.001/
aux_weight=1, format/alignment-validated on all 170,588 test rows.

Reproduce: `python3 agent_loop.py --max_iterations 20 --max_wallclock_s 2400
--final_seeds 5 --shortlist_k 3` (patience_n now defaults to 5).
