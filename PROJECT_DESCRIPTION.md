# Project Description

## How this solution addresses the problem statement

The problem statement asks for an **autonomous ML research agent**: a system that itself
reads the problem, proposes hypotheses, implements and runs them, evaluates results, and
decides what to try next — with minimal human intervention — converging per a fixed rule
(ε=0.002 improvement over the last N=3 iterations, or a 50-iteration / 6-hour cap).

This submission has two distinct parts, kept deliberately separate rather than conflated:

1. **`agent_loop.py` — the autonomous agent itself.** A Python driver that calls Claude
   headlessly (via the local Claude Code binary in `--print` mode — genuine, separately
   metered API calls) once per iteration to propose a configuration from a fixed action
   space (`baseline.py`'s existing `--loss`, `--wt_target`, `--k`, `--lr`, `--aux_weight`,
   `--dns_n`, `--adt_beta` flags — no new code is written by the agent). Each proposal is
   run as a real training job, its validation/test metrics are read back, and the
   competition's own convergence rule is checked in Python code — not delegated to the
   model's judgment. A logged run converged in 5 iterations with **zero manual
   intervention after launch**, landing on `pairwise_combined` (aux_weight=0.3): test
   GAUC 0.6694 / nDCG@5 0.5329, a **+0.0066** score_dataset over the official baseline,
   at 963s wall-clock and 14,715 total tokens.

   The action space was deliberately kept narrow — CLI-flag selection, not free-form code
   generation — as a considered reliability trade-off: a tight space that reliably runs to
   convergence beats an ambitious one that breaks mid-loop, and it isolates what's being
   measured (autonomous propose→test→evaluate→decide iteration) from open-ended
   architecture search.

2. **A separate, hand-driven research track** (`RUN_LOG.md`, `baseline.py`,
   `ablation_*.py`, `sequence_model.py`, `deepfm_model.py`, `finalmlp_model.py`,
   `lightgcn_model.py`, the ensembling scripts, and the bonus-benchmark runners), where a
   human (using Claude Code interactively as a tool, not as the autonomous loop) explored
   a much wider space — architectures (DIN, BST, DeepFM, FinalMLP, LightGCN), training
   strategies (DNS, ADT, temporal reweighting), auxiliary objectives (CWM watch-time, RAD
   quantile targets), and four ensembling strategies. This reached test primary **0.6034**
   (heterogeneous ensemble of BST + two FM variants) — the best number found in this
   project overall — and produced `AGENT_VS_MANUAL.md`, an honest comparison showing the
   autonomous agent's own genuine contribution (once the shared feature pipeline it
   inherits is factored out) matches the same order of magnitude the hand-driven search
   found for the analogous intervention.

We consider the coexistence of both tracks a feature, not a compromise: the hand-driven
work establishes what headroom actually exists and where the ceiling sits (useful ground
truth for judging whether the autonomous loop is finding real signal or noise), while
`agent_loop.py` is the actual answer to the problem statement's core ask.

## Development tools

- **Claude Code** (Anthropic) — both as the interactive tool driving the hand-built
  research track, and as the invoked-headlessly LLM inside `agent_loop.py` itself.
- Standard shell/Python tooling on macOS (local development) and a university SLURM GPU
  cluster (for BST and the KuaiRand-27K bonus run).

## APIs used

- **Claude (via the local Claude Code CLI binary, headless `--print` mode)** — the only
  API call the autonomous agent makes. Model used for agent proposal calls:
  `claude-haiku-4-5`. No other external API was used (no OpenAI, no Google, etc.) and no
  data outside the officially provided KuaiRand files was used.

## Libraries and frameworks

- **NumPy** — the entire core baseline (Factorization Machine, all loss variants, BPR
  sampling, evaluation) is hand-implemented in pure NumPy, matching the starter kit's
  original numpy-only constraint.
- **PyTorch** — added later, isolated to sequence/graph/deep architectures where
  hand-deriving backprop stopped being worthwhile: `sequence_model.py` (DIN, BST),
  `deepfm_model.py`, `finalmlp_model.py`, `lightgcn_model.py`, and `bonus_fm_torch.py`
  (a sparse-embedding FM for the KuaiRand-27K bonus benchmark, needed because dense Adam
  updates over a ~41M-row embedding table are computationally infeasible regardless of
  hardware).
- No pandas, scikit-learn, or other ML framework — kept intentionally minimal.

## Datasets and assets

- **KuaiRand-Pure** (required benchmark) — the official starter-kit dataset, including
  the standard interaction logs and, for one ablation, the previously-unused official
  `user_features_pure.csv` and `video_features_statistic_pure.csv` side-information files.
- **KuaiRand-1K** and **KuaiRand-27K** (bonus benchmarks) — same task and metrics as
  Pure, run via a memory-efficient columnar loader built specifically because the
  starter kit's list-of-tuples approach doesn't scale to these sizes on commodity
  hardware.
- No external or synthetic data of any kind was used, per the competition's rules.
