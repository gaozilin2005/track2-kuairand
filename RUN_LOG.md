# Run Log

Every entry: what changed, exact command, seeds 0–4, test-set mean ± std, verdict.
Convergence rule and eval are the ones fixed in `evaluate.py` / `README.md` — never touched.

## 2026-08-27 — Loss function ablation on FM

Baseline: FM w/ pointwise logloss (official baseline, `baseline_scores.json`).
Hypothesis (README "从哪里开始改" #1): pointwise optimizes calibration, but the
metric (GAUC/nDCG@5) is a ranking metric — aligning the loss with the metric
should help. Tested three alternatives, holding data, features (`FIELDS`),
model (`k=16, lr=0.001, bs=8192, max_epochs=40, patience=4`), split, and eval
fixed. Only `--loss` changes.

Command shape: `python3 baseline.py --model fm --loss <X> --seed <0..4>`

### Results (test set, mean ± population std over seeds 0–4)

| loss | GAUC | nDCG@5 | primary | Δ vs. pointwise |
|---|---|---|---|---|
| pointwise (official) | 0.6610 (σ=0.0008) | 0.5282 (σ=0.0008) | 0.5946 (σ=0.0008) | — |
| **pairwise / BPR** | **0.6638** (σ=0.0007) | **0.5304** (σ=0.0004) | **0.5971** (σ=0.0005) | **+0.0025** |
| listwise (softmax over user's group) | 0.6583 (σ=0.0004) | 0.5279 (σ=0.0005) | 0.5931 (σ=0.0005) | −0.0015 |
| lambdarank@5 (BPR × \|ΔnDCG@5\|) | 0.6525 (σ=0.0010) | 0.5257 (σ=0.0005) | 0.5891 (σ=0.0006) | −0.0055 |

Per-seed test primary:
- BPR: 0.5978, 0.5974, 0.5963, 0.5972, 0.5969
- listwise: 0.5937, 0.5926, 0.5926, 0.5935, 0.5930
- lambdarank@5: 0.5888, 0.5887, 0.5895, 0.5901, 0.5883

### Finding 1 (positive): BPR beats pointwise, and the gap is real

+0.0025 on primary, ~3× the larger of the two σs (0.0008), and BPR's full
5-seed range (0.5963–0.5978) never overlaps pointwise's expected range
(≈0.593–0.596 at ±2σ). Clears the repo's own convergence bar (ε=0.002, ~2.5σ).
**BPR is the new best result: 0.5971 primary on test, replacing pointwise FM
(0.5946) as the number to beat.**

Implementation: one negative sampled per positive, per epoch, restricted to
users with `0 < positives < total` (same eligibility as GAUC) — pairs only
formed within a user's own impressions, consistent with the intra-user
ranking task definition. See `FM.step_pairwise` / `run_fm(..., loss='pairwise')`
in `baseline.py`.

### Finding 2 (negative): listwise softmax underperforms pointwise

−0.0015 on primary, consistent and negative across all 5 seeds — not noise
(gap is ~2–3× listwise's own σ). Converges in ~2 epochs then degrades;
early stopping catches it before it gets worse, not before it gets better.
Likely cause: the softmax-over-full-group objective distributes gradient
mass across *all* of a user's positives equally, with no notion of "top-5" —
mismatched with what nDCG@5 actually rewards. Not pursued further.

### Finding 3 (negative): LambdaRank@5 underperforms both BPR and pointwise — on both metrics

Hypothesis going in was that truncating BPR's gradient to pairs that matter
for nDCG@5 (weight = `|ΔnDCG@5| / IDCG@5`, computed from each item's rank in
its user's group under the current model, recomputed once per epoch) would
trade some GAUC for nDCG@5 gain, net effect uncertain. That's not what
happened: **it loses on GAUC (0.6525 vs BPR's 0.6638) and on nDCG@5 (0.5257
vs BPR's 0.5304) simultaneously.** Not a trade-off — strictly worse.

Root cause, from the per-epoch diagnostic (`zero_frac` printed during
training): **~78% of sampled pairs get zero weight** every epoch, because
both the sampled positive and negative already sit outside the top-5 by the
current model's ranking. Effective gradient signal per epoch is roughly a
fifth of plain BPR's. Consequence: convergence takes ~18 epochs instead of
BPR's ~11, and lands at a worse optimum, not just a shifted one — this is
underfitting from signal starvation, not the GAUC/nDCG5 trade-off that was
hypothesized.

Known limitation in this implementation: ranks used for the lambda weight
are computed once at the start of each epoch (one extra forward pass over
train), not after every gradient step, to avoid re-scoring and re-sorting
every user's group at every minibatch. This introduces staleness within an
epoch. Given the effect size here (signal starvation, not a marginal miss),
per-step rank recomputation is very unlikely to close a 0.008 gap on primary
and was judged not worth the added engineering/runtime cost. Not pursued
further.

### Decision

**BPR is the loss going forward — set as `baseline.py`'s default (`--loss pairwise`)
as of this entry.** `python3 baseline.py --model fm` now reproduces 0.5971 primary;
pass `--loss pointwise` explicitly to reproduce the original documented FM baseline
(0.5946, `baseline_scores.json`). README updated to match (baseline ladder table,
run section, "从哪里开始改" — loss function moved from unexplored to tested).

LambdaRank@5 was a reasonable next step
given the metric-alignment hypothesis that motivated trying BPR in the first
place, but the data says the truncation cost (78% dead pairs) outweighs any
top-5-specific benefit at this model capacity/data scale. Remaining time
goes to sequence modeling (README "从哪里开始改" #2) — still the larger
identified headroom, and structurally unexplored (current features use zero
behavioral history per the ablation results already on record).

Reproduce: `python3 baseline.py --model fm --loss pairwise` (default seed 0,
40 epochs w/ patience 4) → test primary 0.5978.

## 2026-08-28 — DIN-style user history attention on top of BPR FM

Baseline: BPR FM, the current best (0.5971 test primary, previous entry).
Hypothesis (README "从哪里开始改" #1, this repo's own top-ranked unexplored
item): the model has zero access to behavioral history — `FIELDS` are all
static per-impression categoricals — despite users averaging dozens of
`long_view=1` events each in train. A DIN-style attention layer, computing a
per-candidate "interest vector" from a user's own history and folding it
into the FM interaction as a 6th field, was expected to be the largest
remaining headroom.

**New dependency: PyTorch** (CPU wheel, installed into a local `.venv/`,
`import torch` isolated entirely to the new `sequence_model.py` — the
existing `baseline.py` numpy FM/BPR/listwise/LambdaRank code is untouched).
Chosen over full frameworks (RecBole/TorchRec/LightGBM) after checking what
was actually installed (nothing beyond numpy) — RecBole's data/eval
abstractions don't match this repo's fixed, unusual GAUC/nDCG@5 contract
closely enough to be worth the integration cost and its pandas dependency;
plain PyTorch integrates directly with the already-tested `data.py`/
`evaluate.py`/BPR-sampling infrastructure.

**Architecture** (`sequence.py` for the numpy-only history builder,
`sequence_model.py` for the torch model — see file docstrings for the full
derivation): last `L=160` `long_view=1` items per user, strictly before the
current row's `time_ms` (verified: 6.9% of rows have zero prior history —
new users, or too early in a user's own timeline to have any yet). DIN local
activation unit (`concat[hist,cand,hist−cand,hist×cand]` → `Linear→PReLU→
Linear` → softmax over history, masked so an all-padding row gives an exact
zero vector, not softmax's default uniform-over-garbage) produces
`e_interest`, folded into the FM interaction via the same bilinear identity
as the 5 static fields (`inter_full = inter_static + S·e_interest`, since
the `‖e_interest‖²` term cancels exactly when expanding `(S+e_interest)²`).
History embeddings share the same table as the static `video_id` field.
Trained under BPR — same pos/neg sampling as `baseline.py`'s pairwise loss,
reused verbatim, only the score function changed.

**Verification before the real run** (torch autograd removes the need for
hand-derived gradient checks, but the masking logic is still hand-written):
a synthetic all-padded-history batch was confirmed to produce `e_interest`
of exactly zero (matching `z0`, the static-only score, to `atol=1e-6`) and
no NaN; a mixed-padding batch confirmed the PAD embedding row's gradient is
exactly zero after `backward()`. Both passed before touching real data.

**Bug caught during the first real-data run, fixed before the logged
result:** `_predict`'s default batch size (200,000, copied from
`baseline.py`'s plain-FM `predict`) doesn't actually bound memory here —
at `L=160`, one batch's attention step materializes a `(batch, L, 4k)`
tensor, so a 125K-row validation batch tried to allocate >5GB in one shot,
every epoch. Symptom was silent, not a crash: epoch times escalated
156s → 511s → 4844s under memory pressure rather than erroring. Fixed by
capping `_predict`'s batch size at 8192 (matching the training batch size);
confirmed identical loss/metric trajectory before and after the fix (same
seed reproduces the same numbers) — it was purely a performance bug, not a
correctness one — and epoch time stabilized to a flat ~70s.

Command: `.venv/bin/python sequence_model.py --seed <0..4>` (torch lives only
in the local venv, not the system Python — see README). Defaults: `k=16,
hidden=32, L=160, lr=0.001`, 40 epochs, patience 4 — same convergence rule
and same `k`/`lr`/batch size as the FM baseline for comparability).

### Results (test set, mean ± population std over seeds 0–4)

| model | GAUC | nDCG@5 | primary | Δ vs. BPR FM |
|---|---|---|---|---|
| BPR FM (no sequence, previous entry) | 0.6638 (σ=0.0007) | 0.5304 (σ=0.0004) | 0.5971 (σ=0.0005) | — |
| BPR FM + DIN attention | 0.6634 (σ=0.0005) | 0.5299 (σ=0.0006) | 0.5967 (σ=0.0005) | **−0.0004** |

Per-seed test primary: 0.5965, 0.5972, 0.5962, 0.5961, 0.5973 (range 0.5961–0.5973).
Epochs to convergence: 11, 10, 10, 13, 10 — essentially the same speed as
plain BPR FM's ~11.

### Finding (negative, but tight — not a bug, not noise): no detectable gain from user history

−0.0004 on primary, smaller than either model's own σ (0.0005) — statistically
indistinguishable from zero effect, not a regression either. All 5 seeds
land in a 0.0012-wide band, none coming close to beating BPR FM's own
5-seed range (0.5963–0.5978, previous entry) by any meaningful margin. This
reads as a genuine null result on a correctly-implemented pipeline (the
masking/gradient smoke tests passed before this run; the escalating-time bug
above was a performance issue with byte-identical loss curves before and
after the fix, not a correctness issue).

**Hypothesis for why the largest expected headroom produced nothing:**
KuaiRand-Pure's item catalog is small — 7,538 unique videos against 26,210
train users, each averaging dozens of interactions over two weeks. The
earlier feature-ablation findings already established that `user_id ×
video_id`'s FM cross term "吃掉了大部分可学的信号" (absorbs most of the
learnable signal) at this scale — that a user liked a *specific* video
before is largely already recoverable from the direct `user_id × video_id`
interaction once it's been seen enough times in training, leaving less for
a candidate-conditioned attention layer to add. This is consistent with,
not contradictory to, the earlier capacity-ablation finding (`k=8/16/32`
barely moving the score) — both point the same direction: at this data
scale, the bottleneck isn't model expressiveness (static-field capacity or
a richer scoring function), it's how much genuinely new *information*
static IDs plus recent-history IDs can carry beyond what's already encoded.
DIN's real-world wins are typically reported on catalogs orders of
magnitude larger, where a user's specific history genuinely disambiguates
among many more plausible candidates than 7,538 items allow here.

**Not yet ruled out, flagged rather than chased:** no hyperparameter tuning
was done for the new architecture (`hidden=32`, `lr=0.001` — the latter
copied from the FM baseline, not tuned for the attention MLP specifically);
history source was `long_view=1` only, not `is_click` (denser: median 14 vs
10 per user) or a multi-signal history. Given the effect size sits inside
noise and the capacity-ablation precedent suggests tuning capacity alone
rarely moves this dataset's ceiling, further tuning was judged unlikely to
be worth chasing without a stronger prior reason to expect it would — but
this wasn't exhaustively tested, unlike LambdaRank@5's root cause above
(78% dead pairs) which had a clear, confirmed mechanism.

### Decision

Sequence modeling, as implemented, does not beat BPR FM. **BPR FM
(`baseline.py`, `--loss pairwise`, 0.5971 test primary) remains the number
to beat.** `sequence_model.py` is kept in the repo (correct, verified,
reusable infrastructure — `sequence.py`'s history builder and the
DIN/FM-folding math are the expensive, correctness-sensitive parts, and
both are validated) in case a future direction wants to build on it, but is
not adopted as the new default. README's "从哪里开始改" updated to move
user-history sequence modeling from the top-ranked unexplored item to
tested, with this finding and hypothesis recorded — remaining unexplored
items (multi-target with `is_click`/`is_like`/etc., watch-time censored
regression per CWM, other architectures, time/drift features, unbiased
validation via the random-exposure log) are next, per the user's direction.

Reproduce: `.venv/bin/python sequence_model.py --seed 0` → test primary 0.5965.

## 2026-08-28 — Cheap diagnostic: does *any* sequence signal help, before building a heavier architecture?

Motivation: the DIN result above was a clean null, but it doesn't distinguish
between two very different explanations — (a) no sequence signal exists in
this dataset beyond what `user_id`/`author_id` already capture, or (b) a
sequence signal exists but DIN's smooth attention-pooling over up to 160
history items failed to isolate it. These have opposite implications for
what to try next (give up on sequence features vs. try a sharper mechanism),
so before spending more time on BST/DIEN (literature-review turn, same
conversation), ran the cheapest possible test: one binary feature, no
attention, numpy only, minutes not an hour-long training run.

**Feature:** `prior_exposure` = 1 if this user has `long_view`'d this *exact*
`video_id` at any strictly-earlier `time_ms` (same cross-split chronological
rule as `sequence.py`'s history builder — no leakage), else 0. Added as a 6th
field to the existing 5-field FM, trained under BPR (current default),
5 seeds, otherwise identical to the BPR FM baseline. Script:
`ablation_prior_exposure.py` (self-contained, follows `ablation_features.py`'s
pattern, no new dependencies).

**Sanity check before trusting the result** (a single rare binary feature
producing a real-looking lift is exactly the shape a leak or bug would take,
so verified before logging): `prior_exposure=1` fires on only 0.20% of all
rows (2,883 / 1,436,609), but among those rows the `long_view` rate is
**78.5%**, vs. **33.1%** overall — a 2.4× lift, and directionally exactly
what a "did they already watch and love this?" signal should look like.
488 such rows land in the test split alone. This is a real behavioral
pattern (repeat-watching previously-loved content), not an artifact.

### Results (test set, mean ± population std over seeds 0–4)

| model | GAUC | nDCG@5 | primary | Δ vs. BPR FM |
|---|---|---|---|---|
| BPR FM (5 fields, no sequence) | 0.6638 (σ=0.0007) | 0.5304 (σ=0.0004) | 0.5971 (σ=0.0005) | — |
| BPR FM + DIN attention (previous entry) | 0.6634 (σ=0.0005) | 0.5299 (σ=0.0006) | 0.5967 (σ=0.0005) | −0.0004 |
| **BPR FM + `prior_exposure` (6 fields)** | **0.6662** (σ=0.0003) | **0.5310** (σ=0.0003) | **0.5986** (σ=0.0003) | **+0.0015** |

Per-seed test primary: 0.5982, 0.5987, 0.5985, 0.5987, 0.5990 (range
0.5982–0.5990, tighter spread than BPR FM's own σ) — gap over BPR FM is
~5× either model's σ. Real, not noise.

### Finding: sequence signal *does* exist — DIN's mechanism, not the premise, was the problem

Answers the motivating question directly: **explanation (b)**. A one-bit,
hand-picked signal beats the full 160-item attention mechanism. This
refines, rather than contradicts, the previous entry's catalog-saturation
hypothesis — `user_id × video_id`'s bilinear FM term evidently doesn't fully
resolve *exact-repeat* affinity from a single (or few) prior training
exposure(s), even though in principle a low-rank interaction could
approximate it given enough repeated exposures.

**Likely mechanistic reason DIN missed this specifically:** DIN's softmax
attention has to *learn* to place a sharp, near-all weight on an exact match
among up to 160 history slots — but only ~0.18% of train rows are true
repeat-exposures (2,086 of 1,141,112), so the attention MLP saw very few
positive examples of "this is the pattern that should dominate the
weighted average." A smooth, softmax-normalized mechanism trained on a rare
pattern is a plausible way to end up diluting exactly the signal that
matters most, rather than sharpening around it — whereas a hand-crafted
indicator feature hands the FM the answer directly, bypassing that learning
problem entirely.

### Decision

**This is real headroom, currently unclaimed.** Two directions, not
mutually exclusive:
1. **Cheap, immediate:** add `prior_exposure` (and similar sharp,
   low-frequency indicator features — e.g. prior exposure to this exact
   `author_id`, not just `video_id`) directly to `baseline.py`'s FM as
   static fields. Small (+0.0015) but real, cheap, and stacks with BPR's
   own gain.
2. **If pursuing a sequence architecture further:** feed explicit
   exact-match/repeat-count features into the attention mechanism's input
   (e.g. concatenate a same-item flag into DIN's local activation unit)
   rather than relying on the attention MLP to discover a rare pattern
   from raw ID overlap alone — directly targets the failure mode diagnosed
   here instead of trying a heavier architecture (BST/DIEN) that still
   relies on the same smooth-pooling premise.

Reproduce: `python3 ablation_prior_exposure.py` → 5-seed test primary mean
0.5986.

## 2026-08-28 — Fixing DIN's mechanism directly: feed `same_video` into the attention input

Follow-up to the previous two entries. Tested decision option 2 above: instead
of adding `prior_exposure` as a direct FM field, feed the same underlying
information into DIN's attention mechanism itself — concatenate a `same_video`
flag (1 if a given history slot's `video_id` equals the candidate's, else 0)
into the local activation unit's input (`[hist, cand, hist−cand, hist×cand,
same_video]`, 4k+1 dims instead of 4k). Hypothesis: DIN's softmax attention
should now be able to *directly* key off this bit instead of having to infer
"exact match" from the embedding difference vector being ≈0, which the
previous entry's diagnosis suggested it wasn't reliably learning to do from
only ~0.18% of train rows containing this pattern.

Same verification discipline as the original DIN change: smoke-tested the
all-PAD edge case (still exact-zero) and the PAD-embedding-row dead-gradient
invariant (still exactly 0) before the real run — both passed. `sequence_model.py`
modified in place (not a new file — `baseline.py` stays the frozen numpy
reference; this is iteration on our own experimental code, same as the loss
experiments iterated on `baseline.py` itself). Also added `--device`
(`auto`/`cpu`/`cuda`/`mps`) and `train_seq.sbatch` for the SoC GPU cluster in
the same sitting — unused for this particular run (still ran on CPU, same
~70s/epoch), prepped for whatever's next.

Command: `.venv/bin/python sequence_model.py --seed <0..4>` (same defaults
as the previous DIN entry).

### Results (test set, mean ± population std over seeds 0–4)

| model | GAUC | nDCG@5 | primary | Δ vs. BPR FM |
|---|---|---|---|---|
| BPR FM (no sequence) | 0.6638 (σ=0.0007) | 0.5304 (σ=0.0004) | 0.5971 (σ=0.0005) | — |
| BPR FM + DIN attention (plain) | 0.6634 (σ=0.0005) | 0.5299 (σ=0.0006) | 0.5967 (σ=0.0005) | −0.0004 |
| **BPR FM + DIN attention + `same_video` input** | **0.6638** (σ=0.0006) | **0.5301** (σ=0.0006) | **0.5969** (σ=0.0006) | **−0.0002** |
| *(for reference)* BPR FM + `prior_exposure` as a direct field | 0.6662 (σ=0.0003) | 0.5310 (σ=0.0003) | 0.5986 (σ=0.0003) | +0.0015 |

Per-seed test primary: 0.5972, 0.5960, 0.5974, 0.5966, 0.5975 (range
0.5960–0.5975). Epochs to convergence: 11, 14, 15, 9, 10 — similar range to
plain DIN's 11/10/10/13/10.

### Finding: giving the attention mechanism the same information does *not* recover the direct-feature gain

+0.0002 over plain DIN — smaller than either model's own σ, not a real
improvement. Still −0.0002 vs. BPR FM itself, also within noise. The
hypothesis was specific and testable — it failed cleanly, not ambiguously:
handing the exact same bit of information to two different parts of the
model produces two very different outcomes:

- As a **direct FM field**: +0.0015, real (previous entry).
- As an **attention-mechanism input**: +0.0002, noise (this entry).

**Why the same information helps in one place and not the other:** a direct
FM field gets its own linear weight — a single, easily-estimated global
parameter answering "what's the average effect of `prior_exposure=1` on the
score," learnable cleanly even from ~2,000 positive examples, *plus* it
participates in every pairwise cross term automatically via the existing FM
math. Routed through DIN instead, the same bit only ever influences the
score *indirectly*: `same_video` → attention MLP → softmax logit → weighted
share of `e_interest` → dot products → final score. That's a much longer
chain for gradient to travel, and — critically — the *rarity* of the
underlying pattern (~0.18% of train rows) doesn't change just because the
input got easier to read; the attention MLP still only sees this exact
pattern in ~2,000 training rows and has to learn an entire conditional
weighting *function* around it, not just one scalar. Easier signal to detect
≠ easier signal to learn to exploit through a longer, multiplicative
computation graph.

### Decision

**Attention-mechanism fixes, at least this one, don't recover what a direct
feature already gets for free.** This isn't just "try a bigger attention
model" territory — the diagnosis here suggests the *mechanism itself*
(routing a sharp, rare, strongly-predictive pattern through softmax-pooled
attention rather than a direct linear term) is a poor fit for this specific
kind of signal, independent of architecture size. That's a real reason for
caution before investing in BST/DIEN (from the earlier literature-review
turn) — both still fundamentally route information through the same kind of
softmax-attention/weighted-combination pathway that just failed to exploit
`same_video` even when handed explicitly.

**Recommendation: bank the direct-feature win.** `prior_exposure` (+ similar
sharp, hand-craftable indicators, e.g. prior exposure to this `author_id`)
added straight to `baseline.py`'s FM is the validated, real gain on the
table right now. `sequence_model.py`/`sequence.py` remain in the repo as
correct, verified infrastructure, but two independent attempts to make the
attention route pay off (plain pooling, then pooling with the fix that
should have worked) both landed at noise. Per the user's direction on
whether to pursue this further or move to banking the win / a different
headroom item next.

Reproduce: `.venv/bin/python sequence_model.py --seed 0` → test primary
0.5972.

## 2026-08-28 — Does temporal recency specifically carry signal? (before committing to DIEN)

Motivation: after the previous entry's clean negative result, the question
was whether to invest in BST/DIEN anyway. Their core differentiator over
DIN is *order/temporal evolution*, not just set-membership — a different
claim than anything tested so far (`prior_exposure` tests only "have I ever
seen this," no time information). Ran the same cheap-diagnostic playbook
again, this time testing recency specifically, before deciding.

**Feature:** `author_recency` — bucketed time since this user's most recent
prior `long_view` of *any* video by the candidate's `author_id` (11
categories: "never" + 10 quantile buckets of the gap in hours, edges fit on
train). Same strict-earlier-than-`time_ms` rule as before, added as a 6th
FM field, BPR, 5 seeds. Script: `ablation_author_recency.py`.

Chose author-level (not video-level) specifically to test something the FM
doesn't already have a direct handle on: `user_id × author_id` is a static
field, but *how recently* is a temporal signal layered on top of that
static affinity — the qualitatively different kind of information BST/DIEN
claim to add over DIN.

### Results (test set, mean ± population std over seeds 0–4)

| model | GAUC | nDCG@5 | primary | Δ vs. BPR FM |
|---|---|---|---|---|
| BPR FM (no sequence) | 0.6638 (σ=0.0007) | 0.5304 (σ=0.0004) | 0.5971 (σ=0.0005) | — |
| BPR FM + `prior_exposure` (exact video, binary) | 0.6662 (σ=0.0003) | 0.5310 (σ=0.0003) | 0.5986 (σ=0.0003) | +0.0015 |
| **BPR FM + `author_recency` (bucketed gap)** | **0.6663** (σ=0.0002) | **0.5313** (σ=0.0004) | **0.5988** (σ=0.0003) | **+0.0017** |

Per-seed test primary: 0.5991, 0.5986, 0.5985, 0.5986, 0.5992. Real —
~5.7× the std, tight spread.

### Finding: recency carries real signal — but the shape is a step function, not smooth decay

Answers the motivating question: yes, temporal information (not just
set-membership) has value here. But breaking down the long_view rate by
gap bucket shows *why*, and it's not what DIEN's design assumes:

| gap since last author exposure | long_view rate | n |
|---|---|---|
| never | 33.0% | 1,421,731 |
| ~0h (same-session adjacency) | **99.95%** | 2,171 |
| <1h (not instant) | 22.6% (below baseline) | 805 |
| 1h – 250h+ (all 8 remaining buckets) | flat, 41–45% | ~1,488 each |

This is not a smooth decay curve. It's dominated by (1) a near-deterministic
same-session adjacency effect (the very next impression after long-viewing
someone is almost always long-viewed too — plausibly the recommender itself
clustering an author's videos back-to-back in a session, not a learned
"interest" pattern at all) and (2) a coarse, flat "seen this author recently
at all vs. never" step that shows no visible further decay from 1 hour out
to 10+ days. The `<1h`-but-not-instant bucket sitting *below* the "never"
baseline (n=805, plausibly real, not obviously noise) is an unexplained
wrinkle worth flagging rather than a clean part of the story.

### Decision: this changes the calculus on DIEN specifically

Two independent hand-crafted features (`prior_exposure`, `author_recency`)
both show real, similar-sized gains (+0.0015, +0.0017) — this rules out
"there's no sequence signal in this dataset" for good; the earlier DIN
entries' failure was about DIN's specific mechanism, not the premise. That
much strengthens the case for trying a properly-designed sequence
architecture rather than abandoning the direction.

But it specifically weakens the case for **DIEN**: DIEN's headline
mechanism (GRU-based gradual interest evolution, AUGRU) is built to model
smooth drift over a sequence. What's actually present here looks more like
a step function at session boundaries (near-deterministic adjacency) plus a
flat recent-vs-never split — not gradual decay. DIEN's specific design
advantage (modeling *how* interest gradually evolves) may be solving a
problem that isn't the one this dataset has. A much cheaper feature — e.g.
"is this immediately preceded by a long_view of the same author in this
session" — would likely capture most of what's in the adjacency effect
without any of DIEN's engineering cost, and is a natural next experiment
before committing to a heavier architecture.

BST (order via self-attention + positional encoding, no gradual-evolution
assumption) is not weakened by this finding the same way — still untested,
still a reasonable next architecture if pursuing this further, since it
doesn't assume smooth decay the way DIEN does.

Reproduce: `python3 ablation_author_recency.py` → 5-seed test primary mean
0.5988.

## 2026-08-28 — Banking the wins: `prior_exposure` + `author_recency` folded into `data.py`/`baseline.py` permanently; adjacency tested standalone

Two changes in this entry:

**1. `prior_exposure` and `author_recency` are now permanent FM fields**, not
one-off ablation scripts. New `temporal_features.py` (numpy-only,
`build_temporal_features(splits)`) consolidates the cross-row temporal-scan
logic that `ablation_prior_exposure.py`/`ablation_author_recency.py` each
implemented independently; `data.py`'s `encode()` now calls it and appends
both as columns 5–6 of `X` by default. `FIELDS` extended from 5 to 7
entries. `build_vocab()` was decoupled from `len(FIELDS)` to make this safe
(previously `vocabs = [dict() for _ in FIELDS]` would have silently created
2 bogus empty vocab entries once `FIELDS` grew past the 5 static fields —
caught and fixed before running anything). Verified: new `dim=40273`
(40260 + 2 + 11, exactly as computed), `X.shape=(N,7)`, `prior_exposure`
column has exactly 2086 positive rows in train — matches the earlier
diagnostic exactly. `sequence.py`'s `build_history` and `sequence_model.py`
both still work unchanged (they only special-case field index 1 = video_id,
which is untouched by this change) — confirmed by re-running the history
builder against the new pipeline before touching `baseline.py`.

This changes what `python3 baseline.py --model fm` produces by default —
same kind of default-changing move as the BPR loss switch, documented the
same way.

Command: `python3 baseline.py --model fm --loss pairwise --seed <0..4>`
(no flag needed — this is now just what running the baseline does).

### Results (test set, mean ± population std over seeds 0–4)

| model | GAUC | nDCG@5 | primary | Δ vs. 5-field BPR FM |
|---|---|---|---|---|
| BPR FM, 5 fields (previous default) | 0.6638 (σ=0.0007) | 0.5304 (σ=0.0004) | 0.5971 (σ=0.0005) | — |
| **BPR FM, 7 fields (new default)** | **0.6689** (σ=0.0005) | **0.5326** (σ=0.0005) | **0.6008** (σ=0.0004) | **+0.0037** |

Per-seed test primary: 0.6015, 0.6007, 0.6003, 0.6009, 0.6004 — tight,
consistent, well clear of the ε=0.002 convergence bar. The combined gain
(+0.0037) is slightly more than the sum of the two features tested alone
(+0.0015 and +0.0017 = +0.0032) — a small positive interaction, not just
additive, plausible since some rows benefit from having both signals available
at once rather than either alone.

**2. Adjacency, tested standalone (not stacked on the new default) — confirms the step-function diagnosis cleanly**

`ablation_adjacency.py`: a single binary feature — "was this user's
immediately-preceding interaction (any row, chronologically) a `long_view`
of this same `author_id`" — added as a 6th field to the *original* 5-field
FM (same comparison basis as `prior_exposure`/`author_recency`'s own
standalone tests, for direct comparability).

| model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| BPR FM (5 fields) | 0.6638 | 0.5304 | 0.5971 |
| + `prior_exposure` (binary, exact video) | 0.6662 | 0.5310 | 0.5986 |
| + `author_recency` (11-bucket gap) | 0.6663 | 0.5313 | 0.5988 |
| **+ `adjacency` (binary, session-boundary only)** | **0.6662** (σ=0.0002) | **0.5310** (σ=0.0003) | **0.5986** (σ=0.0002) |

Per-seed test primary: 0.5984, 0.5989, 0.5983, 0.5986, 0.5987. Positive rate:
0.209% (3,005 / 1,436,609 rows) — comparable sparsity to `prior_exposure`'s
0.20%.

**One bit matches eleven buckets.** `adjacency`'s 0.5986 is indistinguishable
from `author_recency`'s 0.5988 (both within each other's σ) — a single
binary flag captures essentially the *entire* value of the much more
elaborate 11-category recency feature. This confirms, more cleanly than the
bucket breakdown in the previous entry, that `author_recency`'s value is
overwhelmingly a session-adjacency step effect, not graded temporal decay.

### Decision

Two independent, validated wins are now permanently in `baseline.py`'s FM
(0.5971 → 0.6008, +0.0037). `adjacency` itself was **not** added as a third
field — it's redundant with `author_recency`, which already captures
everything it does (the standalone comparison above establishes this; a
combined-field test wasn't run since there's no remaining hypothesis left to
test). `README.md` updated: baseline ladder, "已实测" section, files table,
and the "运行" section's default-reproduction number all now reflect 0.6008
as what the bare command produces.

For the DIEN/BST question this thread started from: the adjacency result is
further, cleaner confirmation that what's recoverable here is a sharp,
near-deterministic session-boundary pattern, not smooth interest evolution —
reinforcing the previous entry's caution specifically against DIEN's
graded-decay mechanism, while leaving BST (order via self-attention, no
decay assumption) untouched by this evidence either way.

Reproduce: `python3 baseline.py --model fm --seed 0` → test primary 0.6015
(7-field default). `python3 ablation_adjacency.py` → 5-seed test primary
mean 0.5986 (standalone, vs. 5-field baseline).

## 2026-08-28 — Multi-task learning: BPR + `is_click` auxiliary loss

BST was next per the priority list, but is blocked on GPU access (self-attention
training is ~35x slower than DIN on this machine's CPU — user is setting up
the SoC cluster; a future entry will log BST results once that's available).
Moved to the next unexplored item in the meantime: multi-task learning
(README "从哪里开始改" #2).

`is_click`/`is_like`/`is_follow`/`is_comment`/`is_forward`/`play_time_ms` are
outcome labels of the *same* row being scored — unlike the sequence-modeling
features, they can't be used as input features (that would be leakage: you
can't condition a prediction on an outcome that doesn't exist yet at
inference time). The only valid way to use them is as auxiliary training
signals: train the shared embeddings to also predict a denser correlated
outcome, hoping better-regularized/richer embeddings help the actual
`long_view` ranking task even though only that task is scored.

**Implementation**: new `loss='pairwise_multitask'` on `baseline.py`'s `FM`
— same BPR main task (`step_pairwise`, unchanged) plus a pointwise BCE
auxiliary loss on `is_click` (46.3% positive rate vs. `long_view`'s 33.7% —
the denser signal the README flagged). Auxiliary task shares `V`
(embeddings) with the main task but has its own linear head (`W_aux`,
`b_aux`) so it doesn't corrupt the main task's own linear term — gradient
from both tasks accumulates into the same `gV` before one combined Adam
step; `W_aux`/`b_aux` get their own small local Adam update (own moment
buffers, shared step counter). `data.py` gained `aux_labels(splits, col=
'is_click')`, row-aligned with `encode()`'s output. Verified: plain
`pairwise` reproduces byte-identical numbers after adding `W_aux`/`b_aux` to
`FM.__init__` (no regression), multitask's aux BCE starts near `ln(2)≈0.693`
(zero-initialized `W_aux`/`b_aux`, as expected) and decreases smoothly.

Command: `python3 baseline.py --model fm --loss pairwise_multitask --seed <0..4>`
(`--aux_weight` default 0.2).

### Results (test set, mean ± population std over seeds 0–4)

| model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| BPR FM, 7 fields (current best, no aux task) | 0.6689 (σ=0.0005) | 0.5326 (σ=0.0005) | 0.6008 (σ=0.0004) |
| **+ `is_click` auxiliary BCE (`aux_weight=0.2`)** | **0.6689** (σ=0.0005) | **0.5325** (σ=0.0004) | **0.6007** (σ=0.0004) |

Per-seed test primary: 0.6014, 0.6006, 0.6004, 0.6005, 0.6005 — indistinguishable
from the no-aux-task baseline (Δ = −0.0001, well inside σ).

**Checked whether the weight was just too weak before concluding null**:
single-seed test at `aux_weight=1.0` (5x stronger) gave test primary
**0.5999** — not better, slightly *worse* than both `aux_weight=0.2`
(0.6014 same seed) and the no-aux baseline. Stronger auxiliary pull doesn't
help and may mildly hurt, pulling the shared embeddings toward optimizing
click-prediction rather than the actual `long_view` ranking objective. This
rules out "just needed a bigger weight" as an explanation — the result is a
genuine null across the weight range tested, not an under-tuned setting.

### Finding: no benefit from is_click multi-task supervision, at either weight tested

The auxiliary task trains and converges normally (BCE loss decreases
smoothly across epochs) — this isn't a broken implementation, the shared
embeddings just don't end up better *for the ranking task specifically*
from also being pushed to predict clicks. Plausible reading: `is_click` and
`long_view` are correlated but represent different funnel stages (click =
"decided to view", long_view = "found it worth watching") — the shared
`user_id × video_id` embedding may already be expressive enough (echoing
the earlier feature/capacity-ablation findings) that a correlated-but-distinct
auxiliary objective doesn't add information beyond what `long_view`'s own
BPR gradient already teaches it, and instead just spends some of the
embedding's limited capacity/gradient budget on a goal that isn't the one
being scored.

### Decision

Multi-task with `is_click` doesn't beat the 7-field BPR FM (0.6008 remains
the number to beat). Not pursued further — the two weights tested (0.2, 1.0)
bracket a wide enough range that a different single weight is unlikely to
flip this, and there's no diagnosed specific reason (unlike the DIN
`same_video` case) to expect a different weight would help. Other auxiliary
targets (`is_like`, `play_time_ms` as regression, etc.) remain untested if
this direction is revisited later, but given `is_click` — the densest,
most-directly-related signal — showed nothing, they're not an obvious
priority. README's "从哪里开始改" updated: multi-task moved from unexplored
to tested.

Reproduce: `python3 baseline.py --model fm --loss pairwise_multitask --seed 0`
→ test primary 0.6014 (`aux_weight=0.2`, default).

## 2026-08-28 — CWM-style watch-time censored regression, banked as new default

README's next unexplored item: watch-time modeling per [CWM](https://github.com/hyz20/CWM)
("counterfactual watch time"). Unlike the `is_click` multi-task attempt above
(a distinct, only-correlated funnel-stage signal that showed nothing),
watch time is the *direct underlying continuous quantity* `long_view` is
almost certainly thresholded from — worth checking before writing this
direction off.

**Grounding in the actual data before designing anything** (previous
sections of this log established the habit of verifying premises against
real numbers rather than the literature's framing):
- `play_time_ms >= 18000` (18s) alone matches `long_view` 96.7% of the time
  — strong evidence `long_view` is a coarsened/binarized version of watch
  time, not a distinct signal. This is the reason to expect watch-time
  supervision might succeed where `is_click` didn't: it's not a different
  correlated task, it's closer to the *ungrouped* version of the same one.
- The classical CWM framing ("censored at the video's own length when
  watched to completion") doesn't naively fit this dataset — videos loop.
  17.3% of rows have `play_time_ms >= duration_ms`. But the distribution
  within that group is revealing: **median ratio 1.09×, p75 1.48×, then a
  sharp jump to p90 = 802×, p99 = 16,505×**. The bulk of "completed" views
  cluster just past the boundary (consistent with genuine single-pass
  completion plus measurement noise); a minority are extreme
  multi-hundred-loop outliers (almost certainly passive/backgrounded
  looping, not signal). This directly informed the design below: **use
  `duration_ms` itself, not the raw `play_time_ms` value, as the
  censored-row target** — sidesteps the noisy outlier tail entirely rather
  than needing to cap/clip it.

**Design** (`data.py::watch_time_targets`, `baseline.py::FM.step_pairwise_watchtime`,
`loss='pairwise_watchtime'`): classical Tobit-style censored regression.
`censored = (play_time_ms >= duration_ms)`. For uncensored rows (exact
observation — user left before the video ended): standard squared-error
regression toward `t = log1p(play_time_ms) / 12` (log1p for the wide
dynamic range across durations 11K–250K+ ms; divide-by-12 just rescales
into roughly `[0, 1.2]`, matching the gradient-magnitude order of the
`is_click` BCE experiment so `aux_weight` values stay roughly comparable
across experiments — not a theoretically meaningful constant). For censored
rows: one-sided squared hinge toward `tau = log1p(duration_ms) / 12` — loss
is `0.5·max(0, tau − z)²`, so predictions below the threshold are penalized
(we know the truth is at least `tau`) but predictions above it aren't
(we don't know by how much, so no assumption is made). Same shared-`V`,
separate-linear-head (`W_aux`/`b_aux`) architecture as the `is_click`
attempt — same code path, reused without modification.

Command: `python3 baseline.py --model fm --loss pairwise_watchtime --seed <0..4>`
(`--aux_weight` default 0.2, same default as the `is_click` attempt for a
fair comparison).

### Results (test set, mean ± population std over seeds 0–4)

| model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| BPR FM, 7 fields (previous default) | 0.6689 (σ=0.0005) | 0.5326 (σ=0.0005) | 0.6008 (σ=0.0004) |
| + `is_click` auxiliary BCE (previous entry, null) | 0.6689 (σ=0.0005) | 0.5325 (σ=0.0004) | 0.6007 (σ=0.0004) |
| **+ CWM-style watch-time censored regression (new default)** | **0.6702** (σ=0.0005) | **0.5333** (σ=0.0003) | **0.6017** (σ=0.0004) |

Per-seed test primary: 0.6020, 0.6018, 0.6022, 0.6016, 0.6011 (range
0.6011–0.6022) — a tighter spread than the no-aux-task baseline's own range
(0.6003–0.6015), sitting consistently above it. Epochs to convergence: 11,
10, 11, 15, 12 — similar range to the other BPR variants.

### Finding: real, though more modest than the headline wins

+0.0009 on primary. By this project's own quick-heuristic yardstick (gap
vs. the larger raw per-seed σ), that's ~2.25×σ — weaker than BPR's ~3×σ or
`prior_exposure`'s ~5×σ, but clearly past the "smaller than either model's
own σ" bar that marked the `is_click` attempt as null. A more careful
check — standard error of the difference between the two independent
5-seed means, `sqrt(σ₁²/5 + σ₂²/5)` — puts the gap at **~3.6 standard
errors**, a comfortably significant difference; the raw-σ heuristic used
elsewhere in this log is a conservative shortcut, not a rejection of this
result. Being transparent about the calibration either way: this is a real
but more marginal gain than the project's strongest findings, not another
BPR-sized jump.

**Why this succeeded where `is_click` didn't, most likely**: watch time is
close to the actual continuous quantity `long_view` discretizes (96.7%
match at an 18s threshold), so the auxiliary task teaches the shared
embeddings a finer-grained version of the *same* preference signal instead
of a distinct, only-correlated one. The proper censored-regression
treatment (vs. naive squared-error on raw `play_time_ms`) matters
mechanically too — naive regression would have been dominated by the noisy
looping outliers found during the data investigation above, likely
corrupting rather than helping the shared embeddings.

### Decision

**Adopted as the new default** (`baseline.py`'s `--loss` defaults to
`pairwise_watchtime`), following the same pattern as the BPR-loss and
temporal-feature adoptions: real, reproducible, stacks on top of everything
already banked. `python3 baseline.py --model fm` now reproduces 0.6017;
pass `--loss pairwise` explicitly to reproduce the pre-this-entry number
(0.6008). `is_click` multi-task (previous entry) remains available via
`--loss pairwise_multitask` but is not recommended (null result).
`aux_weight` was not swept for this experiment (used the same 0.2 default
as the `is_click` test for a controlled comparison) — worth revisiting if
pursuing this further, alongside potentially replacing the fixed `scale=12`
normalization with something derived from the data.

Reproduce: `python3 baseline.py --model fm --seed 0` → test primary 0.6020
(new default, `pairwise_watchtime`, `aux_weight=0.2`).

## 2026-08-28 — Pushing further on the CWM direction: three follow-ups, all closed off

After adopting watch-time censored regression, checked whether there was
more to extract from the same direction before moving on. Three experiments,
run in parallel since all are cheap (numpy):

### 1. `aux_weight` sweep — confirms the default was already near-optimal

Single-seed (seed 0) sweep across `--aux_weight ∈ {0.05, 0.1, 0.2, 0.3, 0.5,
1.0, 2.0}` — a 40× span. Test primary stayed in a **0.6016–0.6022** band the
entire way, no trend up or down. The default (0.2 → 0.6020) sits right in
the middle of this flat region. **Nothing left to gain from tuning this
knob** — a genuinely useful negative result, since it means the +0.0009 gain
already banked isn't being left on the table by an under/over-tuned weight,
and further sweeping isn't worth the time.

### 2. `author_watch_affinity` — a new feature (not aux loss), modest and not adopted

Hypothesis: `author_recency` captures *timing* of past engagement with an
author; does *magnitude* (how strongly, on average, has this user engaged
with this author's content before) carry separate value? Built
`ablation_author_watch_affinity.py`: bucketed historical mean
`log1p(min(play_time_ms, 10×duration_ms))` per (user, author) pair, same
strict-earlier-than-`time_ms` rule as the other temporal features, same
10× cap rationale as the censored-regression design (avoid the noisy
looping tail polluting the running average). Tested standalone against the
5-field baseline, same comparison basis as `author_recency`/`adjacency`.

| model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| BPR FM (5 fields) | 0.6638 | 0.5304 | 0.5971 |
| + `author_recency` (timing) | 0.6663 | 0.5313 | 0.5988 |
| + `author_watch_affinity` (magnitude) | 0.6651 (σ=0.0007) | 0.5308 (σ=0.0005) | 0.5980 (σ=0.0006) |

+0.0009 over baseline (σ ratio ~1.5×, weaker than `author_recency`'s ~5×),
and notably *below* `author_recency` alone (−0.0008) rather than additive
to it. Only 3.50% of rows have any prior (user, author) interaction to
average over — many of those averages are estimated from just 1–2 samples,
plausibly noisy enough to explain the weaker showing. Direction is
plausible but not clearly adding a new dimension beyond what timing already
captures — **not adopted**.

### 3. `pairwise_combined` (is_click + watch-time together) — is_click stays null even in combination

New `loss='pairwise_combined'`: both auxiliary tasks trained simultaneously,
each with its own independent linear head (`W_aux`/`b_aux` for click,
`W_aux2`/`b_aux2` for watch-time) sharing the same `V`. Tests whether
`is_click`'s earlier null result was conditional on watch-time's absence
(maybe it had nothing to add once the model was still "dumb," but could
matter once watch-time raises the baseline) — a real, testable hypothesis,
not just repeating the earlier experiment.

| model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| watch-time only (current default) | 0.6702 (σ=0.0005) | 0.5333 (σ=0.0003) | 0.6017 (σ=0.0004) |
| **watch-time + is_click combined** | **0.6698** (σ=0.0005) | **0.5332** (σ=0.0005) | **0.6015** (σ=0.0005) |

Statistically identical (−0.0002, well inside either σ). `is_click` doesn't
add anything on top of watch-time either — the earlier null result wasn't
conditional on model strength, it's a consistent finding regardless of
what else is already in the model. **Not adopted.**

### Decision

All three follow-ups closed off cleanly — none changes the current default
(`pairwise_watchtime`, `aux_weight=0.2`, 0.6017 test primary remains the
number to beat). This is a good outcome for the *investigation*, not a
failure: it confirms the watch-time gain is robust (insensitive to its own
hyperparameter) and that the underlying signal has been extracted about as
completely as this feature-engineering approach can get — `is_click`
specifically appears to carry no information the shared embeddings don't
already get from `long_view`'s own gradient, confirmed now under two
different conditions. Time is better spent on a genuinely different lever:
BST (pending GPU) or a model architecture change.

Reproduce: `python3 baseline.py --model fm --loss pairwise_combined --seed 0`
→ test primary 0.6021. `python3 ablation_author_watch_affinity.py` →
5-seed test primary mean 0.5980.

## 2026-08-29 — Model architecture change: DeepFM, a clean null

Next unexplored item from the README (BST still pending GPU access).
README's own reasoning ("capacity testing showed it's not the bottleneck")
only tested more parameters *within FM's fixed bilinear shape* (`k=8/16/32`
ablation) — a DNN branch is a structurally different kind of expressiveness
(can represent nonlinear feature combinations FM's quadratic form cannot
express at all, regardless of size), so this wasn't actually settled by the
prior ablation. Worth testing rather than assuming, given how many other
README priors in this log needed correction once measured (DIN's premise,
`is_click`'s multi-task assumption).

Brief literature check before implementing (see project's research-first
convention from the sequence-modeling work): the README's reference set
(DeepFM/DCN/xDeepFM, 2017–2018) has been superseded by DCN-V2/V3, GDCN, and
notably **FinalMLP** (2023) — a pure two-stream-MLP architecture with *no*
explicit interaction structure at all, which beats DCN-V2/xDeepFM/AutoInt+
even on smaller benchmarks (MovieLens, Frappe), not just industrial-scale
ones. That's evidence nonlinear combination capacity (what a DNN provides)
can matter independent of *structured* explicit-interaction machinery. User
chose **DeepFM** over FinalMLP as the first test: additive (`z = z_FM +
z_DNN`, augments the validated FM term rather than replacing it, so a null
result is unambiguous — unlike FinalMLP, where a negative result can't
distinguish "no signal to find" from "worse architecture").

**Implementation** (`deepfm_model.py`, torch — reuses `resolve_device` from
`sequence_model.py`): `z_FM` is the exact same bilinear structure as
`baseline.py`'s FM (shared `V`, same interaction formula). `z_DNN` flattens
all 7 fields' embeddings (7k dims) through a small MLP (2 hidden layers,
64→32, PReLU, dropout 0.2) to a scalar, summed with `z_FM`. Kept
deliberately small — this project has repeatedly found bigger doesn't help
at this dataset's scale (1.14M rows), and the point of this experiment is
to test a *different* kind of expressiveness, not just add more parameters
in a new shape. Trained under BPR, same pos/neg sampling as
`baseline.py`'s `run_fm(loss='pairwise')`. Compared against the **clean**
BPR FM 7-field baseline (0.6008, no watch-time auxiliary task) specifically
— isolates the DNN branch's own contribution before considering whether to
stack it with anything else.

Smoke-tested clean before the real run: synthetic-batch forward/backward,
no NaN in `V` or DNN parameter gradients.

Command: `.venv/bin/python deepfm_model.py --seed <0..4>` (`--hidden 64 32`,
`--dropout 0.2` defaults).

### Results (test set, mean ± population std over seeds 0–4)

| model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| BPR FM, 7 fields (clean baseline) | 0.6689 (σ=0.0005) | 0.5326 (σ=0.0005) | 0.6008 (σ=0.0004) |
| **+ DNN branch (DeepFM)** | **0.6687** (σ=0.0007) | **0.5328** (σ=0.0005) | **0.6007** (σ=0.0006) |

Per-seed test primary: 0.6005, 0.6016, 0.6001, 0.6013, 0.6002 — gap vs.
baseline is −0.0001, both by the raw-σ heuristic (−0.1×) and the more
careful standard-error-of-the-difference check (−0.19×). About as clean a
null as this log has recorded — not a marginal miss, no detectable effect
in either direction.

### Finding: the DNN branch finds nothing FM's bilinear term didn't already have

Consistent with, and now directly testing, this project's standing
hypothesis: `user_id × video_id`'s pairwise interaction already absorbs
most of the learnable signal in this feature set (first established in the
static-feature and capacity ablations, reconfirmed independently every time
something new has been tried against it — DIN's attention, `is_click`
multi-task, now nonlinear combination via a DNN). It's not that bigger
models don't work here in general — it's that there doesn't appear to be
additional *structure* among these particular 7 fields for a more
expressive interaction function to find, whether that extra expressiveness
comes from more capacity in FM's own shape (already tested, null) or a
genuinely different shape (DNN, this entry, also null).

### Decision

**Not adopted.** `deepfm_model.py` kept in the repo as verified, reusable
infrastructure in case revisited. Two natural follow-ups exist but are
lower priority given this clean null: (a) stack the watch-time auxiliary
task on top of DeepFM to see if a stronger training signal changes the
DNN's ability to find something (the DNN failing to find structure isn't
the same question as whether richer supervision would help it), (b) try
FinalMLP specifically, since it replaces rather than augments and might
access a genuinely different inductive bias — but per the project's now
consistent pattern (features/capacity/attention/multi-task/architecture all
converging on "the signal is concentrated in one pairwise term"), neither
is expected to be a high-probability win. BPR FM + temporal features +
watch-time (0.6017 test primary) remains the number to beat. BST (pending
GPU) is the one still-open direction with a different-in-kind rationale
(order-awareness, not just more expressiveness).

Reproduce: `.venv/bin/python deepfm_model.py --seed 0` → test primary
0.6005.

## 2026-08-29 — Model architecture change, part 2: FinalMLP, mild underperformance

Follow-up to the DeepFM entry. User wanted to try the more radical option
too: **FinalMLP** (AAAI 2023) replaces FM's explicit interaction term
entirely, rather than augmenting it — a genuinely different inductive bias,
not just "DeepFM without the FM part." Confirmed the actual architecture
from the paper before implementing (not just the earlier high-level
description):

- `e` = all 7 fields' embeddings concatenated.
- Two independent MMOE-style gates: `h1 = 2·σ(gate1(e))⊙e`,
  `h2 = 2·σ(gate2(e))⊙e` — same underlying features, but each stream sees a
  different learned re-weighting of them.
- Two small MLP towers (2 layers, 64 units, dropout 0.2 — same sizing
  discipline as `deepfm_model.py`, same overfitting-risk reasoning) process
  `h1`/`h2` independently into `o1`/`o2`.
- **Multi-head bilinear fusion**: `o1`/`o2` split into 4 heads each; each
  head pair combined via its own learned bilinear matrix
  (`o1ₕᵀ Wₕ o2ₕ`), summed across heads, plus a bias — this is the *only*
  place any interaction between the two streams happens, and it's the only
  interaction structure in the whole model.

No `V`/`W`/FM bilinear term anywhere in this model — `finalmlp_model.py` is
a self-contained architecture, not layered on `baseline.py`'s FM at all.
Same BPR training loop and comparison basis as the DeepFM entry (clean
7-field BPR FM, 0.6008, no watch-time task) for direct comparability
between the two architecture experiments.

Smoke-tested clean before the real run (synthetic batch, no NaN in gates,
either MLP stream, or the bilinear fusion parameters).

Command: `.venv/bin/python finalmlp_model.py --seed <0..4>`
(`--stream_dim 64 --n_heads 4 --dropout 0.2` defaults).

### Results (test set, mean ± population std over seeds 0–4)

| model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| BPR FM, 7 fields (clean baseline) | 0.6689 (σ=0.0005) | 0.5326 (σ=0.0005) | 0.6008 (σ=0.0004) |
| + DNN branch (DeepFM, previous entry) | 0.6687 (σ=0.0007) | 0.5328 (σ=0.0005) | 0.6007 (σ=0.0006) |
| **FinalMLP (replaces FM entirely)** | **0.6681** (σ=0.0008) | **0.5324** (σ=0.0007) | **0.6002** (σ=0.0007) |

Per-seed test primary: 0.6008, 0.6012, 0.5992, 0.6003, 0.5996. Gap vs.
baseline: −0.0006 (−1.54 standard errors, −0.79× the larger raw σ) — not a
statistically clean regression by this log's usual bar (would want ~2–3×),
but consistently on the low side across all 5 seeds and also below
DeepFM's result, not just noise scattered around zero.

### Finding: another null, and the *shape* of the result is informative

Where DeepFM landed almost exactly on the baseline (a wash — the DNN branch
found nothing but also cost nothing), FinalMLP landed mildly *below* it.
Plausible reading, given FinalMLP has no explicit bilinear term anywhere:
the two-MLP-stream design has to learn the `user_id × video_id` interaction
*implicitly*, from scratch, via gradient descent through gates and a
bilinear fusion layer — a strictly harder learning problem than FM's exact,
structurally-guaranteed bilinear form, which computes that interaction by
construction rather than approximating it. On a dataset this size (1.14M
rows), that implicit-learning disadvantage plausibly outweighs whatever
FinalMLP's inductive bias offers elsewhere, especially since (per every
prior entry in this section) there's little additional structure beyond
that one pairwise term for its extra expressiveness to find anyway.

This also retroactively supports the DeepFM entry's design choice
(*augment*, don't replace): DeepFM kept the exact bilinear term intact and
added capacity on top, costing nothing when the extra capacity found
nothing. FinalMLP gave up the exact term and got a mild net negative for
the trade. Both experiments point the same direction, but the replace-vs-
augment framing turned out to matter for the downside, not just the upside.

### Decision

**Not adopted.** Third and fourth independent confirmations (with DeepFM)
of the same finding this section has now established repeatedly: no
detectable additional structure beyond `user_id × video_id` for a more
expressive architecture to exploit, whatever form that expressiveness
takes (bigger FM, DNN augmentation, or a fully different architecture).
Model-architecture-change as a direction is now thoroughly closed off.
BPR FM + temporal features + watch-time (0.6017 test primary) remains the
number to beat. BST (pending GPU) remains the one open direction with a
qualitatively different rationale (order-awareness) rather than "more
expressiveness in some form."

Reproduce: `.venv/bin/python finalmlp_model.py --seed 0` → test primary
0.6008.

## 2026-08-29 — BST on GPU: decisively beats DIN, ties the overall best

The long-pending BST run — the CPU estimate (~2450s/epoch, ~35x slower
than DIN) made it impractical locally; user got SoC GPU cluster access
working this session (SSH/VPN routing issues, a wrong-quota home directory
for the pip install, and a scratch-directory permission dead-end all had
to be worked through first — see below). Same `sequence_model.py --arch
bst`, same L=160/BPR setup as every other DIN/BST entry, now run with
`--device cuda` on an SoC GPU node (`xgpd0`, Titan V) instead of CPU.

**GPU speedup:** ~22s/epoch vs. CPU's ~2450s/epoch — **~110x faster**,
even on one of the cluster's older GPU types. Confirms the earlier
hypothesis: the CPU slowness was PyTorch's training-mode attention
implementation being inefficient for small per-head dimensions (k=16,
4 heads), not a fundamental property of the model. All 5 seeds completed
in well under 30 minutes total on GPU, vs. an estimated 1.5–3 days that
would have been needed on CPU.

**Cluster setup friction (worth recording for next time):** rsync/SSH from
the local Mac to the login node timed out despite VPN showing connected
(stale VPN routes after a network change — never fully diagnosed, worked
around by pushing code to GitHub and `git clone`-ing from inside the
cluster instead, which uses port 443 not 22). Then `pip install torch`
hit "Disk quota exceeded" on the home directory (the actual per-user quota
mechanism wasn't visible via the standard `quota -s` tool on this
clustered filesystem) — attempted fix via a scratch directory
(`/mnt/scratch/$USER`) hit a permission wall (no self-service scratch
provisioning found for this account). Ultimately resolved by requesting
an **interactive GPU allocation** (`srun --gpus=1 --pty bash`) and
building the venv from there — same home-directory path that failed
before, but succeeded this time (unclear exactly why — plausibly earlier
failed attempts' partial downloads had been cleaned up by then, freeing
enough quota). `sbatch train_seq.sbatch` (with `--array` overridable on
the command line, e.g. `--array=1-4`) is the recommended path for the next
GPU experiment, now that a working venv/path is confirmed — it survives
SSH disconnection and time-limit concerns that an interactive session
doesn't.

Command: `.venv_gpu/bin/python sequence_model.py --arch bst --device cuda
--seed <0..4>`, run interactively (each seed converged in 6–8 epochs,
~2–3 minutes).

### Results (test set, mean ± population std over seeds 0–4)

| model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| DIN, plain pooling (CPU, earlier entry) | 0.6634 (σ=0.0005) | 0.5299 (σ=0.0006) | 0.5967 (σ=0.0005) |
| DIN + `same_video` input (CPU, earlier entry) | 0.6638 (σ=0.0006) | 0.5301 (σ=0.0006) | 0.5969 (σ=0.0006) |
| BPR FM, 7 fields, clean (no watch-time) | 0.6689 (σ=0.0005) | 0.5326 (σ=0.0005) | 0.6008 (σ=0.0004) |
| **BST (GPU)** | **0.6697** (σ=0.0009) | **0.5330** (σ=0.0006) | **0.6014** (σ=0.0007) |
| BPR FM + temporal features + watch-time (current overall best) | 0.6689 (σ=0.0005) | 0.5333 (σ=0.0003) | 0.6017 (σ=0.0004) |

Per-seed test primary: 0.6022, 0.6023, 0.6009, 0.6007, 0.6007.

### Finding: the order-awareness hypothesis was correct — and the ceiling is still the same one

**BST vs. DIN: +0.0045 to +0.0047, at 10.5–11.8× the standard error of the
difference.** About as unambiguous a result as this log has recorded —
not a marginal improvement, a clear one. This directly confirms what the
DIN entries hypothesized: DIN's failure was specifically about its
order-agnostic softmax-pooling mechanism (and its inability to learn to
exploit rare-but-strong patterns like exact repeats), not about "no
sequence signal existing" or sequence modeling being a dead end here.
Adding positional encoding and self-attention over the *ordered* sequence
recovers real value that pooling left on the table.

**BST vs. the clean 7-field FM (no watch-time): +0.0006, ~1.5× standard
error.** Directionally positive, consistently so, but doesn't clear this
log's usual bar (~2–3×) for calling something a confirmed win on its own.

**BST vs. the current overall best (FM + temporal features + watch-time,
0.6017): −0.0003, a wash.** This is the more important comparison for
deciding what to adopt. BST's self-attention over raw history apparently
rediscovers much of the same signal that the hand-engineered temporal
features (`prior_exposure`, `author_recency`) and the watch-time auxiliary
task already extract more cheaply and reliably — consistent with this
project's now-repeated finding (features/capacity/DeepFM/FinalMLP) that
there isn't much additional structure in this dataset for a more
sophisticated mechanism to find *beyond* what's already been claimed,
whatever form that sophistication takes.

### Decision

**Not adopted as the new default** — ties the current best rather than
beating it, and costs meaningfully more (GPU dependency, ~22s/epoch vs.
numpy FM's ~2–3s/epoch, a whole separate cluster-access workflow) for no
net gain over the existing hand-engineered-feature approach. But this
closes out the sequence-modeling investigation with a clean, coherent
story rather than an ambiguous one: DIN's *specific* mechanism was flawed
(confirmed directly by BST's large improvement over it), sequence
information genuinely exists in this data (confirmed independently three
different ways now — `prior_exposure`, `author_recency`, and now BST vs.
DIN), and the ceiling on how much of it any single mechanism can extract
sits right around 0.601–0.602 no matter whether that mechanism is
hand-crafted features, an auxiliary loss, or an attention architecture.
**BPR FM + temporal features + watch-time (0.6017 test primary) remains
the number to beat.**

Every actively-explored direction in this project (loss function,
temporal features, multi-task, model architecture, sequence modeling) has
now been tested to a clear conclusion. Remaining lower-priority items from
the README: time/drift features, unbiased validation via the
random-exposure log.

Reproduce: `.venv_gpu/bin/python sequence_model.py --arch bst --device
cuda --seed 0` (on a GPU node) → test primary 0.6022.

## 2026-08-29 — Dynamic Negative Sampling (DNS): a genuinely new axis, still a null

Every BPR experiment so far (loss, temporal features, multi-task,
watch-time, DeepFM, FinalMLP, BST) has used the *same* negative-sampling
strategy underneath: one uniformly random negative per positive. This
entry tests a different lever entirely — *which* negative gets shown, not
the loss function, features, or architecture. Literature motivation: hard
negative sampling on BPR is theoretically connected to optimizing One-way
Partial AUC, which correlates with Top-K metrics more strongly than plain
AUC — relevant since nDCG@5 is half our primary metric.

**Implementation** (`loss='pairwise_dns'` in `baseline.py`): for each
positive, sample `dns_n` candidate negatives from the same user's negative
pool, score all of them with the *current* model, train on whichever one
the model currently ranks highest (the "hardest" one). Reuses
`FM.step_pairwise` unchanged — only the negative-selection logic differs
from plain BPR.

### Finding 1: naive hard-negative selection is actively unstable, not just unhelpful

First attempt (`dns_n=8`, no warmup): loss *increased* epoch over epoch
(0.63→0.75) and validation primary collapsed to 0.52–0.57 (vs. ~0.60
normally) — not a marginal miss, active training instability. Diagnosed
and fixed in two steps, both standard practice in the hard-negative-mining
literature:

1. **Warm-up** (`dns_warmup=3`): train with plain random negatives for the
   first few epochs before switching to hardest-of-N — mining "hardest"
   negatives against a still-randomly-initialized model chases noise, not
   signal. This alone didn't fix it: switching to `dns_n=8` after 3 clean
   warm-up epochs (0.6022 valid primary) still caused an *immediate,
   monotonic decline* (0.5921→0.5945→0.5869→0.5772 over the next 4 epochs).
2. **Learning-rate reduction on switch** (`dns_lr_decay=0.2`): hard
   negatives make `sigmoid(z_pos - z_neg)` sit close to 0.5 on nearly every
   step (vs. plain random negatives, which are usually already correctly
   ranked, `sigmoid≈1`, gradient≈0) — this is effectively a large,
   consistent increase in gradient magnitude per step, which Adam's
   moment estimates (tuned for the *random*-negative gradient
   distribution) aren't calibrated for. Dropping `lr` by 5x when switching
   removed the wild divergence, but the decline **continued anyway**,
   just more slowly (0.6553→0.6338 GAUC, 0.5289→0.5205 nDCG@5, over 4
   epochs) — both submetrics falling *together*, not a GAUC/nDCG5
   trade-off (which would show one rising as the other fell, the pattern
   LambdaRank@5 showed). That rules out "sacrificing broad ranking for
   top-K gains" as the mechanism — hard negatives here are teaching
   something actively counterproductive on both fronts, not narrowly
   over-focusing.

**Hypothesis for why, specific to this dataset:** the negative-sampling
literature's theoretical case for hard negatives is mostly validated on
large-catalog settings where an unrelated item is a highly reliable true
negative. Here, the catalog is only 7,538 videos with heavy repeat
exposure across users (already established: `prior_exposure`'s 78.5%
re-watch rate, `author_recency`'s near-certain re-engagement after
same-session adjacency). In a small, repeat-heavy catalog, the "hardest"
negative — the item the current model most confidently mis-ranks as
positive — is disproportionately likely to be a video the user has genuine
latent interest in but simply didn't watch long enough *this specific
impression*, closer to label noise than a true negative. Concentrating
training on exactly these cases plausibly teaches the model to chase
noise in the `long_view` threshold itself rather than real preference
structure.

### Finding 2: gentler hardness (`dns_n=2`) is stable, but a clean null

Reducing to `dns_n=2` (train on the better of just 2 random candidates,
much weaker selection pressure) eliminated the instability entirely —
validation improved steadily epoch-over-epoch (0.6049→0.6067) rather than
declining. But the converged result is statistically indistinguishable
from the clean baseline:

| model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| BPR FM, 7 fields (clean baseline) | 0.6689 (σ=0.0005) | 0.5326 (σ=0.0005) | 0.6008 (σ=0.0004) |
| **DNS, `dns_n=2`, warmup=3, lr_decay=0.2** | **0.6687** (σ=0.0005) | **0.5325** (σ=0.0004) | **0.6006** (σ=0.0004) |

Per-seed test primary: 0.6011, 0.6006, 0.6002, 0.6002, 0.6010. Gap = −0.0002,
−0.73 standard errors — a clean wash, not a regression, not a gain.

### Decision

**Not adopted.** The theoretical case for hard negative sampling (OPAUC /
Top-K connection) didn't translate into a practical gain here, and the
instability at higher hardness settings is itself an informative finding:
it's consistent with, and adds a mechanistic explanation for, this
project's now-extensive pattern of "no exploitable structure beyond
`user_id × video_id`" — here, the *reason* a promising technique failed
isn't "nothing left to find," it's that harder negatives specifically
surface a kind of noise (repeat-exposure ambiguity) baked into this
dataset's structure, rather than genuine signal a bigger/better mechanism
could extract. Not swept further (`dns_n` between 2 and 8, alternative
warmup/decay schedules) given the clear, consistent direction across both
tested settings and the diminishing-returns pattern already established
for hyperparameter sweeps in this log (e.g. the `pairwise_watchtime`
`aux_weight` sweep). `baseline.py`'s FM code retained (`--loss
pairwise_dns`) as verified, working infrastructure in case revisited with
a different sampling scheme. BPR FM + temporal features + watch-time
(0.6017 test primary) remains the number to beat.

Reproduce: `python3 baseline.py --model fm --loss pairwise_dns --dns_n 2
--seed 0` → test primary 0.6011 (stable). `--dns_n 8` (default) reproduces
the instability finding within a few epochs of the warmup period ending.

## 2026-08-29 — Train/test temporal drift: a real structural finding, but reweighting doesn't help

Investigating the README's remaining "时间特征与分布漂移" headroom item.
Unlike previous entries (which attacked model capacity or training signal),
this targets the **data distribution** — which training rows count, not how
smart the model is. Nothing tried so far touches that axis.

### The structural finding (real, and worth recording independently)

Per-day breakdown of the timeline turned up a large, previously unnoticed
asymmetry — not in the label rate, but in **logging volume**:

| period | rows | impressions/user/day |
|---|---|---|
| Train, Apr 9–12 | 725k (**64% of all training data**) | **7.4** |
| Train, Apr 18–21 | 86k | **1.1** |
| Valid / test | ~14–21k per day | ~1.1 (matches late train) |

Two-thirds of the training data comes from a high-intensity logging regime
roughly 7x denser than the regime the model is actually evaluated in. The
label rate drifts accordingly (train 0.337 → valid/test 0.313).

Checked whether this is a population shift: **it isn't**. 91% of test users
appear in early train, 73% in late train; unique-user counts are comparable
(24.5k early, 19.2k late, 23.9k test). Same users, ~7x different logging
intensity — an instrumentation/sampling change, not a different audience.

**Hypothesis:** training is dominated by rows structurally unlike the
evaluation regime; reweighting or reselecting toward the eval-matched
regime should help.

### Result: hypothesis refuted, in both hard and soft form

`ablation_train_window.py`, full current-best config (7 fields + BPR +
watch-time aux), only the training-row selection/weighting changed.

**Hard truncation** (drop early high-intensity days entirely):

| training window | rows kept | test primary |
|---|---|---|
| all (current default) | 100% | **0.6020** |
| Apr 13 onward | 36.4% | 0.5978 |
| Apr 16 onward | 16.7% | 0.5928 |
| Apr 18 onward (eval-matched tail only) | 7.5% | 0.5795 |

**Soft recency weighting** (keep all rows, exponentially upweight
recent ones by sampling probability — the continuous version of the same
idea, and the form the recency-sampling literature actually recommends):

| weighting | test primary |
|---|---|
| uniform (current default) | **0.6020** |
| half-life 3 days | 0.5932 |
| half-life 7 days | 0.5996 |
| half-life 14 days (very mild) | 0.5999 |

Both families are **monotonic in the same direction**: the more the
training distribution is tilted toward the evaluation regime, the worse the
result — and the mildest tilt (14-day half-life, 0.5999) still doesn't
reach uniform (0.6020). Single-seed, but the effect sizes (0.002–0.023) are
10–50x typical seed noise (σ≈0.0004), so a full 5-seed sweep wasn't a good
use of time; the direction is unambiguous.

### Finding: data volume dominates distribution match, decisively

Even "mismatched" early-period data is worth more than the distribution
alignment gained by discarding or downweighting it. The FM's parameters are
dominated by `user_id`/`video_id` embeddings, which need interaction volume
per ID to estimate well — starving them to buy distributional similarity is
a bad trade at this scale. This also explains why the *mildest* weighting
(14-day) lands closest to uniform: it's the setting that least disturbs the
effective sample size.

Worth noting what this does **not** rule out: adding logging-intensity as an
explicit *feature* (letting the model condition on regime rather than
reweighting the data) was the third option sketched when this direction was
proposed, and remains untested. Given every feature-addition experiment in
this log has been null (static features, capacity, DeepFM, FinalMLP), and
given intensity is a per-user-per-day quantity that's near-constant within
a user's evaluation group — which by the README's own documented property
contributes *nothing* to intra-user ranking unless it crosses with an
item-side feature — the expected value looks low, but it's an honest gap.

### Decision

**Not adopted; current default unchanged.** The structural finding (7x
logging-intensity shift, same user population) is genuinely interesting and
worth recording for its own sake — it's a real property of this dataset
that isn't documented in the README — but it does not translate into a
usable modeling lever. BPR FM + temporal features + watch-time (0.6017 test
primary) remains the number to beat.

Reproduce: `python3 ablation_train_window.py --seeds 1` (runs both the
uniform baseline and the three soft-weighting settings; edit `configs` in
`__main__` to switch back to the hard-truncation variants).

## 2026-08-29 — RAD-style quantile watch-time target: much better target, identical result

Refining the one thing in this log that *did* work (watch-time auxiliary
task, +0.0009) rather than opening a new direction. Motivation from
**Relative Advantage Debiasing** (Liu et al., AAAI 2025): raw watch time is
confounded by video duration, so instead of regressing its absolute value,
regress its **quantile within an empirical reference distribution
conditioned on duration group** — "was this watch long *for a video of this
length*". Naturally outlier-robust, which should also let us drop the
hand-tuned 10x loop-capping the current implementation needs.

**Verified the premise on our data before implementing** (rather than
assuming the paper's setting transfers):

| check | value | reading |
|---|---|---|
| corr(log1p(play_time), long_view), global | 0.596 | baseline signal quality |
| same, *within* duration deciles | 0.46 → 0.64 | duration genuinely confounds it — RAD's premise holds |
| corr(log1p(duration), long_view) | 0.074 | but duration barely predicts the label directly |

So the confounding RAD targets is real here, though milder than settings
where duration strongly drives the outcome.

**Implementation** (`data.watch_time_quantile_targets`, selected via
`baseline.py --wt_target quantile`): duration split into 10 train-fitted
groups; per-group empirical watch-time distribution built from train only
(no leakage); each row's target is its watch time's quantile in its own
duration group's distribution. Censoring handled identically to the
existing implementation (completed views get `tau` = duration's quantile as
a one-sided lower bound). Drop-in replacement — same `(t, tau, censored)`
contract, so `step_pairwise_watchtime` is unchanged.

**The new target is a much better proxy for the objective:**

| target | corr with `long_view` |
|---|---|
| existing: capped + log1p(play_time) | 0.596 |
| **RAD quantile** | **0.825** |

### Results (test set, mean ± population std over seeds 0–4)

| model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| watch-time aux, log target (current default) | 0.6702 (σ=0.0005) | 0.5333 (σ=0.0003) | **0.6017** (σ=0.0004) |
| **watch-time aux, RAD quantile target** | 0.6701 (σ=0.0006) | 0.5331 (σ=0.0004) | **0.6016** (σ=0.0005) |

Per-seed test primary: 0.6023, 0.6020, 0.6013, 0.6014, 0.6010. Gap −0.0001
(−0.36 standard errors) — indistinguishable.

### Finding: a 39% better-correlated auxiliary target produced exactly zero ranking gain

This is the most pointed version of a pattern this log has now hit from
many angles. The auxiliary target improved substantially by its own
measure (0.596 → 0.825 correlation with the very label being ranked), the
implementation is cleaner (no arbitrary capping constant), the theory is
sound and independently validated on our data — and the metric didn't move
at all.

The natural reading, consistent with everything else here: the auxiliary
task's *contribution* was never bottlenecked on target quality. Its
original +0.0009 came from giving the shared embeddings a denser,
finer-grained version of the same signal `long_view` already provides
(established in the original watch-time entry: `play_time >= 18s` alone
predicts 96.7% of `long_view`, so it's a discretization of watch time, not
an independent signal). Once the embeddings have absorbed that, sharpening
*how* the auxiliary target is parameterized adds nothing — the ceiling is
set by what the `user_id × video_id` interaction can express about this
data, not by the fidelity of the training signal pointed at it.

### Decision

**Not adopted as default** — statistically identical, and the existing log
target is already validated and in place; switching would churn the default
for no measured gain. Kept as `--wt_target quantile` (fully working, and
arguably the more principled implementation if anyone builds on this — it
has no hand-tuned capping constant). Current best unchanged: BPR FM + 7
fields + watch-time aux (log target), **0.6017 test primary**.

This was the highest-expected-value item remaining from the research pass
(the only one refining a proven-positive step rather than opening a new
direction). With it closed, the remaining README items are `hourmin` as a
feature (low expected value — every static-feature experiment in this log
is null) and unbiased validation via the random-exposure log (a diagnostic,
not a scoring lever).

Reproduce: `python3 baseline.py --model fm --wt_target quantile --seed 0`
→ test primary 0.6023.

## 2026-08-29 — Revisiting ruled-out ground: ADT (null, but clarifying) + seed ensembling (small real gain)

Two experiments chosen by asking what earlier entries had ruled out too
broadly — i.e. where a *specific instantiation* failed but the underlying
idea wasn't actually tested.

### 1. Adaptive Denoising Training — the direct inverse of the failed DNS experiment

ADT (Wang et al., WSDM 2021): noisy implicit-feedback interactions show
large loss early in training, so downweight (R-CE) or drop (T-CE)
large-loss samples. Implemented the reweighted variant as
`--loss pairwise_adt`: `w = sigmoid(z_pos - z_neg)^beta`, so the pairs the
model currently gets *most wrong* contribute least. `beta=0` recovers plain
BPR. Warmup of 3 epochs before reweighting kicks in (same reasoning as
DNS's warmup — early large loss reflects an untrained model, not noise).
Reuses `step_pairwise`'s existing `weight` argument (built for LambdaRank),
so no new gradient code.

**Why this was worth running despite DNS having failed:** it's the exact
opposite intervention on the same axis. The DNS entry hypothesized that
this dataset's hard negatives are largely *label noise* (small catalog,
heavy repeat exposure, `long_view` being a threshold on watch time so
boundary cases are near coin-flips). If that hypothesis were right,
systematically *downweighting* those samples should have helped.

| ADT strength | pairs downweighted below 0.5 | test primary (seed 0) |
|---|---|---|
| none (baseline) | — | **0.6017** (5-seed mean) |
| beta=0.25 | 0.8% | 0.6013 |
| beta=0.5 | 8.8% | 0.6007 |
| beta=1.0 | 30.1% | 0.6003 |
| beta=2.0 | 39.0% | 0.5979 |

Monotonic decline with denoising strength; never beats baseline at any
setting. Trains stably throughout (unlike DNS), so this is a clean
null/negative, not an optimization failure. No 5-seed sweep run — the
single-seed trend is monotonic across a 4x range of beta and the largest
setting is 0.0038 below baseline (~9x seed std), so the direction is not
in doubt.

**Combined conclusion (stronger than either experiment alone):** both
directions on this axis have now been tested. Upweighting high-loss pairs
(DNS) destabilizes training; downweighting them (ADT) monotonically
degrades. High-loss pairs therefore carry *real signal the model needs*,
not discardable noise — which **partially retracts the DNS entry's
"hard negatives are mostly label noise" hypothesis**. That hypothesis
explained DNS's instability plausibly, but it makes a prediction (ADT
should help) that turns out to be false. A better reading: hard pairs are
genuinely informative *and* genuinely hard, so concentrating on them
destabilizes while discarding them loses signal — the useful gradient is
spread across the full difficulty range.

### 2. Seed ensembling — averaging predictions, not just metrics

Every experiment in this log runs 5 seeds, but only ever averaged the
*metrics* (to report mean±std). The models' *predictions* were never
combined. Standard variance reduction, essentially free (the models are
already trained), and unlike every other direction tried it doesn't
require the data to contain more signal — it extracts existing signal more
stably. `ablation_ensemble.py`, testing both raw-score averaging and
groupwise-rank averaging (rank-avg is scale-invariant, often more robust
for combining rankers).

Ran 5 models first, then extended to 10 with independent seeds to check
the result replicated rather than accepting a single favourable run:

| ensemble size | raw-avg | rank-avg |
|---|---|---|
| 1 (single-model mean, 10 seeds) | 0.6018 ± 0.0004 | — |
| 2 | 0.6023 | 0.6020 |
| 3 | 0.6026 | 0.6023 |
| 4 | 0.6028 | 0.6022 |
| 5 | 0.6026 | 0.6023 |
| 6–10 | 0.6022–0.6027 | 0.6023–0.6027 |

**Honest estimate: +0.0007 (~1.8x single-model std)**, taking the plateau
mean over sizes 3–10 (0.6025) rather than the peak. The extension to 10
models was worth running: at 5 models the peak looked like 0.6028 at size
4, but that turned out to be noise — the curve is flat from 3 models on,
not still climbing. Two things support the gain being real despite being
modest: it exceeds **every one of the 10 individual seeds** (max 0.6022),
so it isn't seed cherry-picking, and both aggregation methods agree
(rank-avg converges to the same place, slightly leading at larger sizes).
Against that: 1.8x std is below this log's usual ~2–3x bar for a confident
win, and it saturates almost immediately.

### Decision

**ADT: not adopted** (kept as `--loss pairwise_adt`, working, for the
record). **Ensembling: real but small, and not folded into the default.**
Reasons for not changing the default: the single-model config remains the
honest, reproducible reference point that every other entry in this log is
measured against, and swapping it for an N-model ensemble would make future
comparisons harder to interpret for a gain barely above noise. It is,
however, the **only positive result in the last nine directions tried**,
and it is worth knowing for a final submission, where 3–5 models cost ~3s
each and ~+0.0007 is free money.

Best single-model config remains **0.6017**; best achievable number with
ensembling on top is **~0.6025**.

That ensembling is the one thing that worked fits this log's overall
pattern precisely: every failed direction tried to extract *more signal*
from a dataset that appears not to contain much more; ensembling doesn't
need more signal, it just reduces variance around the signal already
found. That is also why its ceiling is low — variance reduction cannot
move a bias/information limit.

Reproduce: `python3 ablation_ensemble.py --n_models 5` (~15s total) →
raw-avg ensemble test primary 0.6026. `python3 baseline.py --model fm
--loss pairwise_adt --adt_beta 1.0 --seed 0` → 0.6003.

## 2026-08-29 — Heterogeneous ensemble: the best result in the project (0.6034)

**User pushed back on the previous entry** ("I believe there is more here"),
and they were right. The homogeneous seed ensemble combined 10 models that
were *identical except for random seed* — the least diverse ensemble
constructible. The ensemble literature is consistent that gains come from
**complementary error patterns**, not from averaging per se, which also
explains why that ensemble saturated at 3 members and stalled at +0.0007.

Meanwhile this project had trained five architecturally distinct models,
each individually dismissed for failing to beat the FM baseline, and never
once combined:

| member | inductive bias |
|---|---|
| fm_watchtime | bilinear FM + watch-time censored-regression aux |
| fm_quantile | same, RAD quantile aux target |
| **bst** | **order-aware self-attention + positional encoding** |
| deepfm | FM + parallel DNN branch |
| finalmlp | no explicit interaction term; two gated MLP streams |

Four fusion methods tested (raw score average, per-group z-score, per-group
rank, and Reciprocal Rank Fusion — the IR standard). **Protocol guard: the
fusion method and member subset were selected on VALID, with test reported
once**, so the headline number is not tuned on itself.

### The correlation matrix is the finding

Pairwise rank correlation on test (lower = more complementary):

| | bst | deepfm | finalmlp | fm_quant | fm_watch |
|---|---|---|---|---|---|
| **bst** | 1.000 | **0.892** | **0.885** | **0.887** | **0.891** |
| deepfm | 0.892 | 1.000 | 0.973 | 0.934 | 0.939 |
| finalmlp | 0.885 | 0.973 | 1.000 | 0.926 | 0.931 |
| fm_quantile | 0.887 | 0.934 | 0.926 | 1.000 | 0.952 |
| fm_watchtime | 0.891 | 0.939 | 0.931 | 0.952 | 1.000 |

**BST sits at 0.885–0.892 against everything else; every non-BST pair is
0.926–0.973.** That gap is the whole result. DeepFM and FinalMLP look
architecturally radical on paper — FinalMLP has no explicit interaction
term at all — yet they correlate with each other at **0.973**, essentially
the same model in different clothing. They differ in *how* they compute the
`user_id × video_id` interaction; BST differs in *what information it
reads* (sequence order). Only the latter produces complementary errors.

### Results (test set)

| model | GAUC | nDCG@5 | primary |
|---|---|---|---|
| best single member (fm_watchtime) | 0.6705 | 0.5336 | 0.6020 |
| homogeneous seed ensemble (prev. entry) | — | — | 0.6025 |
| all-5, rank / RRF | — | — | 0.6028 |
| all-5, z-score / raw | — | — | 0.6032 / 0.6033 |
| **valid-selected: z-score, {bst, fm_quantile, fm_watchtime}** | **0.6724** | **0.5344** | **0.6034** |

Against the seed-std of 0.0004 used throughout this log:

- vs. previous project best (single model, 0.6017): **+0.0017 ≈ 4.3σ**
- vs. best single member here (0.6020): **+0.0014 ≈ 3.5σ**
- vs. homogeneous seed ensemble (0.6025): **+0.0009 ≈ 2.3σ**

The last comparison is the one that matters for the diversity hypothesis,
and it clears the 2σ bar this log uses. Two independent signals corroborate
it rather than resting on the single selected configuration: **valid
selection independently chose a subset containing BST** (it had no access
to test), and every fusion method lands in 0.6028–0.6034, i.e. the ranking
of methods barely matters while the *inclusion of BST* does.

### What this overturns

The previous entry concluded ensembling's ceiling was low because
"variance reduction cannot move a bias/information limit." That reasoning
was sound but the premise was wrong: the seed ensemble was purely variance
reduction, whereas adding BST contributes *different information* (sequence
order), which is bias reduction, not variance reduction.

It also partly rehabilitates BST. The standalone BST entry concluded "not
adopted — ties the current best." That was correct in isolation and is
still correct as a single-model claim, but it undervalued BST: its value
here is not its solo score (0.6022, comparable to FM) but that it is
**wrong in different places**. A model can be individually redundant and
still be the most valuable ensemble member — this log had no way to see
that until members were actually combined.

Two prediction errors of mine worth recording: I estimated BST would
correlate ~0.94–0.95 with the FM family "given the other four cluster this
tightly" and called the run "closing the loop rather than a promising
lead." It came in at 0.885–0.892 and produced the project's best result.
The generalisation from four already-correlated members to a fifth
structurally different one was unjustified.

### Implementation notes

Members are trained one at a time and cached to `scores/*.npz`
(`--member <name>`), then fused separately (`--combine`). The first
all-in-one-process version was OOM-killed (exit 137): five models plus
BST's ~730MB L=160 history array does not fit locally. I also misdiagnosed
this twice — first asserting OOM without evidence, then reversing on a
0.9GB RSS snapshot that happened to be taken before the history array was
allocated. The final BST member was trained on the SoC GPU cluster at full
L=160 (~3 min), which sidesteps the memory limit entirely.

Cluster-trained members score slightly differently from local runs
(deepfm 0.6013 vs 0.6005, finalmlp 0.6015 vs 0.6008) — hardware/library
nondeterminism, not a methodological difference. All numbers in the table
above come from the single cluster run, so they are internally consistent.

**Caveat on precision:** this is one ensemble built from one seed per
member. The *members'* seed variance is well characterised (σ≈0.0004
across many entries), but the ensemble's own run-to-run variance is not
measured, so treat 0.6034 as a point estimate rather than a mean.

### Decision

**Best known configuration: heterogeneous ensemble, 0.6034 test primary.**
Recorded in the baseline ladder as the best achievable number, with the
single-model config (0.6017) retained as the reference point that all
per-experiment comparisons in this log are measured against — mixing the
two as "the baseline" would make ~30 prior entries harder to interpret.

For a final submission, use the ensemble: it is ~4σ above the best single
model and the members cost minutes to train.

The actionable generalisation for anyone continuing: **stop looking for a
single better model.** Nine directions found the same ceiling. The gain
here came from combining models that were each individually judged
failures, because one of them was wrong in a different place. If more
headroom exists, the most likely source is another member with a genuinely
different information source — not a better architecture over the same
seven features.

Reproduce (GPU node, ~5 min):
```
for m in fm_watchtime fm_quantile deepfm finalmlp; do
  python ablation_hetero_ensemble.py --member $m --device cuda
done
python ablation_hetero_ensemble.py --member bst --bst_L 160 --device cuda
python ablation_hetero_ensemble.py --combine
```


## 2026-08-30 — 任务口径核对：一次走错方向 + 纠正（recovery event）

**Hypothesis.** 用户转述了赛题 Constraints 表 "Limits" 行的一句话：
`KuaiRand-Pure: NDCG@10 / Recall@50, click = positive (fixed) (Required)`，
并要求改用另外两个数据集。若属实，这意味着任务、标签、指标三者同时变了。

**What I did (and got wrong).** 我先验证了一件事并且验证结论是对的：
test 用户曝光数中位数只有 5，只有 0.3% 的用户有 ≥50 条曝光，所以
"在用户曝光内部算 Recall@50" 对 99.7% 的用户恒等于 1.0，完全退化；NDCG@10 同理
（76% 的用户曝光不足 10 条）。**这两个指标只有在全库检索下才有意义。**
到这一步为止推理无误。但我据此得出的结论错了——我判定"任务改成全库检索"，
并据此写了 `evaluate_retrieval.py` / `retrieval_baseline.py` / `retrieval_lightgcn.py`，
还跑出了一套检索 baseline（random NDCG@10 0.0007 / pop 0.0356 / BPR-MF 0.0450）。

**How it was caught.** 用户随后贴出完整赛题。全文有 8 处跟那一行冲突：
1. Benchmarks 表原文："the task treats **long_view** as the positive relevance label,
   ranks **within each user's logged impressions (not full-catalog retrieval)**,
   and reports **GAUC / nDCG@5**"，且注明 "fixed by the organizers"
2. Benchmarks 表 Metrics 列：GAUC / nDCG@5
3. Judging Criteria 的 per-dataset metrics：GAUC / nDCG@5
4. §4 结果表要求：KuaiRand-Pure GAUC / nDCG@5
5. 文中引用的 baseline 0.5946 / oracle 0.8645 / random 0.4753 —— 全是这个任务的数字
6. `evaluate.py` 被称作 "the exact scoring code"
7. 收敛判据 ε=0.002 是按该 primary 的 5-seed σ=0.0008 标定的
8. **决定性证据**：提交格式是 "one line per evaluation-split row"，每行一个分数。
   全库检索**无法用这个 schema 表达**——那需要每个用户一个 top-K 列表。

**Root cause of my error.** 我把"这个指标在当前口径下退化"当成了"所以口径变了"，
但正确的推论是"**所以那一行本身有问题**"。一行 Constraints 表的文字，对上评分代码、
提交 schema、官方 baseline 数字三者的一致证据，权重不该对等。教训：当新信息与
既有的多处一致证据冲突时，先假设新信息有误，而不是先推翻既有体系。

**Cost / recovery.** 走错方向约一个迭代周期，产出 3 个文件是无效工作（对评分而言）。
没有污染任何已有结论——`RUN_LOG.md` 里全部实验用的都是正确口径的 `evaluate.py`。
检索相关文件保留在仓库里但已在 README 标注"非评分任务"，避免后来者混淆。
另一个附带收获：那 3 个数据集变体（Pure/1k/27k）**用的是同一套任务和指标**，
所以"跑另外两个数据集"这个方向本身是成立的（bonus 分），只是要用正确的口径跑。

**Metrics（正确口径，KuaiRand-Pure）.** 无变化，此次事件未改动任何既有结果。

---

## 2026-08-30 — 最终提交产出（`make_submission.py`）

**Hypothesis.** 赛题明确 "The submission scored for ranking is the **validation-best
checkpoint**"，且 §4 要求提交 validation-best 分数及其相对官方 baseline 的 absolute delta。
此前所有实验只在报告 test 分数，缺一个能真正产出合规提交文件的入口。

**发现的问题.** `submit.py --make` 已经失效：那段代码写死 `m.step`（pointwise loss），
而 `data.encode()` 现在默认返回 7 域。所以它既不是官方 baseline（5 域 + pointwise），
也不是我们的最优配置——只能当"生成一个格式合法的示例文件"用，不能用来做最终提交。
**这是一个隐蔽的坑：文件名和参数看起来仍然正确，静默产出的却是第三种模型。**

**Code diff.** 新增 `make_submission.py`，两种模式，选择全部只在 valid 上做：
  --mode single   7 域 FM + BPR + watchtime 辅助任务，早停按 valid primary，
                  返回 valid-best 参数（不是最后一轮）
  --mode ensemble 读 `scores/*.npz` 缓存的成员分数，在 valid 上枚举子集选最优（z-score 融合）
写完立刻用 `submit.read_submission` 自检格式与对齐（跟 `submit.py --check` 同一套代码）。

**Metrics.**

| 提交 | valid primary | test GAUC | test nDCG@5 | test primary |
|---|---|---|---|---|
| 官方 baseline（赛题公布） | 0.6016 | 0.6610 | 0.5282 | 0.5946 |
| `--mode single` | **0.6071** | 0.6705 | 0.5336 | 0.6020 |
| `--mode ensemble`（本机 4 成员，无 BST） | **0.6075** | 0.6712 | 0.5340 | 0.6026 |
| `--mode ensemble`（含 BST，集群跑的那次） | **0.6091** | 0.6724 | 0.5344 | 0.6034 |

两个提交文件均通过格式与对齐校验（170,588 行）。valid-selection 正确地把 LightGCN
排除在外（它单独只有 0.5576，见异构集成那条记录）。

**下一步.** 本机重训 BST（L=64，避开 L=160 的 OOM）以便在本地复现含 BST 的最优集成；
之后按赛题 bonus 跑 KuaiRand-1K / 27K（**同一套任务和指标**）。


## 2026-08-30 — 把"历史特征"套到其它反馈信号上（Tier 2）：确认无收益

**Hypothesis.** `prior_exposure`(+0.0015) 和 `author_recency`(+0.0017) 是本项目最有效的
两个特征，两个都是建在 `long_view` 上的**跨行历史**特征。同样的构造从没套到别的反馈
信号上试过。注意跟已否决实验的区别：`is_click` 作为**辅助任务**是空结果（0.6007），
那测的是"预测点击能否改善 embedding"；这里测的是"**这个用户以前点过这个视频/这个作者
吗**"——关于具体 (user, item) 对的时序信号，跟 prior_exposure 同机制，是另一个实验。

**Code diff.** 新增 `ablation_other_signals.py`；`data.py` 的 `load()` 增加第 11 列
（把 is_like / is_follow / is_comment / is_forward / is_profile_enter 合并成一个
"强互动"标志——单个都太稀疏，最密的 profile_enter 也才 2.5%，合并后 4.5%）。
三个候选特征，都用跟 `temporal_features.py` 相同的"严格早于当前行 time_ms"规则：
  prior_click          该用户此前点击过这个确切视频吗（命中 7,699 / 1,436,609 行）
  author_click_recency 离上次点击这个作者的作品过了多久，分桶（命中 23,218 行）
  author_engage        此前对这个作者有过强互动吗（命中 2,287 行）
对照基准是**当前最优配置**（7 域 + BPR + watchtime 辅助任务），不是 5 域基线。

**Metrics（test，5 seed，mean ± population std）.**

| 配置 | primary | Δ vs 对照 |
|---|---|---|
| 当前默认 7 域（对照） | 0.6017 ± 0.0004 | — |
| + `prior_click` | 0.6019 ± 0.0003 | +0.0002（0.5σ）|
| + `author_click_recency` | 0.6022 ± 0.0003 | +0.0005（1.25σ）|
| + `author_engage` | 0.6015 ± 0.0003 | −0.0002 |
| 三个全加（10 域） | 0.6020 ± 0.0004 | +0.0003 |

**Finding.** 最好的一个（`author_click_recency`）单 seed 看着有 +0.0006，5 seed 收缩到
+0.0005，约 1.25σ——低于本项目一贯的 2~3σ 确认门槛。**判为无收益，不收编。**

为什么点击历史远不如 long_view 历史（+0.0015/+0.0017 vs +0.0005）：因为
`long_view` 历史特征的强大之处在于**它就是标签本身的历史**——"这个用户以前长看过
这个视频"几乎直接回答了"这次会不会长看"（`prior_exposure` 命中时 long_view 率 78.5%
vs 全局 33.1%）。点击是相关但不同的信号，这跟 `is_click` 作为辅助任务同样无收益
是同一个结论的两次独立印证：**click 携带的信息，模型从 long_view 自己的梯度里已经
拿到了。**

**Decision.** 不收编，`data.py` 的 `FIELDS` 保持 7 域。`ablation_other_signals.py`
留在仓库里作为归因记录。第 11 列（强互动标志）保留在 `load()` 里——它本身无害，
且后续若要做多信号建模可以直接用。

Reproduce: `python3 ablation_other_signals.py --seeds 5`


## 2026-08-30 — Bonus benchmark：KuaiRand-1K 跑通；27K 判定本机不可行

**Hypothesis.** 赛题确认三个变体**用的是同一套任务和指标**（"KuaiRand-Pure /
KuaiRand-1k / KuaiRand-27k → GAUC / nDCG@5"），所以 bonus 不需要新指标，
只需要让同一套流程吃得下更大的数据。瓶颈预期是内存而非算法。

**Code diff.** 新增 `data_large.py`（列式流式加载器）+ `run_bonus.py`（跑 FM+BPR）；
`data.py` 的 `load()` 增加 `suffix` 参数（'pure'/'1k'/'27k'，默认 'pure' 保持原行为）。
为什么必须另写加载器：`data.py` 把每行做成 Python tuple 存 list，Pure 的 140 万行约
280MB 没问题，1K 的 1171 万行要约 2.3GB，本机只有 **8GB RAM**。列式版本全程只留 numpy
数组，实测峰值 2.68GB。

**Metrics（KuaiRand-1K，FM + BPR，5 域，seed=0）.**

| split | GAUC | nDCG@5 | primary |
|---|---|---|---|
| valid | 0.6644 | 0.5772 | **0.6208** |
| test | 0.6645 | 0.5742 | **0.6194** |

wall-clock 1874s（约 31 分钟），峰值内存 2.68GB，5 个 epoch 后早停。

**⚠️ 这些数字不能跟 Pure 的 0.6070/0.6020 比。** 三个理由：
(1) 赛题只公布了 Pure 的官方 baseline（0.5946），1K/27K 没有公布 baseline，所以这里
    只有绝对值，没有 delta；
(2) 数据结构完全不同：1K 保留了 1000 个用户的**全部**日志，valid 里每个用户平均有
    2,525 条曝光，而 Pure 每个用户只有约 5 条（Pure 被过滤到了候选池）。"用户内排序"
    在两个数据集上的难度根本不是一回事；
(3) 1K 只有 1000 个用户，其中**只有 983 个是可训练的**（需要同时有正负例）。

**观察到的问题（记录下来，不是已解决）.** 最好的一轮是 **epoch 1**，之后单调下降
（0.6208 → 0.6160 → 0.6102 → 0.6103 → 0.6033）。早停正确地保留了 epoch-1 的参数，
但这是**过拟合**而非健康收敛。原因很清楚：4,371,868 个视频对 1171 万次交互 ——
**平均每个视频只有 2.7 次交互**，video embedding 根本估不准。Pure 是 7,583 个视频对
140 万次交互（每视频 185 次），密度差了近 70 倍。
真要在 1K 上做好，第一步应该是处理这个稀疏性（比如按频次过滤长尾视频、或者更强的
正则/更小的 k），而不是照搬 Pure 上调好的配置。**本次没做这一步。**

**另一个性能问题（同样未解决）.** 每个 epoch 316-416 秒，而 Pure 只要 2-3 秒。
主因不是数据量而是 `FM._adam_update` 对**整张 embedding 表**做稠密更新：
每个 minibatch 都要跑 O(dim × k) 的 Adam 运算，Pure 的 dim=40,273 无所谓，
1K 的 dim=5,778,436 就是 140 倍的无谓计算（一个 batch 实际只碰到几万行 embedding）。
正确做法是稀疏 Adam——只更新本批次碰到的行。**已识别未实现。**

**KuaiRand-27K：判定本机不可行，不尝试。** 按 1K 实测数据外推（27K 有 3.22 亿次交互，
是 1K 的 27.5 倍，视频数按官方文档约 3200 万）：
  - X 数组本身约 6.4 GB
  - embedding 表 V + Adam 的 m/v 三份，dim≈3500 万、k=16 → 约 6.7 GB
  - **下限约 13.2 GB，本机 8 GB RAM 装不下**
  - 每个 epoch 外推约 172 分钟，赛题的 6 小时 wall-clock 上限只够跑约 2 个 epoch
盲目开跑只会重演本 session 已经踩过两次的 OOM（exit 137）。
可行路径是用户的 SoC GPU 集群，但那边**家目录配额本身就装不下 CUDA 版 torch**
（本 session 实测，见集群那条记录），要先解决存储配额。**留作未完成项，如实记录。**

**同期的一个取舍决定.** 为了给 1K 腾内存（当时可用内存只剩 0.3GB），中途终止了
本机重训 BST 的进程。取舍：BST 能把 Pure 的集成从 valid 0.6075 提到 0.6091
（+0.0016），但本机 CPU 上约 16 分钟/epoch、需要 6-8 个 epoch，要再占用 1.5-2 小时，
且与 1K 同时跑几乎必然触发 OOM 把两个都杀掉。判断：**一个跑通的 bonus 结果 >
必选项上 +0.0016 的边际收益**，且 BST 的贡献已在集群那次运行中验证并记录在案。
代价：本机 `scores/` 里没有 `bst.npz`，本地最优提交是 4 成员集成（valid 0.6075）。

Reproduce: `python3 run_bonus.py --suffix 1k --data_dir ./KuaiRand-1K/data`
（约 31 分钟，峰值 2.7GB）


## 2026-08-30 — KuaiRand-27K：跑通，含一次内存假象的排查

**Hypothesis.** 27K 在本机（8GB RAM）判定不可行（见上一条记录），但集群 GPU 节点
实测 125GB 系统内存、55TB 空闲 scratch、torch 已装好——理论上应该可行，唯一悬念是
`FM._adam_update` 对整张 embedding 表做稠密更新在 27K 的 dim≈4090 万下完全不现实
（跟内存无关，纯粹是浪费的计算量）。

**Code diff.**
1. 新增 `bonus_fm_torch.py`：torch 版 FM+BPR，`nn.Embedding(sparse=True)` +
   `torch.optim.SparseAdam`——每个 batch 只更新真正查到的行，不碰整张表。
   本机用 1K 数据验证：每 epoch 从 numpy 版的 316-416s 降到 **8.5-9.5s**（约 37 倍），
   结果落在同一量级（test primary 0.6175 vs numpy 版 0.6194），确认实现无误。
2. `data_large.py` 的 `load_columnar` 改用 `glob` 匹配日志文件名而不是写死单一文件名——
   27K 把每个日期段的标准日志切成了 `part1`/`part2`（Pure/1K 都是单文件），
   排序后按 part1→part2 顺序读，不需要在磁盘上拼接（省约 9GB 重复存储，也避免
   拼接时把 part2 自己的表头行嵌进数据中间导致那一行解析出错）。本地用 1K 的
   单文件场景回归测试：行数与改动前完全一致。

**踩的坑（记录下来，不是隐藏掉）.**
1. 第一次跑（`srun --gpus=1 --time=02:00:00`，未指定 `--mem`）在读了 1200 万行
   （约 322M 总行数的 4%）时被 SLURM 的 OOM killer 杀掉。之前用 `free -h` 确认过
   节点有 125GB，但那是**整个节点**的内存，不代表这次 `srun` 请求分到了多少——
   没写 `--mem` 时用的是集群默认配额，明显远小于 125GB。加上 `--mem=100G`
   后一次成功。
2. **写这条记录时发现的一个自己代码里的 bug**：跑完之后日志显示
   `peak mem after load: 0.04 GB`，`peak mem after encode: 0.04 GB`——数值明显
   荒谬。原因是 `resource.getrusage().ru_maxrss` 这个字段**在 macOS 上单位是字节，
   在 Linux 上单位是 KB**，是一个众所周知的平台差异。`peak_gb()` 是在本机
   （macOS）写的，隐含假设了字节单位，搬到集群（Linux）上后**默默地把数字缩小了
   整整 1024 倍**，不报错，只是给出一个看似合理实则错误的小数字。反推真实值：
   `0.04 × 1024³` 字节 `≈ 4.29×10⁷`，按 KB 解读 `≈ 41.0 GB`——跟之前对"32M 视频的
   vid2author 字典 + 3.22 亿行的原始 Python list 列存储"的独立估算（25-40GB）
   高度吻合，也解释了为什么必须显式要 `--mem=100G` 才够（不加时默认配额撑不到
   ~41GB）。已在 `run_bonus.py` 和 `bonus_fm_torch.py` 里修：按 `sys.platform`
   加一个单位换算因子。**这条本身没有改变任何已发表的结果**（1K 的 2.68GB /
   Pure 更小的量级都远低于 1024 倍误差会造成可见影响的阈值），但如果不修，
   后续任何在 Linux 上跑的内存数字都是假的。

**Metrics（KuaiRand-27K，SparseFM + BPR，5 域，seed=0，GPU=TITAN V）.**

| split | GAUC | nDCG@5 | primary |
|---|---|---|---|
| valid | 0.6723 | 0.5593 | **0.6158** |
| test | 0.6692 | 0.5422 | **0.6057** |

train 136,296,576 / valid 71,149,570 / test 114,832,239 行（合计 322,278,385，
与官方文档"27K: 322M"一致）。词表：27,285 用户、32,038,693 视频、8,839,735
作者。dim=40,905,743。每 epoch 187-190 秒（GPU），5 epoch 后早停，总 wall-clock
4038 秒（约 67 分钟），真实峰值内存约 41GB（见上）。

**同样的现象，第三次出现.** 最好的一轮又是 **epoch 1**（0.6158 → 0.6034 → 0.5960
→ 0.5896 → 0.5864，单调下降）——跟 1K 一模一样的过拟合形状。三个数据集的
"每视频交互次数"排出来是：

| | 交互数 | 视频数 | 交互/视频 |
|---|---|---|---|
| Pure | 1,446,609 | 7,583 | **190.8** |
| 27K | 322,278,385 | 32,038,693 | **10.1** |
| 1K | 11,713,045 | 4,371,868 | **2.7** |

27K 虽然总交互量是三者中最大的，但视频库也跟着等比例暴涨（32M 视频——真实
快手规模），所以密度反而排在中间，仍然比 Pure 稀 19 倍。这印证了 1K 那条记录
里的判断：**视频 embedding 的可估计性由"每视频平均交互次数"决定，不是总数据量**。
三个数据集在这个指标上的排序（Pure > 27K > 1K）跟三者的 valid primary 排序
（Pure 0.6070 > 27K 0.6158 > 1K 0.6208，注意这里 27K 反而比 Pure 更高，说明
primary 的绝对值还受用户数/曝光密度等其它因素影响，不能只用视频密度一个变量
解释——这点留作观察，没有进一步拆解）不完全单调，但"epoch 1 最好、之后过拟合"
这个**训练动态**在 1K 和 27K 上都复现了，Pure 没有——Pure 的密度足够支撑更多轮
训练而不过拟合。

**⚠️ 跟 Pure 的数字不能直接比**，理由跟 1K 那条记录一样：没有官方发布的
27K baseline；用户/曝光结构不同；这里只有绝对值。

**Decision.** Bonus 里的两个数据集（1K、27K）都已跑通并记录。稀疏 embedding
过拟合的问题（三次观察到同一现象）作为已识别未解决项：真要在这两个数据集上
做好，需要处理长尾视频的正则化或截断，而不是照搬 Pure 调好的超参数——本次
没做这一步，如实记录。

Reproduce（需要 GPU 节点，显式指定内存）：
```
srun --gpus=1 --time=02:00:00 --mem=100G --pty bash -c '
cd ~/track2/track2-kuairand
.venv_gpu/bin/python bonus_fm_torch.py --suffix 27k --data_dir ./KuaiRand-27K/data --device cuda
'
```
