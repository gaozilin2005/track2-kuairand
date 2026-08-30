"""A second, deliberately WIDER autonomous agent run, testing one specific question:
does widening the action space beyond loss/hyperparameter selection change the
Technical Execution ceiling? `agent_loop.py` (the submitted run) stays untouched as
the narrow, reliable reference; this is a separate experiment, not a replacement.

Motivation, grounded in this project's own prior evidence (not a guess): the entire
hand-driven investigation (`RUN_LOG.md`) tried ~20 independent single-model levers --
loss functions, features, capacity, four architectures, training strategies -- and
every one converged on the same ~0.601-0.602 ceiling. The ONLY lever that broke past
it was heterogeneous ensembling (0.6017 -> 0.6034), because it combines models that
make DIFFERENT errors, not because any one of them is individually stronger. If a
wider action space is going to move the needle at all, this is the specific direction
with actual prior evidence behind it -- not a shot in the dark.

New action space (three action types, still zero code written by the agent -- every
action is a subprocess call to an already-verified script):
  train_member       : train one architecture (fm_watchtime / fm_quantile / deepfm /
                        finalmlp / lightgcn -- via ablation_hetero_ensemble.py --member)
                        and cache its own scores to a run-scoped scores_dir.
  check_correlation   : compute pairwise rank correlation between two ALREADY-cached
                        members' VALIDATION predictions -- a genuine diagnostic action,
                        not something computed for the agent and handed to it. This is
                        deliberately the mechanism by which the agent could discover
                        for itself which members are complementary, mirroring exactly
                        the diagnostic-before-deciding discipline the hand-driven work
                        used for RAD/CWM (verify premises against real numbers before
                        trusting a method's framing) -- just applied to ensembling.
  combine             : run the exhaustive valid-selected fusion search over all
                        currently-cached members (ablation_hetero_ensemble.py
                        --combine) and read back the result.
  stop                : self-reported convergence (in addition to the code-enforced
                        eps=0.002/N=3 check, same as agent_loop.py).

BST is deliberately excluded from `member` -- it takes ~2450s/epoch on this machine's
CPU (established in RUN_LOG.md), which would make even one training action consume
most of a reasonable wall-clock budget. This is a stated hardware constraint, not an
oversight: BST is exactly the kind of member prior evidence says would matter most
(it's the low-correlation member that made the hand-driven ensemble work), so its
absence here is a real, known limitation on how far this run can plausibly go,
disclosed up front rather than discovered by a reader after the fact.

The agent trains its OWN members into a dedicated `agent_scores/` directory -- it
never touches the hand-driven track's cached `scores/*.npz` files, so any ensemble
gain it finds is its own, not a re-combination of human-trained artifacts.
"""
import argparse, collections, json, os, re, subprocess, sys, time

import numpy as np

CLAUDE_BIN = os.environ.get('CLAUDE_CODE_EXECPATH', 'claude')
MEMBERS = ['fm_watchtime', 'fm_quantile', 'deepfm', 'finalmlp', 'lightgcn']

DATASET_FACTS = """Dataset facts (KuaiRand-Pure, computed directly from the data):
  - Catalog size: ~7,551 videos, ~27,077 users with both positive and negative train impressions.
  - Average interactions per video: ~190.8 -- a small, repeat-heavy catalog.
  - long_view positive rate: ~33-34% overall. is_click positive rate: ~46%.
  - play_time_ms >= 18000 alone matches long_view 96.7% of the time.
  - Evaluation is within-user reranking (GAUC + nDCG@5 per user's own impression group)."""

ARCH_REFERENCE = """Architecture reference -- what each member computes structurally (not how well
it performs alone -- that's for you to find out):
  fm_watchtime  : Factorization Machine (bilinear pairwise interaction over 7 fields) + BPR +
                  a watch-time censored-regression auxiliary task.
  fm_quantile   : same FM+BPR structure, but the watch-time auxiliary target is a duration-
                  conditioned empirical quantile instead of a raw log-scaled value.
  deepfm        : the same FM bilinear term, PLUS a parallel small DNN branch over the
                  concatenated field embeddings (z = z_FM + z_DNN) -- adds capacity for
                  nonlinear feature combinations FM's quadratic form cannot express.
  finalmlp      : no explicit bilinear interaction term at all -- two independently-gated
                  MLP streams over the same embeddings, fused via multi-head bilinear fusion.
  lightgcn      : propagates embeddings over the user-item bipartite interaction graph
                  (no feature transform, no nonlinearity -- pure neighborhood aggregation).
                  Only sees user_id/video_id, none of the other 5 fields.

General principle from ensemble learning (not specific to this data): an ensemble's gain over
its best single member depends on how UNCORRELATED the members' errors are, not just on how
individually accurate they are -- a weak-but-different member can be more valuable to combine
than a strong-but-redundant one. You have a `check_correlation` action to measure this directly
instead of assuming it."""

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "mechanism_basis": {"type": "string",
            "description": "What specific fact/result motivates this action -- a dataset fact, "
                            "an architecture-reference line, a prior iteration's score, or a "
                            "prior correlation measurement. Not a generic restatement."},
        "hypothesis": {"type": "string", "description": "One sentence: what you expect and why."},
        "action_type": {"type": "string", "enum": ["train_member", "check_correlation", "combine", "stop"]},
        "member": {"type": "string", "enum": MEMBERS,
                   "description": "Used when action_type=train_member. Ignored otherwise."},
        "member_a": {"type": "string", "enum": MEMBERS,
                     "description": "Used when action_type=check_correlation (must already be trained). Ignored otherwise."},
        "member_b": {"type": "string", "enum": MEMBERS,
                     "description": "Used when action_type=check_correlation, must differ from member_a. Ignored otherwise."},
    },
    "required": ["mechanism_basis", "hypothesis", "action_type", "member", "member_a", "member_b"],
}

SYSTEM_TASK_BRIEF = f"""You are the autonomous optimization loop for a recommendation ranking model
on KuaiRand-Pure. Task: rank each user's own logged impressions by predicted long_view; metric =
mean(GAUC, nDCG@5) on validation, called "primary". Official baseline primary is 0.5946.

Unlike a single-model search, your action space here includes TRAINING DIFFERENT ARCHITECTURES
and COMBINING them into an ensemble -- you are not limited to tuning one model's hyperparameters.

{DATASET_FACTS}

{ARCH_REFERENCE}

Each turn, choose ONE action:
  train_member       : train one of {MEMBERS} with its default settings and cache its scores.
  check_correlation   : measure how correlated two ALREADY-TRAINED members' predictions are on
                        validation (lower = more complementary = more likely to help if combined).
  combine             : run an exhaustive validation-selected fusion search over every member
                        you've trained so far, and see the resulting ensemble's validation score.
  stop                : only if you're confident further actions won't improve on your best score.

You'll see your full history below: every action taken, its result, and any correlation
measurements. A member only needs to be trained ONCE -- don't retrain a member you already have.
`combine` needs at least 2 trained members to do anything. Reason about what your own measurements
imply, not what you'd generically expect -- e.g. if two members are trained, checking their actual
correlation before deciding whether to invest in a third is more informative than guessing."""


def call_agent(log_text, model, max_budget_usd):
    prompt = SYSTEM_TASK_BRIEF + "\n\n--- YOUR LOG SO FAR ---\n" + (log_text or "(empty -- first iteration.)")
    prompt += "\n\n--- YOUR TASK ---\nChoose the next action, as JSON matching the schema."
    cmd = [CLAUDE_BIN, "-p", prompt, "--output-format", "json",
           "--permission-mode", "bypassPermissions", "--model", model,
           "--max-budget-usd", str(max_budget_usd), "--json-schema", json.dumps(ACTION_SCHEMA)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"agent call failed (exit {r.returncode}): {r.stderr[-2000:]}")
    envelope = json.loads(r.stdout)
    if envelope.get("is_error"):
        raise RuntimeError(f"agent call returned error: {envelope}")
    action = json.loads(envelope["result"])
    return action, envelope.get("usage", {}), envelope.get("total_cost_usd", 0.0)


def valid_primary_of(scores_dir, member, uva, yva):
    from evaluate import evaluate
    d = np.load(os.path.join(scores_dir, f"{member}.npz"))
    r = evaluate(uva, yva, d["valid"])
    return r, d


def to_rank(scores, groups):
    out = np.empty(len(scores))
    for idxs in groups:
        s = scores[idxs]
        order = np.argsort(s)
        rr = np.empty(len(s)); rr[order] = np.arange(len(s))
        out[idxs] = rr / max(len(s) - 1, 1)
    return out


def group_index(users):
    by = collections.defaultdict(list)
    for i, u in enumerate(users):
        by[u].append(i)
    return [np.array(v) for v in by.values()]


def run_train_member(member, scores_dir, data_dir, device):
    cmd = [sys.executable, "ablation_hetero_ensemble.py", "--member", member,
           "--scores_dir", scores_dir, "--data_dir", data_dir, "--device", device]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    dt = time.time() - t0
    if r.returncode != 0:
        return None, dt, f"exit {r.returncode}: {r.stderr[-1500:]}"
    return True, dt, None


def run_combine(scores_dir, data_dir):
    cmd = [sys.executable, "ablation_hetero_ensemble.py", "--combine",
           "--scores_dir", scores_dir, "--data_dir", data_dir]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    dt = time.time() - t0
    if r.returncode != 0:
        return None, dt, f"exit {r.returncode}: {r.stderr[-1500:]}"
    out = r.stdout
    mv = re.search(r"best on VALID: transform=(\w+), members=\(([^)]*)\), valid primary=([\d.]+)", out)
    mt = re.search(r"valid-selected ensemble\s+: GAUC ([\d.]+) \| nDCG@5 ([\d.]+) \| primary ([\d.]+)", out)
    if not mv or not mt:
        return None, dt, f"could not parse --combine output:\n{out[-1500:]}"
    result = {"transform": mv.group(1), "members": mv.group(2), "valid_primary": float(mv.group(3)),
              "test_GAUC": float(mt.group(1)), "test_nDCG5": float(mt.group(2)), "test_primary": float(mt.group(3))}
    return result, dt, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_iterations", type=int, default=15)
    ap.add_argument("--max_wallclock_s", type=int, default=3600)
    ap.add_argument("--epsilon", type=float, default=0.002)
    ap.add_argument("--patience_n", type=int, default=3)
    ap.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--max_budget_usd", type=float, default=0.30)
    ap.add_argument("--scores_dir", default="./agent_scores")
    ap.add_argument("--log_path", default="AGENT_LOG_ENSEMBLE.md")
    a = ap.parse_args()

    os.makedirs(a.scores_dir, exist_ok=True)
    from data import load, encode
    splits = load(a.data_dir)
    enc, _ = encode(splits)
    _, yva, uva = enc["valid"]

    interventions = 1
    t_start = time.time()
    trained = set()
    history = []
    best_valid = -1.0
    best_test = None
    total_cost, total_tok = 0.0, {"input": 0, "output": 0}

    log_lines = [f"# Agent Log (widened action space: architecture choice + ensembling)\n\n"
                 f"Automated run started at {time.strftime('%Y-%m-%d %H:%M:%S')}. "
                 f"Manual interventions so far: {interventions} (starting this process). "
                 f"Available members: {MEMBERS} (BST excluded -- too slow on CPU, see docstring).\n"]

    def flush():
        with open(a.log_path, "w") as fh:
            fh.write("\n".join(log_lines))
    flush()

    it, stop_reason = 0, None
    while it < a.max_iterations:
        if time.time() - t_start > a.max_wallclock_s:
            stop_reason = f"wall-clock cap reached ({a.max_wallclock_s}s)"
            break
        it += 1
        try:
            action, usage, cost = call_agent("\n".join(log_lines), a.model, a.max_budget_usd)
        except Exception as e:
            log_lines.append(f"\n## Iteration {it} -- AGENT CALL FAILED\n```\n{e}\n```\n")
            flush()
            continue
        total_cost += cost
        total_tok["input"] += usage.get("input_tokens", 0)
        total_tok["output"] += usage.get("output_tokens", 0)

        entry = [f"\n## Iteration {it}\n",
                 f"**Mechanism basis:** {action.get('mechanism_basis','')}\n",
                 f"**Hypothesis:** {action.get('hypothesis','')}\n",
                 f"**Action:** {action['action_type']}"]

        at = action["action_type"]
        if at == "stop":
            stop_reason = "agent self-reported no further improvement expected"
            entry.append("\n")
            log_lines.extend(entry); flush()
            break

        elif at == "train_member":
            member = action["member"]
            entry[-1] += f" (member={member})\n"
            if member in trained:
                entry.append("**Result:** SKIPPED -- already trained this iteration budget; no wasted compute.\n")
                history.append({"iter": it, "type": "skip"})
            else:
                ok, dt, err = run_train_member(member, a.scores_dir, a.data_dir, a.device)
                if err:
                    entry.append(f"**Result:** FAILED after {dt:.1f}s\n```\n{err}\n```\n")
                    history.append({"iter": it, "type": "train_fail"})
                else:
                    trained.add(member)
                    r, _ = valid_primary_of(a.scores_dir, member, uva, yva)
                    is_best = r["primary"] > best_valid
                    if is_best:
                        best_valid = r["primary"]
                        best_test = {"kind": "single", "member": member}
                    entry.append(f"**Result:** {member} valid GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | "
                                 f"primary {r['primary']:.4f}{'  <- new best' if is_best else ''} [{dt:.1f}s]\n")
                    history.append({"iter": it, "type": "train", "valid_primary": r["primary"]})

        elif at == "check_correlation":
            ma, mb = action["member_a"], action["member_b"]
            entry[-1] += f" ({ma} vs {mb})\n"
            if ma == mb or ma not in trained or mb not in trained:
                entry.append(f"**Result:** SKIPPED -- both members must be already trained and distinct "
                             f"(trained so far: {sorted(trained)}).\n")
                history.append({"iter": it, "type": "skip"})
            else:
                da = np.load(os.path.join(a.scores_dir, f"{ma}.npz"))["valid"]
                db = np.load(os.path.join(a.scores_dir, f"{mb}.npz"))["valid"]
                groups = group_index(uva)
                ra, rb = to_rank(da, groups), to_rank(db, groups)
                corr = float(np.corrcoef(ra, rb)[0, 1])
                entry.append(f"**Result:** rank correlation({ma}, {mb}) on VALID = {corr:.4f}\n")
                history.append({"iter": it, "type": "corr"})

        elif at == "combine":
            if len(trained) < 2:
                entry.append(f"**Result:** SKIPPED -- need >=2 trained members, have {sorted(trained)}.\n")
                history.append({"iter": it, "type": "skip"})
            else:
                result, dt, err = run_combine(a.scores_dir, a.data_dir)
                if err:
                    entry.append(f"**Result:** FAILED after {dt:.1f}s\n```\n{err}\n```\n")
                    history.append({"iter": it, "type": "combine_fail"})
                else:
                    is_best = result["valid_primary"] > best_valid
                    if is_best:
                        best_valid = result["valid_primary"]
                        best_test = {"kind": "ensemble", "transform": result["transform"], "members": result["members"]}
                    entry.append(f"**Result:** ensemble ({result['transform']}, members={result['members']}) "
                                 f"valid primary {result['valid_primary']:.4f}"
                                 f"{'  <- new best' if is_best else ''} "
                                 f"(test primary {result['test_primary']:.4f}) [{dt:.1f}s]\n")
                    history.append({"iter": it, "type": "combine", "valid_primary": result["valid_primary"]})

        log_lines.extend(entry)
        flush()

        scored = [h for h in history if "valid_primary" in h]
        if len(scored) > a.patience_n:
            running_best, b = [], -1.0
            for h in scored:
                b = max(b, h["valid_primary"]); running_best.append(b)
            if running_best[-1] - running_best[-1 - a.patience_n] <= a.epsilon:
                stop_reason = f"converged: best valid primary improved <= eps={a.epsilon} over last N={a.patience_n} scored iterations"
                break

    if stop_reason is None:
        stop_reason = f"iteration cap reached ({a.max_iterations})"

    summary = ["\n---\n## Run summary\n",
               f"- Iterations run: {it}\n", f"- Stop reason: {stop_reason}\n",
               f"- Wall-clock: {time.time()-t_start:.0f}s\n",
               f"- Members trained: {sorted(trained)}\n",
               f"- Agent LLM calls: cost ${total_cost:.4f}, input tokens {total_tok['input']}, output tokens {total_tok['output']}\n",
               f"- Manual interventions: {interventions}\n",
               f"- Best valid primary found: {best_valid:.4f} ({best_test})\n"]
    log_lines.extend(summary)
    flush()
    print("\n".join(summary))


if __name__ == "__main__":
    main()
