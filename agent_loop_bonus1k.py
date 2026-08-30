"""Autonomous agent pointed at the KuaiRand-1K bonus benchmark (same task/metrics as
Pure, per the competition spec). Same design discipline as `agent_loop.py`: grounded
dataset facts (this time 1K-specific), a narrow but real action space over
`bonus_fm_torch.py`'s existing CLI (--k, --lr, --epochs), and a code-enforced
convergence rule.

KuaiRand-27K is deliberately NOT attempted here. It requires the university SLURM GPU
cluster (established in RUN_LOG.md: dense numpy Adam is computationally infeasible at
27K's ~41M-parameter embedding table regardless of hardware, and the sparse-embedding
fix still needs a GPU to run in reasonable time). Running it would mean this loop
periodically SSH-ing into a remote cluster, submitting a job, and polling for
completion mid-run -- each of those steps has previously needed a human to unblock
(credential setup, disk-quota troubleshooting, SLURM memory defaults) and doing it
unattended would risk silently reintroducing exactly the manual intervention this
exercise is trying to measure the absence of. Honest scope limit, not an oversight.
"""
import argparse, json, os, re, subprocess, sys, time

CLAUDE_BIN = os.environ.get('CLAUDE_CODE_EXECPATH', 'claude')

DATASET_FACTS_1K = """Dataset facts (KuaiRand-1K, computed directly from the data -- this is a
DIFFERENT dataset from KuaiRand-Pure, same task/metrics but very different structure):
  - Catalog size: ~4,371,868 videos (vs Pure's ~7,551) -- a huge, sparse catalog.
  - 11,713,045 total interactions -- average ~2.7 interactions per video (vs Pure's ~190.8).
  - Only 1,000 users total, of which 983 have both positive and negative train impressions.
  - Test users average ~2,525 impressions each (vs Pure's ~5) -- 1K keeps each user's entire
    log rather than a filtered candidate pool.
  - Evaluation is within-user reranking (GAUC + nDCG@5 per user's own impression group),
    identical protocol to Pure."""

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "mechanism_basis": {"type": "string",
            "description": "Which specific dataset fact above motivates this choice, and/or "
                            "which prior iteration's result you're responding to. Not a generic "
                            "restatement of what a hyperparameter does."},
        "hypothesis": {"type": "string", "description": "One sentence: what you expect and why."},
        "k": {"type": "integer", "minimum": 4, "maximum": 128},
        "lr": {"type": "number", "minimum": 0.0001, "maximum": 0.1},
        "epochs": {"type": "integer", "minimum": 1, "maximum": 40},
        "stop_early": {"type": "boolean"},
    },
    "required": ["mechanism_basis", "hypothesis", "k", "lr", "epochs", "stop_early"],
}

SYSTEM_TASK_BRIEF = f"""You are the autonomous optimization loop for a sparse-embedding
Factorization Machine (SparseFM) on the KuaiRand-1K bonus benchmark. Task: rank each user's own
logged impressions by predicted long_view; metric = mean(GAUC, nDCG@5) on validation, "primary".
There is no published official baseline for this bonus dataset (only Pure has one) -- your job is
simply to find the best validation primary you can within this action space.

{DATASET_FACTS_1K}

Parameters you control: k (embedding dim), lr (learning rate), epochs (max training epochs --
training already early-stops on validation internally, but a lower epochs cap changes how much
overfitting the model is allowed to reach before that stopping kicks in).

You'll see your full history below. Reason about what the extreme sparsity here (~2.7
interactions/video) implies for embedding dimension and training duration -- this is a very
different regime from a dense catalog, not just "the same problem, more data." Set
stop_early=true only if you're confident the space is exhausted."""


def call_agent(log_text, model, max_budget_usd):
    prompt = SYSTEM_TASK_BRIEF + "\n\n--- YOUR LOG SO FAR ---\n" + (log_text or "(empty -- first iteration.)")
    prompt += "\n\n--- YOUR TASK ---\nPropose the next configuration, as JSON matching the schema."
    cmd = [CLAUDE_BIN, "-p", prompt, "--output-format", "json",
           "--permission-mode", "bypassPermissions", "--model", model,
           "--max-budget-usd", str(max_budget_usd), "--json-schema", json.dumps(ACTION_SCHEMA)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"agent call failed (exit {r.returncode}): {r.stderr[-2000:]}")
    envelope = json.loads(r.stdout)
    if envelope.get("is_error"):
        raise RuntimeError(f"agent call returned error: {envelope}")
    return json.loads(envelope["result"]), envelope.get("usage", {}), envelope.get("total_cost_usd", 0.0)


def run_bonus(action, seed, data_dir, device):
    cmd = [sys.executable, "bonus_fm_torch.py", "--suffix", "1k", "--data_dir", data_dir,
           "--seed", str(seed), "--device", device,
           "--k", str(action["k"]), "--lr", str(action["lr"]), "--epochs", str(action["epochs"])]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    dt = time.time() - t0
    if r.returncode != 0:
        return None, dt, f"exit {r.returncode}: {r.stderr[-1500:]}", cmd
    out = r.stdout
    m = {}
    for split in ("valid", "test"):
        mo = re.search(rf"{split}\s+GAUC ([\d.]+) \| nDCG@5 ([\d.]+) \| primary ([\d.]+)", out)
        if not mo:
            return None, dt, f"could not parse '{split}' metrics:\n{out[-1500:]}", cmd
        m[split] = {"GAUC": float(mo.group(1)), "nDCG@5": float(mo.group(2)), "primary": float(mo.group(3))}
    return m, dt, None, cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_iterations", type=int, default=10)
    ap.add_argument("--max_wallclock_s", type=int, default=1800)
    ap.add_argument("--epsilon", type=float, default=0.002)
    ap.add_argument("--patience_n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_dir", default="./KuaiRand-1K/data")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--max_budget_usd", type=float, default=0.30)
    ap.add_argument("--log_path", default="AGENT_LOG_1K.md")
    a = ap.parse_args()

    interventions = 1
    t_start = time.time()
    history = []
    total_cost, total_tok = 0.0, {"input": 0, "output": 0}
    log_lines = [f"# Agent Log (KuaiRand-1K bonus benchmark)\n\n"
                 f"Automated run started at {time.strftime('%Y-%m-%d %H:%M:%S')}. "
                 f"Manual interventions so far: {interventions} (starting this process).\n"]

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
            flush(); continue
        total_cost += cost
        total_tok["input"] += usage.get("input_tokens", 0)
        total_tok["output"] += usage.get("output_tokens", 0)

        if action.get("stop_early"):
            stop_reason = "agent self-reported the action space exhausted"
            log_lines.append(f"\n## Iteration {it} -- agent requested early stop\n"
                              f"**Mechanism basis:** {action.get('mechanism_basis','')}\n")
            flush(); break

        metrics, dt, err, cmd = run_bonus(action, a.seed, a.data_dir, a.device)
        entry = [f"\n## Iteration {it}\n",
                 f"**Mechanism basis:** {action.get('mechanism_basis','')}\n",
                 f"**Hypothesis:** {action.get('hypothesis','')}\n",
                 f"**Command:** `{' '.join(cmd[1:])}`\n"]
        if err:
            entry.append(f"**Result:** FAILED after {dt:.1f}s\n```\n{err}\n```\n")
            history.append({"iter": it, "failed": True})
        else:
            vp = metrics["valid"]["primary"]
            best_so_far = max([h["valid_primary"] for h in history if not h["failed"]], default=-1.0)
            is_best = vp > best_so_far
            entry.append(f"**Result:** valid GAUC {metrics['valid']['GAUC']:.4f} | "
                         f"nDCG@5 {metrics['valid']['nDCG@5']:.4f} | primary {vp:.4f}"
                         f"{'  <- new best' if is_best else ''} "
                         f"(test primary {metrics['test']['primary']:.4f}) [{dt:.1f}s]\n")
            history.append({"iter": it, "action": action, "valid_primary": vp,
                            "test_primary": metrics["test"]["primary"], "failed": False})
        log_lines.extend(entry)
        flush()

        successes = [h for h in history if not h["failed"]]
        if len(successes) > a.patience_n:
            running_best, b = [], -1.0
            for h in successes:
                b = max(b, h["valid_primary"]); running_best.append(b)
            if running_best[-1] - running_best[-1 - a.patience_n] <= a.epsilon:
                stop_reason = f"converged: best valid primary improved <= eps={a.epsilon} over last N={a.patience_n} successful iterations"
                break

    if stop_reason is None:
        stop_reason = f"iteration cap reached ({a.max_iterations})"

    successes = [h for h in history if not h["failed"]]
    summary = ["\n---\n## Run summary\n", f"- Iterations run: {it}\n", f"- Stop reason: {stop_reason}\n",
               f"- Wall-clock: {time.time()-t_start:.0f}s\n",
               f"- Agent LLM calls: cost ${total_cost:.4f}, input tokens {total_tok['input']}, output tokens {total_tok['output']}\n",
               f"- Manual interventions: {interventions}\n"]
    if successes:
        best = max(successes, key=lambda h: h["valid_primary"])
        summary.append(f"- Best config found (iteration {best['iter']}): `{best['action']}`\n")
        summary.append(f"  valid primary {best['valid_primary']:.4f}, test primary {best['test_primary']:.4f}\n")
        summary.append("  (No official baseline exists for KuaiRand-1K -- this is an absolute score, not a delta.)\n")
    else:
        summary.append("- No successful iterations.\n")
    log_lines.extend(summary)
    flush()
    print("\n".join(summary))


if __name__ == "__main__":
    main()
