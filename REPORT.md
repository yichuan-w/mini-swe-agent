# CS294 HW1 – Technical Report (Minimal ReAct SWE Agent)

## Overview
We implemented a minimal ReAct agent for SWE-Bench with a single primary tool (shell) and a semantic code search helper. The agent reasons briefly, then acts via exactly one tool call per turn, iterating until changes are staged and tests pass.

- Codebase adapted from mini-swe-agent (Princeton/Stanford) [`https://github.com/SWE-agent/mini-swe-agent/tree/main`](https://github.com/SWE-agent/mini-swe-agent/tree/main)
- Single-tool-first design: `run_bash_cmd` suffices for test, view, edit, and stage
- Added `leann_search` semantic search to improve localization

## Dataset & Metric
- SWE-Bench verified subset: `lynnliu030/swebench-eval-subset`
- Max steps: 100
- Metric: resolved instances (patch applies + tests pass)

## Results
- Latest run: 8/20 resolved (40%), 20 completed, 0 errors
- Stronger prompts + semantic search reduced empty patches and improved localization

## System Design
- Message tree: system → user → instructor → linear steps
- Textual function-call protocol parsed via `rfind`
- One tool call per turn; finish only when staged changes exist
- Patch generation returns a clean unified diff from `git diff --cached`

## Custom Tools
- `run_bash_cmd(command: str)`
  - Single universal interface to the sandbox (tests, viewing via `nl -ba`/`sed -n`, edits via `ed`/Python here-doc, staging via `git add -A`).
  - Rationale: reduces tool-call errors; covers all needed ops non-interactively.

- `leann_search(index_name: str, query: str, k: int = 5)`
  - Semantic code search over a prebuilt index (install/build done in setup).
  - Rationale: grep/string search is brittle; semantic retrieval finds concept-level matches (APIs/behaviors/responsibilities) across files.

## Prompting Strategy
- Strict workflow: pytest → grep/nl → edit (ed/Python) → pytest → stage+verify → finish.
- Exactly one function call per message; never finish with no staged files.
- Use `leann_search` when errors reference symbols/behaviors not found via grep or when we need semantically similar implementations.

One-shot examples included for tests, viewing ranges, editing via `ed`, and using `leann_search` on failure contexts.

## Lessons Learned
- Single tool is enough: a shell bridge covers testing, viewing, editing, staging.
- Do edits/views only via cmd: `nl -ba`, `sed -n`, `ed`, and Python here-docs are robust.
- Semantic search is effective for localization when grep fails.
- Protocol matters: strict end token and one-call outputs reduce parser failures.
- Submission hygiene: only clean unified diffs; no prose in patch output.
- Prompt strength is pivotal: more forceful instructions reduce empty patches and premature finishes.

## Limitations
- Semantic search quality depends on the index and query phrasing.


## Future Work
- Test-time scaling: parallel sample multiple trajectories; pick with a verifier.
- Verifier/critic loop: compile/test feedback to accept or repair patches before finish.
- Better localization: hybrid grep + semantic search with reranking by failure signals.

## Reproduction (high-level)
1. `uv pip install -r requirements.txt`
2. `python run_agent.py --model gpt-5-mini --max-steps 100 -o results`
3. `python -m swebench.harness.run_evaluation --dataset_name lynnliu030/swebench-eval-subset --predictions_path ./results/preds.json --max_workers 8 --run_id my_eval`

---
Key takeaway: a single robust cmd tool + semantic search for localization + strict prompting yields consistent, clean patches and improved solve rates. More gains likely from test-time scaling and verification.
