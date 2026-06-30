## GitHub Copilot subscription backends

graphify can use your GitHub Copilot subscription for semantic extraction — no separate API key purchase required.

**`copilot-cli`** — shells out to the standalone `copilot` CLI binary, injecting graphify's extraction prompt via `COPILOT_CUSTOM_INSTRUCTIONS_DIRS`. Auth is handled by the CLI itself:

```bash
copilot auth login                      # one-time authentication
graphify extract . --backend copilot-cli
```

Override the model with `GRAPHIFY_COPILOT_CLI_MODEL` (passed as `COPILOT_MODEL` to the subprocess). Set `GRAPHIFY_COPILOT_CLI_PARALLEL=1` to allow concurrent extraction (serial by default to avoid rate-limit contention).

**`copilot`** — authenticates via `GITHUB_TOKEN` against the GitHub Models API (requires a PAT with models permission or an org-provisioned token):

```bash
export GITHUB_TOKEN=ghp_...
graphify extract . --backend copilot
```

When no Gemini key is available, check for these before falling back to manual extraction:

```python
import graphify.llm as llm, os, shutil
backend = llm.detect_backend()          # auto-detects Gemini/OpenAI/etc. from env vars
if backend is None:
    if os.environ.get("GITHUB_TOKEN"):
        backend = "copilot"
    elif shutil.which("copilot"):
        backend = "copilot-cli"
# pass backend= to extract_corpus_parallel / label_communities as needed
```

Neither `copilot` nor `copilot-cli` is returned by `detect_backend()` — always pass `--backend` explicitly (avoids silent routing through GitHub Actions' ambient `GITHUB_TOKEN`).

---
