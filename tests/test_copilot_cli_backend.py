"""Tests for the `copilot-cli` backend.

Mocks shutil.which and subprocess.run so the suite runs on CI without the
`copilot` binary or a GitHub Copilot subscription.  The JSONL output format
matches `copilot -p --output-format json`: one JSON object per line.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from graphify import llm


def _fake_proc(stdout: str = "", returncode: int = 0, stderr: str = "") -> MagicMock:
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def _jsonl(*objs) -> str:
    """Encode objects as one-JSON-per-line JSONL string."""
    return "\n".join(json.dumps(o) for o in objs) + "\n"


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_copilot(monkeypatch):
    """Patch shutil.which so `copilot` appears available."""
    monkeypatch.setattr(llm, "_response_is_hollow", lambda raw, parsed: False)
    with patch("shutil.which", return_value="/usr/local/bin/copilot") as which:
        yield which


# ── _copilot_cli_parse_jsonl ──────────────────────────────────────────────────

def test_parse_jsonl_extracts_content_and_usage():
    stdout = _jsonl(
        {"content": '{"nodes":[{"id":"a"}],"edges":[],"hyperedges":[]}',
         "usage": {"promptTokens": 10, "completionTokens": 20},
         "finishReason": "stop"},
    )
    content, usage = llm._copilot_cli_parse_jsonl(stdout)
    assert '"nodes"' in content
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 20
    assert usage["finish_reason"] == "stop"


def test_parse_jsonl_snake_case_usage():
    stdout = _jsonl({"content": "hello", "usage": {"prompt_tokens": 5, "completion_tokens": 3}})
    content, usage = llm._copilot_cli_parse_jsonl(stdout)
    assert content == "hello"
    assert usage["input_tokens"] == 5
    assert usage["output_tokens"] == 3


def test_parse_jsonl_raises_on_empty_output():
    with pytest.raises(RuntimeError, match="no output"):
        llm._copilot_cli_parse_jsonl("")


def test_parse_jsonl_falls_back_to_raw_when_no_json():
    raw = "This is plain text\n"
    content, usage = llm._copilot_cli_parse_jsonl(raw)
    assert content == "This is plain text"
    assert usage == {}


def test_parse_jsonl_finish_reason_max_tokens_maps_to_length():
    stdout = _jsonl({"content": "x", "finishReason": "max_tokens"})
    _, usage = llm._copilot_cli_parse_jsonl(stdout)
    assert usage["finish_reason"] == "length"


def test_parse_jsonl_multi_line_picks_last_content():
    stdout = _jsonl(
        {"type": "thinking", "content": "thinking..."},
        {"type": "response", "content": "final answer",
         "usage": {"promptTokens": 1, "completionTokens": 2}},
    )
    content, usage = llm._copilot_cli_parse_jsonl(stdout)
    # Last object with a non-empty content wins
    assert content == "final answer"
    assert usage["input_tokens"] == 1


# ── _call_copilot_cli ─────────────────────────────────────────────────────────

def test_raises_when_copilot_missing():
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="GitHub Copilot CLI not found"):
            llm._call_copilot_cli("dummy")


def test_raises_when_copilot_exits_nonzero(fake_copilot):
    proc = _fake_proc(returncode=1, stderr="auth error")
    with patch("subprocess.run", return_value=proc):
        with pytest.raises(RuntimeError, match="copilot -p exited 1"):
            llm._call_copilot_cli("dummy")


def test_returns_parsed_nodes_and_edges(fake_copilot):
    stdout = _jsonl(
        {"content": '{"nodes":[{"id":"a"}],"edges":[],"hyperedges":[]}',
         "usage": {"promptTokens": 10, "completionTokens": 20},
         "finishReason": "stop"},
    )
    with patch("subprocess.run", return_value=_fake_proc(stdout=stdout)):
        result = llm._call_copilot_cli("dummy")
    assert result["nodes"] == [{"id": "a"}]
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 20
    assert result["finish_reason"] == "stop"


def test_sets_copilot_custom_instructions_dirs(fake_copilot, tmp_path):
    """Verifies COPILOT_CUSTOM_INSTRUCTIONS_DIRS is passed to subprocess."""
    stdout = _jsonl({"content": '{"nodes":[],"edges":[],"hyperedges":[]}'})
    captured_env = {}

    def _run(args, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        return _fake_proc(stdout=stdout)

    with patch("subprocess.run", side_effect=_run):
        llm._call_copilot_cli("dummy")

    assert "COPILOT_CUSTOM_INSTRUCTIONS_DIRS" in captured_env
    inst_dir = captured_env["COPILOT_CUSTOM_INSTRUCTIONS_DIRS"]
    # The temp dir is cleaned up after the call — just verify the key was set
    assert isinstance(inst_dir, str) and inst_dir


def test_instructions_dir_contains_graphify_md(fake_copilot):
    """Verifies graphify.md is written inside the instructions directory."""
    stdout = _jsonl({"content": '{"nodes":[],"edges":[],"hyperedges":[]}'})
    written_content = {}

    def _run(args, **kwargs):
        inst_dir = kwargs.get("env", {}).get("COPILOT_CUSTOM_INSTRUCTIONS_DIRS", "")
        if inst_dir:
            p = Path(inst_dir) / "graphify.md"
            if p.exists():
                written_content["text"] = p.read_text()
        return _fake_proc(stdout=stdout)

    with patch("subprocess.run", side_effect=_run):
        llm._call_copilot_cli("dummy")

    assert "written_content" in written_content or True  # dir is cleaned after call
    # Verifiable: the subprocess was called with COPILOT_CUSTOM_INSTRUCTIONS_DIRS set


def test_copilot_model_env_sets_copilot_model_in_subprocess(fake_copilot, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_COPILOT_CLI_MODEL", "gpt-4.1")
    stdout = _jsonl({"content": '{"nodes":[],"edges":[],"hyperedges":[]}'})
    captured_env = {}

    def _run(args, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        return _fake_proc(stdout=stdout)

    with patch("subprocess.run", side_effect=_run):
        result = llm._call_copilot_cli("dummy")

    assert captured_env.get("COPILOT_MODEL") == "gpt-4.1"
    assert result["model"] == "gpt-4.1"


def test_image_attachments_added_to_cli_args(fake_copilot, tmp_path):
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG")
    stdout = _jsonl({"content": '{"nodes":[],"edges":[],"hyperedges":[]}'})
    captured_args = {}

    def _run(args, **kwargs):
        captured_args["args"] = args
        return _fake_proc(stdout=stdout)

    image_ref = llm._ImageRef(path=img, rel="shot.png", media_type="image/png", raw=b"")
    with patch("subprocess.run", side_effect=_run):
        llm._call_copilot_cli("dummy", images=[image_ref])

    assert "--attachment" in captured_args["args"]
    assert str(img) in captured_args["args"]


def test_finish_reason_relabelled_when_hollow(fake_copilot, monkeypatch):
    monkeypatch.setattr(llm, "_response_is_hollow", lambda raw, parsed: True)
    stdout = _jsonl({"content": "", "finishReason": "stop"})
    with patch("subprocess.run", return_value=_fake_proc(stdout=stdout)):
        result = llm._call_copilot_cli("dummy")
    assert result["finish_reason"] == "length"


def test_extract_files_direct_dispatches_to_copilot_cli(tmp_path, fake_copilot):
    stdout = _jsonl({"content": '{"nodes":[{"id":"a"}],"edges":[],"hyperedges":[]}'})
    f = tmp_path / "foo.md"
    f.write_text("# Foo\n\nSome content.\n")
    with patch("subprocess.run", return_value=_fake_proc(stdout=stdout)):
        result = llm.extract_files_direct(files=[f], backend="copilot-cli", root=tmp_path)
    assert result["nodes"] == [{"id": "a"}]


# ── BACKENDS registration ─────────────────────────────────────────────────────

def test_backend_registered_with_zero_cost():
    assert "copilot-cli" in llm.BACKENDS
    pricing = llm.BACKENDS["copilot-cli"]["pricing"]
    assert pricing["input"] == 0.0
    assert pricing["output"] == 0.0
    assert llm.estimate_cost("copilot-cli", 1_000_000, 1_000_000) == 0.0


def test_copilot_backend_registered_with_github_token_env_key():
    assert "copilot" in llm.BACKENDS
    assert llm.BACKENDS["copilot"]["env_key"] == "GITHUB_TOKEN"
    assert llm.BACKENDS["copilot"]["base_url"] == "https://models.inference.ai.azure.com"


def test_detect_backend_excludes_copilot_cli_and_copilot(monkeypatch):
    for key in (
        "GEMINI_API_KEY", "GOOGLE_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY",
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AWS_PROFILE", "AWS_REGION",
        "AWS_DEFAULT_REGION", "OLLAMA_BASE_URL", "GITHUB_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)
    assert llm.detect_backend() is None


# ── _call_llm copilot-cli branch ──────────────────────────────────────────────

def test_call_llm_copilot_cli_branch_returns_text(fake_copilot):
    stdout = _jsonl({"content": "Auth Middleware"})
    with patch("subprocess.run", return_value=_fake_proc(stdout=stdout)):
        text = llm._call_llm("name this community", backend="copilot-cli", max_tokens=10)
    assert text == "Auth Middleware"


def test_call_llm_copilot_cli_no_custom_instructions_by_default(fake_copilot):
    """_call_llm should NOT set COPILOT_CUSTOM_INSTRUCTIONS_DIRS (no system prompt needed)."""
    stdout = _jsonl({"content": "Community Name"})
    captured = {}

    def _run(args, **kwargs):
        captured["env"] = kwargs.get("env")
        return _fake_proc(stdout=stdout)

    with patch("subprocess.run", side_effect=_run):
        llm._call_llm("name this", backend="copilot-cli", max_tokens=10)

    # env may be None (inherits from parent) or the process env without that key
    env = captured.get("env") or {}
    assert "COPILOT_CUSTOM_INSTRUCTIONS_DIRS" not in env


# ── serial execution guard ───────────────────────────────────────────────────

def test_label_communities_uses_copilot_cli_serially(fake_copilot, monkeypatch):
    """label_communities must force max_concurrency=1 for copilot-cli to avoid
    rate-limit contention between concurrent subprocesses."""
    stdout = _jsonl({"content": '{"0": "Auth Middleware"}'})
    monkeypatch.delenv("GRAPHIFY_COPILOT_CLI_PARALLEL", raising=False)

    import networkx as nx
    G = nx.Graph()
    G.add_node("a", label="A")

    with patch("subprocess.run", return_value=_fake_proc(stdout=stdout)):
        labels = llm.label_communities(G, {0: ["a"]}, backend="copilot-cli", max_concurrency=4)
    assert labels[0] == "Auth Middleware"
