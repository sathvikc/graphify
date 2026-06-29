"""Tests for the `copilot-cli` backend.

Mocks shutil.which, subprocess.run (`gh auth token`), urllib (Copilot token
exchange), and the `openai` module (Copilot chat completion) so the suite
runs on CI without the `gh` binary, a GitHub Copilot subscription, or a live
network call.
"""
from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from graphify import llm


def _fake_openai_response(content, *, finish_reason="stop", prompt_tokens=10, completion_tokens=20):
    class _Usage:
        def __init__(self):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    class _Message:
        def __init__(self):
            self.content = content

    class _Choice:
        def __init__(self):
            self.message = _Message()
            self.finish_reason = finish_reason

    class _Resp:
        def __init__(self):
            self.choices = [_Choice()]
            self.usage = _Usage()

    return _Resp()


def _install_fake_openai(monkeypatch, fake_resp):
    captured = {}

    class _FakeOpenAI:
        def __init__(self, *_, **kwargs):
            captured["init_kwargs"] = kwargs
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            captured["create_kwargs"] = kwargs
            return fake_resp

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return captured


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_gh_and_token_exchange(monkeypatch):
    gh_proc = MagicMock(returncode=0, stdout="gho_faketoken123\n", stderr="")
    monkeypatch.setattr(llm, "_response_is_hollow", lambda raw, parsed: False)
    with patch("shutil.which", return_value="/usr/local/bin/gh"), \
         patch("subprocess.run", return_value=gh_proc) as run, \
         patch(
             "urllib.request.urlopen",
             return_value=_FakeHTTPResponse({"token": "copilot-session-token"}),
         ) as urlopen:
        yield run, urlopen


def test_returns_parsed_nodes_and_edges(fake_gh_and_token_exchange, monkeypatch):
    fake_resp = _fake_openai_response(
        json.dumps({"nodes": [{"id": "a"}], "edges": [], "hyperedges": []})
    )
    _install_fake_openai(monkeypatch, fake_resp)

    result = llm._call_copilot_cli("dummy", max_tokens=8192)
    assert result["nodes"] == [{"id": "a"}]
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 20
    assert result["finish_reason"] == "stop"


def test_uses_copilot_session_token_as_api_key(fake_gh_and_token_exchange, monkeypatch):
    fake_resp = _fake_openai_response('{"nodes":[],"edges":[],"hyperedges":[]}')
    captured = _install_fake_openai(monkeypatch, fake_resp)

    llm._call_copilot_cli("dummy", max_tokens=8192)
    assert captured["init_kwargs"]["api_key"] == "copilot-session-token"
    assert captured["init_kwargs"]["base_url"] == "https://api.githubcopilot.com"


def test_gh_auth_token_invoked(fake_gh_and_token_exchange, monkeypatch):
    fake_resp = _fake_openai_response('{"nodes":[],"edges":[],"hyperedges":[]}')
    _install_fake_openai(monkeypatch, fake_resp)

    run, _ = fake_gh_and_token_exchange
    llm._call_copilot_cli("dummy", max_tokens=8192)
    assert run.call_args.args[0] == ["gh", "auth", "token"]


def test_raises_when_gh_missing():
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="GitHub CLI \\(gh\\) not found"):
            llm._call_copilot_cli("dummy", max_tokens=8192)


def test_raises_when_gh_auth_token_fails():
    gh_proc = MagicMock(returncode=1, stdout="", stderr="not logged in")
    with patch("shutil.which", return_value="/usr/local/bin/gh"), \
         patch("subprocess.run", return_value=gh_proc):
        with pytest.raises(RuntimeError, match="gh auth token failed"):
            llm._call_copilot_cli("dummy", max_tokens=8192)


def test_raises_on_empty_gh_token():
    gh_proc = MagicMock(returncode=0, stdout="  \n", stderr="")
    with patch("shutil.which", return_value="/usr/local/bin/gh"), \
         patch("subprocess.run", return_value=gh_proc):
        with pytest.raises(RuntimeError, match="empty token"):
            llm._call_copilot_cli("dummy", max_tokens=8192)


def test_raises_on_token_exchange_401(monkeypatch):
    import urllib.error
    gh_proc = MagicMock(returncode=0, stdout="gho_faketoken123\n", stderr="")

    def raise_401(*_a, **_kw):
        raise urllib.error.HTTPError(
            "https://api.github.com/copilot_internal/v2/token", 401, "Unauthorized",
            {}, None,
        )

    with patch("shutil.which", return_value="/usr/local/bin/gh"), \
         patch("subprocess.run", return_value=gh_proc), \
         patch("urllib.request.urlopen", side_effect=raise_401):
        with pytest.raises(RuntimeError, match="401 Unauthorized"):
            llm._call_copilot_cli("dummy", max_tokens=8192)


def test_raises_on_token_exchange_403(monkeypatch):
    import urllib.error
    gh_proc = MagicMock(returncode=0, stdout="gho_faketoken123\n", stderr="")

    def raise_403(*_a, **_kw):
        raise urllib.error.HTTPError(
            "https://api.github.com/copilot_internal/v2/token", 403, "Forbidden",
            {}, None,
        )

    with patch("shutil.which", return_value="/usr/local/bin/gh"), \
         patch("subprocess.run", return_value=gh_proc), \
         patch("urllib.request.urlopen", side_effect=raise_403):
        with pytest.raises(RuntimeError, match="403 Forbidden"):
            llm._call_copilot_cli("dummy", max_tokens=8192)


def test_raises_on_empty_token_field(fake_gh_and_token_exchange, monkeypatch):
    with patch(
        "urllib.request.urlopen",
        return_value=_FakeHTTPResponse({"token": ""}),
    ):
        with pytest.raises(RuntimeError, match="no token field"):
            llm._call_copilot_cli("dummy", max_tokens=8192)


def test_finish_reason_relabelled_when_hollow(fake_gh_and_token_exchange, monkeypatch):
    monkeypatch.setattr(llm, "_response_is_hollow", lambda raw, parsed: True)
    fake_resp = _fake_openai_response("", finish_reason="stop", completion_tokens=0)
    _install_fake_openai(monkeypatch, fake_resp)

    result = llm._call_copilot_cli("dummy", max_tokens=8192)
    assert result["finish_reason"] == "length"


def test_extract_files_direct_dispatches_to_copilot_cli(tmp_path, fake_gh_and_token_exchange, monkeypatch):
    fake_resp = _fake_openai_response(
        json.dumps({"nodes": [{"id": "a"}], "edges": [], "hyperedges": []})
    )
    _install_fake_openai(monkeypatch, fake_resp)

    f = tmp_path / "foo.md"
    f.write_text("# Foo\n\nSome content.\n")
    result = llm.extract_files_direct(files=[f], backend="copilot-cli", root=tmp_path)
    assert result["nodes"] == [{"id": "a"}]


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


def test_detect_backend_excludes_claude_cli_and_copilot_cli(monkeypatch):
    for key in (
        "GEMINI_API_KEY", "GOOGLE_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY",
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AWS_PROFILE", "AWS_REGION",
        "AWS_DEFAULT_REGION", "OLLAMA_BASE_URL", "GITHUB_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)
    assert llm.detect_backend() is None


def test_call_copilot_cli_honours_timeout(fake_gh_and_token_exchange, monkeypatch):
    fake_resp = _fake_openai_response('{"nodes":[],"edges":[],"hyperedges":[]}')
    captured = _install_fake_openai(monkeypatch, fake_resp)

    monkeypatch.setenv("GRAPHIFY_API_TIMEOUT", "30")
    llm._call_copilot_cli("dummy", max_tokens=8192)
    assert captured["init_kwargs"]["timeout"] == 30.0


def test_call_llm_copilot_cli_branch_returns_text(fake_gh_and_token_exchange, monkeypatch):
    fake_resp = _fake_openai_response("Auth Middleware")
    _install_fake_openai(monkeypatch, fake_resp)

    text = llm._call_llm("name this community", backend="copilot-cli", max_tokens=10)
    assert text == "Auth Middleware"


def test_label_communities_uses_copilot_cli_serially(fake_gh_and_token_exchange, monkeypatch):
    """label_communities must force max_concurrency=1 for copilot-cli, mirroring
    claude-cli, since the exchanged session token is short-lived per call."""
    fake_resp = _fake_openai_response('{"0": "Auth Middleware"}')
    _install_fake_openai(monkeypatch, fake_resp)
    monkeypatch.delenv("GRAPHIFY_COPILOT_CLI_PARALLEL", raising=False)

    import networkx as nx
    G = nx.Graph()
    G.add_node("a", label="A")
    labels = llm.label_communities(G, {0: ["a"]}, backend="copilot-cli", max_concurrency=4)
    assert labels[0] == "Auth Middleware"
