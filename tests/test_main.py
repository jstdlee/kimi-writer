"""End-to-end tests of the agent loop with a mocked OpenAI client (no network)."""
import importlib.util
import json
import os
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import project  # noqa: E402


def load_app():
    spec = importlib.util.spec_from_file_location("kimi_writer_app", os.path.join(ROOT, "kimi-writer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def chunk(content=None, reasoning=None, tool_calls=None, finish=None, role=None):
    delta = NS(role=role, content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    return NS(choices=[NS(delta=delta, finish_reason=finish)])


def tcd(index, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


class FakeOpenAI:
    """Scripted client: `script` is a list of chunk lists (one per streamed call)."""
    script = []
    stream_calls = []
    summary_calls = []

    def __init__(self, **kw):
        outer = self
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        if kw.get("stream"):
            FakeOpenAI.stream_calls.append(json.loads(json.dumps(kw["messages"])))
            item = FakeOpenAI.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return iter(item)
        FakeOpenAI.summary_calls.append(kw)
        return NS(choices=[NS(message=NS(content="SUMMARY TEXT"))])


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-12345678")
    monkeypatch.setattr(project, "get_output_dir", lambda: str(tmp_path / "output"))
    project.set_active_project_folder(None)
    FakeOpenAI.script, FakeOpenAI.stream_calls, FakeOpenAI.summary_calls = [], [], []
    mod = load_app()
    monkeypatch.setattr(mod, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(mod, "estimate_token_count", lambda *a, **k: 1000)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.chdir(tmp_path)
    return mod


def run(app, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["kimi-writer.py", *argv])
    app.main()


def test_help_works_without_api_key(monkeypatch, capsys):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    mod = load_app()
    monkeypatch.setattr(sys, "argv", ["kimi-writer.py", "--help"])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert e.value.code == 0
    assert "--recover" in capsys.readouterr().out


def test_full_loop_streamed_tool_calls_then_done(app, monkeypatch, tmp_path):
    args1 = json.dumps({"project_name": "My Book"})
    args2 = json.dumps({"filename": "ch1", "content": "Hello 世界", "mode": "create"})
    FakeOpenAI.script = [
        # turn 1: reasoning + two tool calls whose arguments are split over chunks;
        # finish_reason chunk is followed by an empty-choices usage chunk and a None-finish chunk
        [chunk(reasoning="think ", role="assistant"),
         chunk(reasoning="more"),
         chunk(tool_calls=[tcd(0, "call_a", "create_project", args1[:5])]),
         chunk(tool_calls=[tcd(0, None, None, args1[5:])]),
         chunk(tool_calls=[tcd(1, "call_b", "write_file", args2[:10])]),
         chunk(tool_calls=[tcd(1, None, None, args2[10:])]),
         chunk(finish="tool_calls"),
         NS(choices=[]),
         chunk()],
        [chunk(content="All done.", finish="stop")],
    ]
    run(app, monkeypatch, "write a book")
    assert (tmp_path / "output" / "My_Book" / "ch1.md").read_text(encoding="utf-8") == "Hello 世界"
    second = FakeOpenAI.stream_calls[1]
    asst = second[2]
    assert asst["reasoning_content"] == "think more"
    assert [t["id"] for t in asst["tool_calls"]] == ["call_a", "call_b"]
    assert [m["tool_call_id"] for m in second[3:5]] == ["call_a", "call_b"]
    assert all(m["role"] == "tool" for m in second[3:5])
    assert not FakeOpenAI.script


def test_truncated_tool_call_is_answered_not_executed(app, monkeypatch, tmp_path):
    FakeOpenAI.script = [
        [chunk(tool_calls=[tcd(0, "c1", "create_project", '{"project_na')], finish="length")],
        [chunk(content="ok", finish="stop")],
    ]
    run(app, monkeypatch, "x")
    reply = FakeOpenAI.stream_calls[1][-1]
    assert reply["role"] == "tool" and reply["tool_call_id"] == "c1" and "cut off" in reply["content"]
    assert not (tmp_path / "output").exists()


def test_api_errors_back_off_and_stop(app, monkeypatch, capsys):
    FakeOpenAI.script = [RuntimeError("401")] * 10
    run(app, monkeypatch, "x")
    assert len(FakeOpenAI.stream_calls) == app.MAX_CONSECUTIVE_ERRORS
    assert "Stopping" in capsys.readouterr().out


def test_error_then_recovery_continues(app, monkeypatch):
    FakeOpenAI.script = [RuntimeError("503"), [chunk(content="fine", finish="stop")]]
    run(app, monkeypatch, "x")
    assert not FakeOpenAI.script


def test_compression_runs_when_estimate_endpoint_fails(app, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("404")
    monkeypatch.setattr(app, "estimate_token_count", boom)
    monkeypatch.setattr(app, "rough_token_estimate", lambda m: 10**6)
    call = lambda i: [chunk(tool_calls=[tcd(0, f"c{i}", "create_project", '{"project_name": "p"}')], finish="tool_calls")]
    FakeOpenAI.script = [call(i) for i in range(8)] + [[chunk(content="done", finish="stop")]]
    run(app, monkeypatch, "x")
    assert FakeOpenAI.summary_calls
    assert FakeOpenAI.stream_calls[-1][1]["content"].startswith("[CONTEXT SUMMARY")


def test_compress_context_tool_compresses_next_iteration(app, monkeypatch, tmp_path):
    project.set_active_project_folder(None)
    steps = []
    for i in range(8):
        steps.append([chunk(tool_calls=[tcd(0, f"c{i}", "create_project", json.dumps({"project_name": "p"}))], finish="tool_calls")])
    steps.append([chunk(tool_calls=[tcd(0, "cc", "compress_context", "{}")], finish="tool_calls")])
    steps.append([chunk(content="done", finish="stop")])
    FakeOpenAI.script = steps
    run(app, monkeypatch, "x")
    last = FakeOpenAI.stream_calls[-1]
    assert last[1]["content"].startswith("[CONTEXT SUMMARY")
    assert last[2]["role"] != "tool"
    ids_called = {t["id"] for m in last if m.get("tool_calls") for t in m["tool_calls"]}
    ids_replied = {m["tool_call_id"] for m in last if m["role"] == "tool"}
    assert ids_replied <= ids_called


def test_recover_restores_project_only_inside_output(app, monkeypatch, tmp_path):
    proj = tmp_path / "output" / "book"
    proj.mkdir(parents=True)
    f = proj / ".context_summary_20250101_000000.md"
    f.write_text("summary body", encoding="utf-8")
    FakeOpenAI.script = [[chunk(content="done", finish="stop")]]
    run(app, monkeypatch, "--recover", str(f))
    assert project.get_active_project_folder() == str(proj)
    assert "summary body" in FakeOpenAI.stream_calls[0][1]["content"]

    # summary saved in cwd (no project): must NOT make cwd the project folder
    project.set_active_project_folder(None)
    g = tmp_path / ".context_summary_20250101_000001.md"
    g.write_text("x", encoding="utf-8")
    FakeOpenAI.script = [[chunk(content="done", finish="stop")]]
    run(app, monkeypatch, "--recover", str(g))
    assert project.get_active_project_folder() is None


def test_max_iterations_saves_context(app, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "MAX_ITERATIONS", 2)
    call = lambda i: [chunk(tool_calls=[tcd(0, f"c{i}", "create_project", '{"project_name": "p"}')], finish="tool_calls")]
    FakeOpenAI.script = [call(1), call(2)]
    run(app, monkeypatch, "x")
    assert list((tmp_path / "output" / "p").glob(".context_summary_*.md"))
