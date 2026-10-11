import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import compression
from tools.compression import compress_context_impl
from tools.dispatch import execute_tool_calls, fill_dangling_tool_calls
from tools.project import set_active_project_folder
from tools.writer import write_file_impl


class FakeClient:
    def __init__(self, content="SUMMARY"):
        self.calls = []
        msg = SimpleNamespace(content=content)
        resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._resp = resp

    def _create(self, **kw):
        self.calls.append(kw)
        return self._resp


def tc(i, name="write_file", args=None):
    return {"id": f"c{i}", "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args or {})}}


def history(n_pairs):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]
    for i in range(n_pairs):
        msgs.append({"role": "assistant", "tool_calls": [tc(i, args={"filename": f"f{i}", "content": "x" * 50, "mode": "create"})]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "write_file", "content": "ok"})
    return msgs


# 1. save-only mode
def test_save_only_writes_file_and_keeps_messages(tmp_path):
    set_active_project_folder(str(tmp_path))
    msgs = history(1)  # shorter than keep_recent
    res = compress_context_impl(msgs, FakeClient(), "m", save_only=True)
    assert res["compressed_messages"] is msgs
    assert os.path.isfile(res["summary_file"])
    assert "SUMMARY" in open(res["summary_file"]).read()


def test_summary_sees_tool_calls_and_empty_summary_falls_back(tmp_path):
    set_active_project_folder(str(tmp_path))
    client = FakeClient(content=None)
    res = compress_context_impl(history(2), client, "m", save_only=True)
    prompt = client.calls[0]["messages"][1]["content"]
    assert "write_file(file=f0" in prompt and "chars=50" in prompt
    assert "digest" in open(res["summary_file"]).read()


# 3. dangling tool calls
def test_every_tool_call_gets_reply_on_errors():
    msgs = [{"role": "assistant", "tool_calls": []}]
    calls = [tc(1, args="{bad json"), tc(2, "boom", {}), tc(3, "nope", {}), tc(4, "ok", {"a": 1})]

    def boom(**kw):
        raise ValueError("bad")

    execute_tool_calls(calls, "tool_calls", msgs, {"boom": boom, "ok": lambda **kw: "fine"}, log=lambda s: None)
    replies = [m for m in msgs if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in replies] == ["c1", "c2", "c3", "c4"]
    assert replies[3]["content"] == "fine"
    assert all(m["content"].startswith("Error") for m in replies[:3])


def test_length_finish_reason_returns_shorter_files_error():
    msgs = []
    ran = []
    execute_tool_calls([tc(1)], "length", msgs, {"write_file": lambda **kw: ran.append(1)}, log=lambda s: None)
    assert not ran and "shorter" in msgs[0]["content"]


def test_fill_dangling():
    msgs = [{"role": "assistant", "tool_calls": [tc(1), tc(2)]},
            {"role": "tool", "tool_call_id": "c1", "name": "write_file", "content": "ok"}]
    assert fill_dangling_tool_calls(msgs) == 1
    assert msgs[-1]["tool_call_id"] == "c2"


# 4. boundary
def test_recent_block_never_starts_with_tool(tmp_path):
    set_active_project_folder(str(tmp_path))
    msgs = history(6)
    for keep in range(1, 9):
        res = compress_context_impl(msgs, FakeClient(), "m", keep_recent=keep)
        out = res["compressed_messages"]
        if res["summary_file"]:
            assert out[2]["role"] != "tool"
            assert out[0]["role"] == "system" and out[1]["role"] == "user"


# 6. filenames
def test_write_file_rejects_escapes(tmp_path):
    set_active_project_folder(str(tmp_path / "proj"))
    os.makedirs(tmp_path / "proj")
    for bad in ["../x", "a/b", "..", "a\\b", "/etc/passwd"]:
        assert write_file_impl(bad, "x", "create").startswith("Error")
    assert not (tmp_path / "x.md").exists()
    assert write_file_impl("ch1", "x", "create").startswith("Successfully")
    assert (tmp_path / "proj" / "ch1.md").exists()
