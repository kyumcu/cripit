import sys
import types


def test_download_whisperx_uses_snapshot_download(monkeypatch):
    import terminal_app

    calls = []

    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(*, repo_id, **_kwargs):
        calls.append(repo_id)
        return "/tmp/fake"

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert terminal_app._download_whisperx_model("tiny") == 0
    assert calls == ["openai/whisper-tiny"]


def test_download_whisperx_large_v3_mapping(monkeypatch):
    import terminal_app

    calls = []

    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(*, repo_id, **_kwargs):
        calls.append(repo_id)
        return "/tmp/fake"

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    assert terminal_app._download_whisperx_model("large-v3") == 0
    assert calls == ["openai/whisper-large-v3"]
