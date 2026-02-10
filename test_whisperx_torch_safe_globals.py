import sys
import types


def _install_fake_omegaconf(monkeypatch):
    pkg = types.ModuleType("omegaconf")

    list_mod = types.ModuleType("omegaconf.listconfig")
    dict_mod = types.ModuleType("omegaconf.dictconfig")
    base_mod = types.ModuleType("omegaconf.base")

    class ListConfig:  # noqa: D401
        """Fake OmegaConf ListConfig."""

    class DictConfig:  # noqa: D401
        """Fake OmegaConf DictConfig."""

    class ContainerMetadata:  # noqa: D401
        """Fake OmegaConf ContainerMetadata."""

    list_mod.ListConfig = ListConfig
    dict_mod.DictConfig = DictConfig
    base_mod.ContainerMetadata = ContainerMetadata

    monkeypatch.setitem(sys.modules, "omegaconf", pkg)
    monkeypatch.setitem(sys.modules, "omegaconf.listconfig", list_mod)
    monkeypatch.setitem(sys.modules, "omegaconf.dictconfig", dict_mod)
    monkeypatch.setitem(sys.modules, "omegaconf.base", base_mod)
    return ListConfig, DictConfig


def test_whisperx_load_model_retries_after_torch_safe_globals(monkeypatch):
    # Import module (may have whisperx/torch unavailable in this env)
    import core.whisperx_transcriber as wx

    # Force availability and stub deps
    monkeypatch.setattr(wx, "WHISPERX_AVAILABLE", True)
    _install_fake_omegaconf(monkeypatch)

    calls = {"load_model": 0, "add_safe_globals": 0}

    class FakeSerialization:
        def add_safe_globals(self, _items):
            calls["add_safe_globals"] += 1

    class FakeTorch:
        __version__ = "2.6.0"
        serialization = FakeSerialization()

    class FakeWhisperX:
        def load_model(self, *_args, **_kwargs):
            calls["load_model"] += 1
            if calls["load_model"] == 1:
                raise RuntimeError(
                    "Weights only load failed. "
                    "WeightsUnpickler error: Unsupported global: GLOBAL omegaconf.listconfig.ListConfig"
                )
            return object()

    monkeypatch.setattr(wx, "torch", FakeTorch)
    monkeypatch.setattr(wx, "whisperx", FakeWhisperX())

    t = wx.WhisperXTranscriber(model_name="tiny", device="cpu", compute_type="int8", language=None)
    assert t.load_model() is True
    assert calls["load_model"] == 2
    assert calls["add_safe_globals"] >= 1
    assert t.last_error is None


def test_whisperx_load_model_does_not_retry_on_other_errors(monkeypatch):
    import core.whisperx_transcriber as wx

    monkeypatch.setattr(wx, "WHISPERX_AVAILABLE", True)
    _install_fake_omegaconf(monkeypatch)

    calls = {"load_model": 0, "add_safe_globals": 0}

    class FakeSerialization:
        def add_safe_globals(self, _items):
            calls["add_safe_globals"] += 1

    class FakeTorch:
        __version__ = "2.6.0"
        serialization = FakeSerialization()

    class FakeWhisperX:
        def load_model(self, *_args, **_kwargs):
            calls["load_model"] += 1
            raise RuntimeError("some other failure")

    monkeypatch.setattr(wx, "torch", FakeTorch)
    monkeypatch.setattr(wx, "whisperx", FakeWhisperX())

    t = wx.WhisperXTranscriber(model_name="tiny", device="cpu", compute_type="int8", language=None)
    assert t.load_model() is False
    assert calls["load_model"] == 1
    assert t.last_error == "some other failure"


def test_whisperx_load_model_falls_back_to_silero_vad_on_weights_only(monkeypatch):
    import core.whisperx_transcriber as wx

    monkeypatch.setattr(wx, "WHISPERX_AVAILABLE", True)
    _install_fake_omegaconf(monkeypatch)

    class FakeSerialization:
        def add_safe_globals(self, _items):
            return None

    class FakeTorch:
        __version__ = "2.8.0"
        serialization = FakeSerialization()

    calls = []

    class FakeWhisperX:
        def load_model(self, *_args, **kwargs):
            calls.append(kwargs.get("vad_method"))
            if kwargs.get("vad_method") == "pyannote":
                raise RuntimeError(
                    "Weights only load failed. WeightsUnpickler error: Unsupported global: GLOBAL typing.Any"
                )
            return object()

    monkeypatch.setattr(wx, "torch", FakeTorch)
    monkeypatch.setattr(wx, "whisperx", FakeWhisperX())

    t = wx.WhisperXTranscriber(model_name="tiny", device="cpu", compute_type="int8", language=None, vad_method="pyannote")
    assert t.load_model() is True
    assert t.vad_method == "silero"
    # Called at least once with pyannote, then silero.
    assert "pyannote" in calls
    assert "silero" in calls
