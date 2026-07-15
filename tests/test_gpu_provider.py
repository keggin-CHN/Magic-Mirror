"""Tests for TinyFace execution-provider configuration."""

from magic import face
from magic.face import _prepare_tinyface_with_provider, _tinyface_active_providers


class _FakeSession:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return self._providers


class _FakeComponent:
    def __init__(self, providers):
        self._session = _FakeSession(providers)


class _FakeConfig:
    def __init__(self):
        self.face_inference_providers = ['CPUExecutionProvider']


class _FakeTinyFace:
    def __init__(self):
        self.config = _FakeConfig()
        self.prepared_with = None
        self.detector = _FakeComponent(['DmlExecutionProvider', 'CPUExecutionProvider'])
        self.embedder = _FakeComponent(['DmlExecutionProvider', 'CPUExecutionProvider'])
        self.swapper = _FakeComponent(['DmlExecutionProvider', 'CPUExecutionProvider'])
        self.enhancer = _FakeComponent(['DmlExecutionProvider', 'CPUExecutionProvider'])

    def prepare(self):
        self.prepared_with = list(self.config.face_inference_providers)


def test_prepare_tinyface_uses_real_provider_field_and_restores_config():
    tinyface = _FakeTinyFace()

    _prepare_tinyface_with_provider(tinyface, 'DmlExecutionProvider')

    assert tinyface.prepared_with == [
        'DmlExecutionProvider',
        'CPUExecutionProvider',
    ]
    assert tinyface.config.face_inference_providers == ['CPUExecutionProvider']


def test_tinyface_active_providers_reads_prepared_sessions():
    tinyface = _FakeTinyFace()

    assert _tinyface_active_providers(tinyface) == [
        'DmlExecutionProvider',
        'CPUExecutionProvider',
    ]


def test_gpu_modes_only_expose_runtime_verified_providers(monkeypatch):
    monkeypatch.setattr(
        face,
        '_get_available_execution_providers',
        lambda: ['DmlExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    monkeypatch.setattr(
        face,
        '_is_execution_provider_ready',
        lambda provider: provider == 'DmlExecutionProvider',
    )

    result = face.get_gpu_acceleration_modes()

    assert [mode['id'] for mode in result['modes']] == ['cpu', 'directml']
    assert result['verifiedProviders'] == ['DmlExecutionProvider']
