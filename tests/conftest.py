"""Shared test fixtures for Magic-Mirror test suite."""

import os
import sys
import types

# Add src-python to path so we can import the magic package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src-python'))


def _install_missing_av_stub() -> None:
    """Allow non-codec unit tests to import modules when PyAV is unavailable."""
    try:
        import av  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    av_stub = types.ModuleType('av')

    def _missing_av(*_args, **_kwargs):
        raise RuntimeError('av-not-installed')

    class _VideoFrame:
        @staticmethod
        def from_ndarray(*_args, **_kwargs):
            return _missing_av()

    av_stub.open = _missing_av
    av_stub.VideoFrame = _VideoFrame
    sys.modules['av'] = av_stub


def _install_missing_multipart_stub() -> None:
    """Satisfy FastAPI route registration in lean test environments."""
    try:
        import multipart  # noqa: F401
        import multipart.multipart  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    multipart_stub = types.ModuleType('multipart')
    multipart_stub.__version__ = '0.0.9'
    multipart_submodule = types.ModuleType('multipart.multipart')
    multipart_submodule.parse_options_header = lambda value: (value, {})
    sys.modules['multipart'] = multipart_stub
    sys.modules['multipart.multipart'] = multipart_submodule


_install_missing_av_stub()
_install_missing_multipart_stub()
