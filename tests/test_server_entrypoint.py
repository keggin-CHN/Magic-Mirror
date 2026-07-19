"""Regression tests for the packaged desktop server entrypoint."""

import pytest
import server


def test_parse_port_reads_valid_environment_value(monkeypatch):
    monkeypatch.setenv('MIRROR_PORT', '9100')

    assert server._parse_port('MIRROR_PORT', 8023) == 9100


def test_parse_port_rejects_non_integer_environment_value(monkeypatch):
    monkeypatch.setenv('MIRROR_PORT', 'not-a-port')

    with pytest.raises(ValueError, match='MIRROR_PORT must be an integer'):
        server._parse_port('MIRROR_PORT', 8023)


@pytest.mark.parametrize('value', ['0', '-1', '65536'])
def test_parse_port_rejects_out_of_range_value(monkeypatch, value):
    monkeypatch.setenv('MIRROR_PORT', value)

    with pytest.raises(ValueError, match='MIRROR_PORT must be between 1 and 65535'):
        server._parse_port('MIRROR_PORT', 8023)


def test_main_logs_invalid_port_before_importing_app(monkeypatch):
    logs = []

    monkeypatch.setenv('MIRROR_PORT', 'not-a-port')
    monkeypatch.setattr(server, '_append_boot_log', logs.append)
    monkeypatch.setattr(server.time, 'sleep', lambda _seconds: None)

    assert server.main(['server.py']) == 1
    assert any('MIRROR_PORT must be an integer' in item for item in logs)
    assert not any(item == 'import magic.app: OK' for item in logs)


def test_provider_check_requires_provider_argument(monkeypatch):
    logs = []
    monkeypatch.setattr(server, '_append_boot_log', logs.append)

    assert server._check_ort_provider_from_cli(['server.py', '--check-ort-provider']) == 2
    assert logs == ['--check-ort-provider requires a provider name']
