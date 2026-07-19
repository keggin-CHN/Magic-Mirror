"""Regression tests for Web server credential persistence."""

import json

import pytest
import web_server
from fastapi.testclient import TestClient


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    config_path = tmp_path / 'config.json'
    monkeypatch.setattr(web_server, 'CONFIG_PATH', str(config_path))
    monkeypatch.setattr(web_server, 'PASSWORD_HASH_ITERATIONS', 10)
    with web_server.TOKENS_LOCK:
        web_server.TOKENS.clear()
    with web_server.LOGIN_FAILURES_LOCK:
        web_server.LOGIN_FAILURES.clear()
    yield config_path
    with web_server.TOKENS_LOCK:
        web_server.TOKENS.clear()
    with web_server.LOGIN_FAILURES_LOCK:
        web_server.LOGIN_FAILURES.clear()


def test_new_config_stores_only_a_salted_password_hash(isolated_config):
    cfg = web_server._load_config()

    assert 'password' not in cfg
    assert cfg['passwordHash']
    assert cfg['passwordSalt']
    assert web_server._verify_password('123456', cfg)
    assert not web_server._verify_password('wrong-password', cfg)
    assert json.loads(isolated_config.read_text(encoding='utf-8')) == cfg


def test_successful_login_migrates_legacy_plaintext_config(isolated_config):
    isolated_config.write_text(
        json.dumps({'password': 'legacy-password'}), encoding='utf-8'
    )

    response = TestClient(web_server.app).post(
        '/api/login', json={'password': 'legacy-password'}
    )

    assert response.status_code == 200
    assert response.json()['token']
    assert web_server.AUTH_COOKIE_NAME in response.cookies
    migrated = json.loads(isolated_config.read_text(encoding='utf-8'))
    assert 'password' not in migrated
    assert web_server._verify_password('legacy-password', migrated)


def test_atomic_config_write_preserves_existing_file_on_replace_failure(
    monkeypatch, isolated_config
):
    original = '{"password":"keep-me"}\n'
    isolated_config.write_text(original, encoding='utf-8')

    def fail_replace(_source, _destination):
        raise OSError('replace-failed')

    monkeypatch.setattr(web_server.os, 'replace', fail_replace)

    with pytest.raises(OSError, match='replace-failed'):
        web_server._save_config(web_server._build_password_config('new-password'))

    assert isolated_config.read_text(encoding='utf-8') == original
    assert list(isolated_config.parent.glob('.config-*.tmp')) == []

def test_login_rate_limit_throttles_repeated_failures(monkeypatch, isolated_config):
    now = 1000.0
    monkeypatch.setattr(web_server.time, 'time', lambda: now)
    client = TestClient(web_server.app)

    for _ in range(web_server.LOGIN_MAX_FAILURES):
        response = client.post('/api/login', json={'password': 'wrong-password'})
        assert response.status_code == 401

    response = client.post('/api/login', json={'password': '123456'})

    assert response.status_code == 429
    assert response.headers['Retry-After'] == str(web_server.LOGIN_LOCKOUT_SECONDS)
    assert response.json() == {'error': 'too-many-login-attempts'}


def test_successful_login_after_lockout_clears_failed_attempts(
    monkeypatch, isolated_config
):
    now = 1000.0
    monkeypatch.setattr(web_server.time, 'time', lambda: now)
    client = TestClient(web_server.app)

    for _ in range(web_server.LOGIN_MAX_FAILURES):
        response = client.post('/api/login', json={'password': 'wrong-password'})
        assert response.status_code == 401

    now += web_server.LOGIN_LOCKOUT_SECONDS + 1
    response = client.post('/api/login', json={'password': '123456'})

    assert response.status_code == 200
    assert response.json()['token']
    with web_server.LOGIN_FAILURES_LOCK:
        assert web_server.LOGIN_FAILURES == {}


def test_updating_password_revokes_existing_tokens(isolated_config):
    client = TestClient(web_server.app)
    token = client.post('/api/login', json={'password': '123456'}).json()['token']

    response = client.post(
        '/api/credential',
        headers={'Authorization': f'Bearer {token}'},
        json={'password': 'new-password'},
    )

    assert response.status_code == 200
    assert response.json()['token']
    assert response.json()['token'] != token
    stale_response = client.get('/api/library', headers={'Authorization': f'Bearer {token}'})
    assert stale_response.status_code == 401


def test_login_cookie_can_authenticate_requests(monkeypatch, tmp_path, isolated_config):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    monkeypatch.setattr(web_server, 'LIBRARY_DIR', str(library_dir))
    client = TestClient(web_server.app)
    response = client.post('/api/login', json={'password': '123456'})

    assert response.status_code == 200
    assert response.json()['token']
    assert response.cookies.get(web_server.AUTH_COOKIE_NAME)

    library_response = client.get('/api/library')

    assert library_response.status_code == 200
    assert library_response.json() == {'items': []}


def test_query_token_remains_supported_for_legacy_links(
    monkeypatch, tmp_path, isolated_config
):
    library_dir = tmp_path / 'library'
    library_dir.mkdir()
    monkeypatch.setattr(web_server, 'LIBRARY_DIR', str(library_dir))
    token = web_server._issue_token()

    response = TestClient(web_server.app).get(f'/api/library?token={token}')

    assert response.status_code == 200
    assert response.json() == {'items': []}

