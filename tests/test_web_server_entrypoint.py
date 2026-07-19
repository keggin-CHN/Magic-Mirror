"""Regression tests for the standalone Web server entrypoint."""

import json
import sys
from types import SimpleNamespace

import pytest
import web_server
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


def test_parse_env_port_reads_valid_value(monkeypatch):
    monkeypatch.setenv('WEB_PORT', '19033')

    assert web_server._parse_env_port('WEB_PORT', 8033) == 19033


def test_parse_env_port_rejects_non_integer_value(monkeypatch):
    monkeypatch.setenv('WEB_PORT', 'not-a-port')

    with pytest.raises(ValueError, match='WEB_PORT must be an integer'):
        web_server._parse_env_port('WEB_PORT', 8033)


@pytest.mark.parametrize('value', ['0', '-1', '65536'])
def test_parse_env_port_rejects_out_of_range_value(monkeypatch, value):
    monkeypatch.setenv('WEB_PORT', value)

    with pytest.raises(ValueError, match='WEB_PORT must be between 1 and 65535'):
        web_server._parse_env_port('WEB_PORT', 8033)


def test_main_returns_failure_for_invalid_port(monkeypatch, capsys):
    monkeypatch.setenv('WEB_PORT', 'not-a-port')

    assert web_server.main() == 1

    captured = capsys.readouterr()
    assert 'WEB_PORT must be an integer' in captured.out


def test_init_config_requires_initial_password(monkeypatch, tmp_path):
    config_path = tmp_path / 'config.json'
    monkeypatch.setattr(web_server, 'CONFIG_PATH', str(config_path))
    monkeypatch.delenv('WEB_INITIAL_PASSWORD', raising=False)

    assert web_server.main(['web_server.py', '--init-config']) == 2
    assert not config_path.exists()


def test_init_config_writes_hashed_password(monkeypatch, tmp_path):
    config_path = tmp_path / 'config.json'
    monkeypatch.setattr(web_server, 'CONFIG_PATH', str(config_path))
    monkeypatch.setattr(web_server, 'PASSWORD_HASH_ITERATIONS', 10)
    monkeypatch.setenv('WEB_INITIAL_PASSWORD', 'generated-password')

    assert web_server.main(['web_server.py', '--init-config']) == 0

    config = json.loads(config_path.read_text(encoding='utf-8'))
    assert 'password' not in config
    assert web_server._verify_password('generated-password', config)


def test_init_config_does_not_overwrite_existing_config(monkeypatch, tmp_path):
    config_path = tmp_path / 'config.json'
    config_path.write_text('{"password":"keep-me"}\n', encoding='utf-8')
    monkeypatch.setattr(web_server, 'CONFIG_PATH', str(config_path))
    monkeypatch.setenv('WEB_INITIAL_PASSWORD', 'generated-password')

    assert web_server.main(['web_server.py', '--init-config']) == 0
    assert config_path.read_text(encoding='utf-8') == '{"password":"keep-me"}\n'


def test_main_passes_host_and_port_to_uvicorn(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setenv('WEB_HOST', '127.0.0.1')
    monkeypatch.setenv('WEB_PORT', '19033')
    monkeypatch.setitem(sys.modules, 'uvicorn', SimpleNamespace(run=run))

    assert web_server.main() == 0
    assert calls == [
        (
            (web_server.app,),
            {'host': '127.0.0.1', 'port': 19033, 'access_log': False},
        )
    ]


def test_web_index_serves_dist_index_without_cache(monkeypatch, tmp_path):
    index_path = tmp_path / 'index.html'
    index_path.write_text('<main>Magic Mirror</main>', encoding='utf-8')
    monkeypatch.setattr(web_server, 'DIST_DIR', str(tmp_path))

    response = TestClient(web_server.app).get('/')

    assert response.status_code == 200
    assert response.headers.get('cache-control') == 'no-store'
    assert response.text == '<main>Magic Mirror</main>'


def test_web_assets_serve_static_file_without_cache(monkeypatch, tmp_path):
    asset_path = tmp_path / 'assets' / 'app.js'
    asset_path.parent.mkdir()
    asset_path.write_text('console.log("ok");', encoding='utf-8')
    monkeypatch.setattr(web_server, 'DIST_DIR', str(tmp_path))

    response = TestClient(web_server.app).get('/assets/app.js')

    assert response.status_code == 200
    assert response.headers.get('cache-control') == 'no-store'
    assert response.text == 'console.log("ok");'


def test_web_assets_cache_hashed_vite_assets(monkeypatch, tmp_path):
    asset_path = tmp_path / 'assets' / 'index-B8d9f3a1.js'
    asset_path.parent.mkdir()
    asset_path.write_text('console.log("hashed");', encoding='utf-8')
    monkeypatch.setattr(web_server, 'DIST_DIR', str(tmp_path))

    response = TestClient(web_server.app).get('/assets/index-B8d9f3a1.js')

    assert response.status_code == 200
    assert (
        response.headers.get('cache-control')
        == 'public, max-age=31536000, immutable'
    )
    assert response.text == 'console.log("hashed");'


def test_web_assets_return_404_when_dist_has_no_index(monkeypatch, tmp_path):
    monkeypatch.setattr(web_server, 'DIST_DIR', str(tmp_path))

    response = TestClient(web_server.app).get('/missing-route')

    assert response.status_code == 404


def test_unknown_api_route_does_not_fall_back_to_spa(monkeypatch, tmp_path):
    index_path = tmp_path / 'index.html'
    index_path.write_text('<main>Magic Mirror</main>', encoding='utf-8')
    monkeypatch.setattr(web_server, 'DIST_DIR', str(tmp_path))

    response = TestClient(web_server.app).get('/api/typo')

    assert response.status_code == 404
    assert response.json() == {'error': 'not-found'}


def test_cors_wildcard_does_not_advertise_credential_support():
    response = TestClient(web_server.app).options(
        '/api/status',
        headers={
            'Origin': 'https://example.test',
            'Access-Control-Request-Method': 'GET',
        },
    )

    assert response.status_code == 200
    assert response.headers['access-control-allow-origin'] == '*'
    assert 'access-control-allow-credentials' not in response.headers


def test_video_progress_websocket_rejects_missing_token():
    with pytest.raises(WebSocketDisconnect) as error:
        with TestClient(web_server.app).websocket_connect('/api/task/video/ws/task-1'):
            pass

    assert error.value.code == 1008


def test_video_progress_websocket_accepts_valid_token(monkeypatch):
    token = web_server._issue_token()
    monkeypatch.setattr(
        web_server,
        '_get_video_task_progress',
        lambda task_id: {'task_id': task_id, 'status': 'success'},
    )

    with TestClient(web_server.app).websocket_connect(
        f'/api/task/video/ws/task-1?token={token}'
    ) as websocket:
        assert websocket.receive_json() == {
            'task_id': 'task-1',
            'status': 'success',
        }
