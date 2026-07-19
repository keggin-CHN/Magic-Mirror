"""Static regressions for the Linux web installer."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = PROJECT_ROOT / 'scripts' / 'install-server-linux.sh'


def test_installer_does_not_default_to_fixed_task_config_secret():
    source = INSTALLER.read_text(encoding='utf-8')

    assert 'VIDEO_TASK_CONFIG_SECRET="${VIDEO_TASK_CONFIG_SECRET:-}"' in source
    assert 'VIDEO_TASK_CONFIG_SECRET="${VIDEO_TASK_CONFIG_SECRET:-magic-mirror-config-secret}"' not in source
    assert 'VIDEO_TASK_CONFIG_SECRET="$(random_hex)"' in source


def test_installer_initializes_web_config_without_persisting_plaintext_password():
    source = INSTALLER.read_text(encoding='utf-8')

    assert '"$WEB_SERVER_BIN" --init-config' in source
    assert 'WEB_INITIAL_PASSWORD="$WEB_INITIAL_PASSWORD"' in source
    assert 'Environment="WEB_INITIAL_PASSWORD=' not in source


def test_installer_avoids_query_token_access_logs_and_ambiguous_tarball():
    source = INSTALLER.read_text(encoding='utf-8')

    assert 'access_log off;' in source
    assert 'Multiple magicmirror_web_*.tar.gz bundles found' in source
    assert '| head -n 1' not in source

