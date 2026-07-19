"""Static regressions for the Web frontend API client."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_CLIENT = PROJECT_ROOT / 'src' / 'services' / 'webServer.ts'


def test_media_url_builders_do_not_append_auth_token_query():
    source = WEB_CLIENT.read_text(encoding='utf-8')

    assert '_withTokenQuery' not in source
    assert 'token=${encodeURIComponent' not in source
    assert 'buildFileUrl(fileId: string)' in source
    assert 'buildLibraryUrl(fileName: string)' in source
    assert 'buildDownloadUrl(fileId: string)' in source

