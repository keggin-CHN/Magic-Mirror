"""Regression tests for Python dependency manifest consistency."""

from pathlib import Path

import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PYTHON = PROJECT_ROOT / 'src-python'
FASTAPI_STACK = {'fastapi', 'uvicorn', 'python-multipart', 'av'}


def _requirement_names(path: Path) -> set[str]:
    names = set()
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.split('#', 1)[0].strip()
        if not line:
            continue
        name = line.split(';', 1)[0].strip()
        for separator in (' ', '<', '>', '=', '!', '~'):
            name = name.split(separator, 1)[0].strip()
        if name:
            names.add(name.lower())
    return names


def test_pip_requirement_files_use_fastapi_stack_without_bottle():
    for relative_path in ('requirements.txt', 'requirements-docker.txt'):
        names = _requirement_names(SRC_PYTHON / relative_path)

        assert 'bottle' not in names
        assert FASTAPI_STACK <= names


def test_poetry_pyproject_uses_fastapi_stack_without_bottle():
    pyproject = tomllib.loads(
        (SRC_PYTHON / 'pyproject.toml').read_text(encoding='utf-8')
    )
    dependencies = {
        name.lower()
        for name in pyproject['tool']['poetry']['dependencies']
        if name.lower() != 'python'
    }

    assert 'bottle' not in dependencies
    assert FASTAPI_STACK <= dependencies


def test_poetry_lock_uses_fastapi_stack_without_bottle():
    lock = tomllib.loads((SRC_PYTHON / 'poetry.lock').read_text(encoding='utf-8'))
    package_names = {
        package['name'].lower() for package in lock.get('package', [])
    }

    assert 'bottle' not in package_names
    assert FASTAPI_STACK <= package_names


def test_docker_ci_builds_image_with_fast_model_skip():
    dockerfile = (PROJECT_ROOT / 'Dockerfile').read_text(encoding='utf-8')
    workflow = (
        PROJECT_ROOT / '.github' / 'workflows' / 'docker-build.yml'
    ).read_text(encoding='utf-8')

    assert 'ARG SKIP_MODEL_DOWNLOAD=0' in dockerfile
    assert 'docker build --build-arg SKIP_MODEL_DOWNLOAD=1' in workflow
