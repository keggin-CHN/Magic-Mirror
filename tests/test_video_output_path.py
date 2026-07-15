"""Tests for unique video output paths."""

from types import SimpleNamespace

from magic import face


def test_output_video_path_is_unique_and_uses_mp4_extension(monkeypatch, tmp_path):
    tokens = iter(['first-task', 'second-task'])
    monkeypatch.setattr(
        face.uuid,
        'uuid4',
        lambda: SimpleNamespace(hex=next(tokens)),
    )
    input_path = tmp_path / 'sample.video.mov'

    first_path = face._get_output_video_path(str(input_path))
    second_path = face._get_output_video_path(str(input_path))

    assert first_path != second_path
    assert first_path.endswith('_output_first-task.mp4')
    assert second_path.endswith('_output_second-task.mp4')


def test_output_image_path_is_unique_and_preserves_extension(monkeypatch, tmp_path):
    tokens = iter(['first-image', 'second-image'])
    monkeypatch.setattr(
        face.uuid,
        'uuid4',
        lambda: SimpleNamespace(hex=next(tokens)),
    )
    input_path = tmp_path / 'portrait.final.png'

    first_path = face._get_output_file_path(str(input_path))
    second_path = face._get_output_file_path(str(input_path))

    assert first_path != second_path
    assert first_path.endswith('_output_first-image.png')
    assert second_path.endswith('_output_second-image.png')
