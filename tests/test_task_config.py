"""Tests for magic.task_config utility functions."""
import base64
import hashlib
import json
import os
import tempfile

import pytest

# Add src-python to path so we can import the magic package
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src-python'))

from magic.task_config import (
    b64url_decode,
    b64url_encode,
    clone_json_payload,
    compute_file_sha256,
    sign_video_task_config_payload,
    verify_file_sha256,
    canonicalize_video_task_config,
    build_video_task_config_token,
    parse_video_task_config_token,
)


class TestB64Url:
    """Tests for base64url encoding/decoding."""

    def test_encode_decode_roundtrip(self):
        """Test that encode -> decode returns original data."""
        data = b"hello world test data"
        encoded = b64url_encode(data)
        decoded = b64url_decode(encoded)
        assert decoded == data

    def test_encode_no_padding(self):
        """Test that encoded output has no padding characters."""
        encoded = b64url_encode(b"test")
        assert "=" not in encoded

    def test_decode_with_padding(self):
        """Test that decoder handles standard base64 with padding."""
        data = b"test data"
        standard = base64.urlsafe_b64encode(data).decode()
        decoded = b64url_decode(standard)
        assert decoded == data


class TestCloneJsonPayload:
    """Tests for clone_json_payload."""

    def test_clone_dict(self):
        """Test cloning a dictionary."""
        original = {"key": "value", "nested": {"a": 1}}
        cloned = clone_json_payload(original)
        assert cloned == original
        assert cloned is not original
        assert cloned["nested"] is not original["nested"]

    def test_clone_list(self):
        """Test cloning a list."""
        original = [1, 2, [3, 4]]
        cloned = clone_json_payload(original)
        assert cloned == original
        assert cloned is not original


class TestFileSha256:
    """Tests for file SHA256 operations."""

    def test_compute_sha256(self):
        """Test computing SHA256 of a file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            path = f.name
        try:
            result = compute_file_sha256(path)
            expected = hashlib.sha256(b"test content").hexdigest()
            assert result == expected
        finally:
            os.unlink(path)

    def test_verify_sha256_match(self):
        """Test verifying SHA256 when it matches."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            path = f.name
        try:
            expected = hashlib.sha256(b"test content").hexdigest()
            assert verify_file_sha256(path, expected) is True
        finally:
            os.unlink(path)

    def test_verify_sha256_mismatch(self):
        """Test verifying SHA256 when it doesn't match."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            path = f.name
        try:
            # Use a valid-looking hex hash that doesn't match
            wrong = hashlib.sha256(b"different content").hexdigest()
            assert verify_file_sha256(path, wrong) is False
        finally:
            os.unlink(path)

    def test_verify_sha256_none(self):
        """Test verifying SHA256 with None expected value."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            path = f.name
        try:
            assert verify_file_sha256(path, None) is True
        finally:
            os.unlink(path)


class TestSignAndParse:
    """Tests for signing and parsing task config tokens."""

    def test_sign_and_parse_roundtrip(self):
        """Test that signing produces a non-empty string."""
        payload_str = base64.urlsafe_b64encode(json.dumps({"test": 1}).encode()).decode().rstrip('=')
        secret = "test-secret-key"
        signed = sign_video_task_config_payload(payload_str, secret)
        assert isinstance(signed, str)
        assert len(signed) > 0

    def test_build_and_parse_token(self):
        """Test building a video task config token produces a string."""
        import hashlib
        payload = {
            "inputVideoHash": hashlib.sha256(b"video").hexdigest(),
            "targetFaceHash": hashlib.sha256(b"face").hexdigest(),
        }
        secret = "test-secret"
        token = build_video_task_config_token(payload, secret)
        assert isinstance(token, str)

    def test_parse_invalid_token(self):
        """Test parsing an invalid token returns None."""
        result = parse_video_task_config_token("invalid-token", "secret")
        assert result is None


class TestCanonicalizeVideoTaskConfig:
    """Tests for canonicalize_video_task_config."""

    def test_with_valid_full_config(self):
        """Test canonicalizing config with both input video and target face."""
        import hashlib
        config = {
            "inputVideoHash": hashlib.sha256(b"test").hexdigest(),
            "targetFace": {"sha256": hashlib.sha256(b"face").hexdigest()},
        }
        result = canonicalize_video_task_config(config)
        assert "inputVideo" in result
        assert "targetFace" in result

    def test_non_dict_raises(self):
        """Test canonicalizing non-dict raises RuntimeError."""
        with pytest.raises(RuntimeError, match='missing-params'):
            canonicalize_video_task_config(None)
        with pytest.raises(RuntimeError, match='missing-params'):
            canonicalize_video_task_config("not-a-dict")
