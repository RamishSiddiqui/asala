"""Tests for Asala CLI."""

import json
import os
import tempfile

import pytest

from asala import Asala, CryptoUtils
from asala.cli import keys_command, sign_command, verify_command


class _Args:
    """Minimal args namespace for testing CLI functions."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestVerifyCommand:
    def test_verify_signed_file_json(self, tmp_path, key_pair, capsys):
        """Verify a signed file in JSON mode."""
        _, private_key = key_pair
        asala = Asala()

        content = b"CLI test content"
        content_path = tmp_path / "test.txt"
        content_path.write_bytes(content)

        manifest = asala.sign_content(content, private_key, "tester")

        # Write manifest JSON
        manifest_dict = {
            "id": manifest.id,
            "content_hash": manifest.content_hash,
            "content_type": manifest.content_type.value,
            "created_at": manifest.created_at,
            "created_by": manifest.created_by,
            "signatures": [
                {
                    "algorithm": s.algorithm,
                    "public_key": s.public_key,
                    "signature": s.signature,
                    "timestamp": s.timestamp,
                    "signer": s.signer,
                }
                for s in manifest.signatures
            ],
            "assertions": [
                {"type": a.type, "data": a.data, "hash": a.hash}
                for a in manifest.assertions
            ],
            "chain": [],
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_dict))

        args = _Args(
            file=str(content_path),
            manifest=str(manifest_path),
            trust=[],
            json=True,
            verbose=False,
            physics=False,
        )

        with pytest.raises(SystemExit) as exc:
            verify_command(args)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "status" in output
        assert "layers" in output

    def test_verify_missing_file(self, capsys):
        args = _Args(
            file="/nonexistent/file.txt",
            manifest=None,
            trust=[],
            json=False,
            verbose=False,
            physics=False,
        )
        with pytest.raises(SystemExit) as exc:
            verify_command(args)
        assert exc.value.code == 1


class TestSignCommand:
    def test_sign_creates_manifest(self, tmp_path, key_pair):
        _, private_key = key_pair
        content_path = tmp_path / "image.jpg"
        content_path.write_bytes(b"\xff\xd8fake jpeg content")

        key_path = tmp_path / "private.pem"
        key_path.write_text(private_key)

        output_path = tmp_path / "output.manifest.json"

        args = _Args(
            file=str(content_path),
            key=str(key_path),
            output=str(output_path),
            creator="test-signer",
            device="test-device",
        )

        sign_command(args)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["created_by"] == "test-signer"
        assert "content_hash" in data
        assert len(data["signatures"]) == 1

    def test_sign_missing_file(self, tmp_path, key_pair):
        _, private_key = key_pair
        key_path = tmp_path / "private.pem"
        key_path.write_text(private_key)

        args = _Args(
            file="/nonexistent/file",
            key=str(key_path),
            output=None,
            creator=None,
            device=None,
        )
        with pytest.raises(SystemExit) as exc:
            sign_command(args)
        assert exc.value.code == 1

    def test_sign_missing_key(self, tmp_path):
        content_path = tmp_path / "data.bin"
        content_path.write_bytes(b"data")

        args = _Args(
            file=str(content_path),
            key="/nonexistent/key.pem",
            output=None,
            creator=None,
            device=None,
        )
        with pytest.raises(SystemExit) as exc:
            sign_command(args)
        assert exc.value.code == 1


class TestKeysCommand:
    def test_generate_keys(self, tmp_path, capsys):
        output_dir = tmp_path / "keys"
        args = _Args(generate=True, output=str(output_dir))

        keys_command(args)

        assert (output_dir / "private.pem").exists()
        assert (output_dir / "public.pem").exists()

        private_text = (output_dir / "private.pem").read_text()
        public_text = (output_dir / "public.pem").read_text()
        assert "BEGIN PRIVATE KEY" in private_text
        assert "BEGIN PUBLIC KEY" in public_text

    def test_no_generate_prints_help(self, capsys):
        args = _Args(generate=False, output="./keys")
        keys_command(args)
        captured = capsys.readouterr()
        assert "Generate" in captured.out or "generate" in captured.out
