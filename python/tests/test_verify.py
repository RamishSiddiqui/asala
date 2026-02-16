"""Tests for Asala verification functionality."""

import pytest
import json
from pathlib import Path

from asala import Asala
from asala.types import VerificationStatus


class TestAsalaVerify:
    """Test cases for Asala verification."""

    def test_generate_key_pair(self):
        """Test key pair generation."""
        asala = Asala()
        public_key, private_key = asala.generate_key_pair()
        
        assert public_key is not None
        assert private_key is not None
        assert "-----BEGIN PUBLIC KEY-----" in public_key
        assert "-----END PUBLIC KEY-----" in public_key
        assert "-----BEGIN PRIVATE KEY-----" in private_key
        assert "-----END PRIVATE KEY-----" in private_key

    def test_sign_and_verify_content(self):
        """Test signing and verifying content."""
        asala = Asala()
        
        # Generate keys
        public_key, private_key = asala.generate_key_pair()
        
        # Sign content
        content = b"Test content for verification"
        manifest = asala.sign_content(content, private_key, "Test Creator")
        
        assert manifest is not None
        assert manifest.content_hash is not None
        assert manifest.created_by == "Test Creator"
        assert len(manifest.signatures) > 0
        
        # Verify content
        result = asala.verify(content, manifest)
        
        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence > 90
        assert len(result.errors) == 0

    def test_verify_tampered_content(self):
        """Test verification of tampered content."""
        asala = Asala()
        
        # Generate keys
        public_key, private_key = asala.generate_key_pair()
        
        # Sign original content
        original_content = b"Original content"
        manifest = asala.sign_content(original_content, private_key, "Test Creator")
        
        # Tamper with content
        tampered_content = b"Modified content"
        
        # Verify tampered content
        result = asala.verify(tampered_content, manifest)
        
        assert result.status != VerificationStatus.VERIFIED
        assert result.confidence < 50
        assert len(result.errors) > 0

    def test_verify_without_manifest(self):
        """Test verification without manifest."""
        asala = Asala()
        
        content = b"Content without manifest"
        result = asala.verify(content, None)
        
        assert result.status == VerificationStatus.MISSING_PROVENANCE
        assert result.confidence == 0

    def test_key_pair_uniqueness(self):
        """Test that generated key pairs are unique."""
        asala = Asala()
        
        # Generate multiple key pairs
        pk1, sk1 = asala.generate_key_pair()
        pk2, sk2 = asala.generate_key_pair()
        
        assert pk1 != pk2
        assert sk1 != sk2

    def test_invalid_signature_detection(self):
        """Test detection of invalid signatures."""
        asala = Asala()
        
        # Generate keys
        public_key, private_key = asala.generate_key_pair()
        
        # Sign content
        content = b"Test content"
        manifest = asala.sign_content(content, private_key, "Test Creator")
        
        # Corrupt signature
        if manifest.signatures:
            manifest.signatures[0].signature = "invalid_signature"
        
        # Verify with corrupted signature
        result = asala.verify(content, manifest)
        
        assert result.status != VerificationStatus.VERIFIED
        # Check that signature verification layer failed
        sig_layer = next((layer for layer in result.layers if layer.name == "Signature Verification"), None)
        assert sig_layer is not None
        assert not sig_layer.passed
        assert sig_layer.score == 0.0