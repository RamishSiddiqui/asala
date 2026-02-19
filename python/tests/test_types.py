"""Tests for Asala type definitions."""

import pytest

from asala.types import (
    Assertion,
    ChainLink,
    ContentManifest,
    ContentType,
    LayerResult,
    SignatureInfo,
    VerificationOptions,
    VerificationResult,
    VerificationStatus,
)


class TestContentType:
    def test_values(self):
        assert ContentType.IMAGE.value == "image"
        assert ContentType.VIDEO.value == "video"
        assert ContentType.AUDIO.value == "audio"
        assert ContentType.DOCUMENT.value == "document"

    def test_from_value(self):
        assert ContentType("image") is ContentType.IMAGE
        assert ContentType("audio") is ContentType.AUDIO

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            ContentType("unknown")


class TestVerificationStatus:
    def test_all_statuses_exist(self):
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.UNVERIFIED.value == "unverified"
        assert VerificationStatus.TAMPERED.value == "tampered"
        assert VerificationStatus.INVALID_SIGNATURE.value == "invalid_signature"
        assert VerificationStatus.MISSING_PROVENANCE.value == "missing_provenance"


class TestSignatureInfo:
    def test_creation(self):
        sig = SignatureInfo(
            algorithm="RSA-SHA256",
            public_key="pk",
            signature="sig",
            timestamp="2024-01-01",
            signer="alice",
        )
        assert sig.algorithm == "RSA-SHA256"
        assert sig.signer == "alice"


class TestAssertion:
    def test_creation(self):
        a = Assertion(type="c2pa.created", data={"device": "cam"}, hash="abc")
        assert a.type == "c2pa.created"
        assert a.data["device"] == "cam"


class TestChainLink:
    def test_creation(self):
        link = ChainLink(
            action="created",
            timestamp="2024-01-01",
            actor="bob",
            previous_hash="aaa",
            current_hash="bbb",
            signature="sig",
        )
        assert link.action == "created"
        assert link.previous_hash == "aaa"


class TestContentManifest:
    def test_defaults(self):
        m = ContentManifest(
            id="urn:uuid:test",
            content_hash="hash",
            content_type=ContentType.IMAGE,
            created_at="2024-01-01",
            created_by="tester",
        )
        assert m.signatures == []
        assert m.assertions == []
        assert m.chain == []

    def test_with_signatures(self):
        sig = SignatureInfo("RSA", "pk", "sig", "now", "alice")
        m = ContentManifest(
            id="id",
            content_hash="h",
            content_type=ContentType.AUDIO,
            created_at="now",
            created_by="alice",
            signatures=[sig],
        )
        assert len(m.signatures) == 1
        assert m.content_type is ContentType.AUDIO


class TestLayerResult:
    def test_defaults(self):
        r = LayerResult(name="test", passed=True, score=95.0)
        assert r.details == {}

    def test_with_details(self):
        r = LayerResult(name="x", passed=False, score=0, details={"error": "bad"})
        assert r.details["error"] == "bad"


class TestVerificationResult:
    def test_defaults(self):
        r = VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=100,
        )
        assert r.manifest is None
        assert r.warnings == []
        assert r.errors == []
        assert r.layers == []


class TestVerificationOptions:
    def test_defaults(self):
        opts = VerificationOptions()
        assert opts.include_metadata is True
        assert opts.include_chain_analysis is True
        assert opts.include_physics_check is False
        assert opts.trust_store == []
