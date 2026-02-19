"""Tests for ManifestBuilder."""

import pytest

from asala.crypto import CryptoUtils
from asala.manifest import ManifestBuilder
from asala.types import ContentType


class TestManifestBuilder:
    def test_build_basic(self):
        mb = ManifestBuilder("hash123", ContentType.IMAGE, "creator")
        manifest = mb.build()

        assert manifest.content_hash == "hash123"
        assert manifest.content_type is ContentType.IMAGE
        assert manifest.created_by == "creator"
        assert manifest.id.startswith("urn:uuid:")
        assert manifest.created_at  # non-empty
        assert manifest.signatures == []

    def test_add_assertion(self):
        mb = ManifestBuilder("h", ContentType.DOCUMENT, "c")
        mb.add_assertion("test.type", {"key": "value"})
        manifest = mb.build()

        assert len(manifest.assertions) == 1
        assert manifest.assertions[0].type == "test.type"
        assert manifest.assertions[0].data["key"] == "value"
        assert manifest.assertions[0].hash  # non-empty hash

    def test_add_metadata(self):
        mb = ManifestBuilder("h", ContentType.AUDIO, "c")
        mb.add_metadata({"format": "wav", "duration": 5.0})
        manifest = mb.build()

        assert len(manifest.assertions) == 1
        assert manifest.assertions[0].type == "stds.metadata"

    def test_add_creation_info(self):
        mb = ManifestBuilder("h", ContentType.VIDEO, "c")
        mb.add_creation_info("iPhone 15", "asala/1.0")
        manifest = mb.build()

        assert len(manifest.assertions) == 1
        assert manifest.assertions[0].type == "c2pa.created"
        assert manifest.assertions[0].data["device"] == "iPhone 15"
        assert manifest.assertions[0].data["software"] == "asala/1.0"

    def test_add_creation_info_with_location(self):
        mb = ManifestBuilder("h", ContentType.IMAGE, "c")
        mb.add_creation_info("cam", "sw", location={"lat": 40.7, "lon": -74.0})
        manifest = mb.build()
        assert manifest.assertions[0].data["location"]["lat"] == 40.7

    def test_sign(self, key_pair):
        _, private_key = key_pair
        mb = ManifestBuilder("h", ContentType.IMAGE, "alice")
        mb.sign(private_key, "alice")
        manifest = mb.build()

        assert len(manifest.signatures) == 1
        sig = manifest.signatures[0]
        assert sig.algorithm == "RSA-SHA256"
        assert sig.signer == "alice"
        assert sig.signature  # non-empty
        assert sig.public_key  # non-empty
        assert sig.timestamp  # non-empty

    def test_sign_produces_valid_signature(self, key_pair):
        public_key, private_key = key_pair
        mb = ManifestBuilder("h", ContentType.IMAGE, "alice")
        mb.sign(private_key, "alice")
        manifest = mb.build()

        sig = manifest.signatures[0]
        # Reconstruct the signed data
        manifest_data = {
            "id": manifest.id,
            "content_hash": manifest.content_hash,
            "created_at": manifest.created_at,
            "created_by": manifest.created_by,
            "assertions": [
                {"type": a.type, "data": a.data} for a in manifest.assertions
            ],
        }
        canonical = CryptoUtils.canonical_json(manifest_data)
        assert CryptoUtils.verify_signature(canonical, sig.signature, sig.public_key)

    def test_multiple_signatures(self):
        _, sk1 = CryptoUtils.generate_key_pair()
        _, sk2 = CryptoUtils.generate_key_pair()
        mb = ManifestBuilder("h", ContentType.IMAGE, "c")
        mb.sign(sk1, "signer1")
        mb.sign(sk2, "signer2")
        manifest = mb.build()
        assert len(manifest.signatures) == 2

    def test_add_chain_link(self, key_pair):
        _, private_key = key_pair
        mb = ManifestBuilder("h", ContentType.IMAGE, "c")
        mb.add_chain_link("created", "alice", private_key)
        manifest = mb.build()

        assert len(manifest.chain) == 1
        link = manifest.chain[0]
        assert link.action == "created"
        assert link.actor == "alice"
        assert link.previous_hash == "h"  # first link references content hash
        assert link.current_hash  # non-empty
        assert link.signature  # non-empty

    def test_chain_link_ordering(self, key_pair):
        _, private_key = key_pair
        mb = ManifestBuilder("h", ContentType.IMAGE, "c")
        mb.add_chain_link("created", "alice", private_key)
        mb.add_chain_link("edited", "bob", private_key)
        manifest = mb.build()

        assert len(manifest.chain) == 2
        assert manifest.chain[0].previous_hash == "h"
        assert manifest.chain[1].previous_hash == manifest.chain[0].current_hash

    def test_builder_chaining(self, key_pair):
        _, private_key = key_pair
        manifest = (
            ManifestBuilder("h", ContentType.IMAGE, "c")
            .add_creation_info("device", "sw")
            .add_metadata({"format": "jpeg"})
            .sign(private_key, "signer")
            .build()
        )
        assert len(manifest.assertions) == 2
        assert len(manifest.signatures) == 1
