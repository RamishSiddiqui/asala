"""Tests for CryptoUtils."""

import pytest

from asala.crypto import CryptoUtils
from asala.types import ChainLink


class TestHashing:
    def test_hash_content_deterministic(self):
        h1 = CryptoUtils.hash_content(b"hello")
        h2 = CryptoUtils.hash_content(b"hello")
        assert h1 == h2

    def test_hash_content_different_inputs(self):
        h1 = CryptoUtils.hash_content(b"hello")
        h2 = CryptoUtils.hash_content(b"world")
        assert h1 != h2

    def test_hash_content_returns_hex(self):
        h = CryptoUtils.hash_content(b"data")
        assert len(h) == 64  # SHA-256 hex = 64 chars
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_string(self):
        h = CryptoUtils.hash_string("test")
        assert len(h) == 64

    def test_hash_empty(self):
        h = CryptoUtils.hash_content(b"")
        assert len(h) == 64


class TestKeyPair:
    def test_generate_key_pair(self, key_pair):
        public_key, private_key = key_pair
        assert "BEGIN PUBLIC KEY" in public_key
        assert "BEGIN PRIVATE KEY" in private_key

    def test_key_pair_uniqueness(self):
        pk1, sk1 = CryptoUtils.generate_key_pair()
        pk2, sk2 = CryptoUtils.generate_key_pair()
        assert pk1 != pk2
        assert sk1 != sk2

    def test_extract_public_key(self, key_pair):
        public_key, private_key = key_pair
        extracted = CryptoUtils.get_public_key_from_private_key(private_key)
        assert extracted == public_key


class TestSigning:
    def test_sign_and_verify(self, key_pair):
        public_key, private_key = key_pair
        content = "message to sign"
        signature = CryptoUtils.sign_content(content, private_key)
        assert CryptoUtils.verify_signature(content, signature, public_key)

    def test_verify_wrong_content(self, key_pair):
        public_key, private_key = key_pair
        signature = CryptoUtils.sign_content("original", private_key)
        assert not CryptoUtils.verify_signature("tampered", signature, public_key)

    def test_verify_invalid_signature(self, key_pair):
        public_key, _ = key_pair
        assert not CryptoUtils.verify_signature("msg", "badsig", public_key)

    def test_verify_wrong_key(self):
        _, sk1 = CryptoUtils.generate_key_pair()
        pk2, _ = CryptoUtils.generate_key_pair()
        sig = CryptoUtils.sign_content("data", sk1)
        assert not CryptoUtils.verify_signature("data", sig, pk2)


class TestChainIntegrity:
    def test_empty_chain(self):
        assert CryptoUtils.verify_chain_integrity([], "hash")

    def test_valid_single_link(self):
        link = ChainLink(
            action="created",
            timestamp="now",
            actor="alice",
            previous_hash="content_hash",
            current_hash="link_hash",
            signature="sig",
        )
        assert CryptoUtils.verify_chain_integrity([link], "content_hash")

    def test_invalid_first_link(self):
        link = ChainLink(
            action="created",
            timestamp="now",
            actor="alice",
            previous_hash="wrong_hash",
            current_hash="link_hash",
            signature="sig",
        )
        assert not CryptoUtils.verify_chain_integrity([link], "content_hash")

    def test_valid_multi_link_chain(self):
        link1 = ChainLink("create", "t1", "alice", "ch", "h1", "s1")
        link2 = ChainLink("edit", "t2", "bob", "h1", "h2", "s2")
        link3 = ChainLink("export", "t3", "carol", "h2", "h3", "s3")
        assert CryptoUtils.verify_chain_integrity([link1, link2, link3], "ch")

    def test_broken_chain(self):
        link1 = ChainLink("create", "t1", "alice", "ch", "h1", "s1")
        link2 = ChainLink("edit", "t2", "bob", "WRONG", "h2", "s2")
        assert not CryptoUtils.verify_chain_integrity([link1, link2], "ch")


class TestCanonicalJson:
    def test_sorted_keys(self):
        j = CryptoUtils.canonical_json({"b": 2, "a": 1})
        assert j == '{"a":1,"b":2}'

    def test_no_spaces(self):
        j = CryptoUtils.canonical_json({"key": "value"})
        assert " " not in j


class TestGenerateUuid:
    def test_format(self):
        uid = CryptoUtils.generate_uuid()
        assert uid.startswith("urn:uuid:")
        assert len(uid) > 10

    def test_uniqueness(self):
        u1 = CryptoUtils.generate_uuid()
        u2 = CryptoUtils.generate_uuid()
        assert u1 != u2
