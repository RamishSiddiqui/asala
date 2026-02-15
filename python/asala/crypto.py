"""Cryptographic utilities for Asala."""
import hashlib
import json
import uuid
from datetime import datetime
from typing import Tuple

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature


class CryptoUtils:
    """Utility class for cryptographic operations."""

    @staticmethod
    def hash_content(content: bytes) -> str:
        """Generate SHA-256 hash of content."""
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def hash_string(data: str) -> str:
        """Generate SHA-256 hash of string."""
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def generate_key_pair() -> Tuple[str, str]:
        """Generate RSA key pair.
        
        Returns:
            Tuple of (public_key_pem, private_key_pem)
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
        
        return public_pem, private_pem

    @staticmethod
    def get_public_key_from_private_key(private_key_pem: str) -> str:
        """Extract public key from private key.
        
        Args:
            private_key_pem: Private key in PEM format
            
        Returns:
            Public key in PEM format
        """
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode(),
            password=None
        )
        
        public_key = private_key.public_key()
        return public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()

    @staticmethod
    def sign_content(content: str, private_key_pem: str) -> str:
        """Sign content with private key.
        
        Args:
            content: Content to sign
            private_key_pem: Private key in PEM format
            
        Returns:
            Base64-encoded signature
        """
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode(),
            password=None
        )
        
        signature = private_key.sign(
            content.encode(),
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        
        import base64
        return base64.b64encode(signature).decode()

    @staticmethod
    def verify_signature(content: str, signature: str, public_key_pem: str) -> bool:
        """Verify signature with public key.
        
        Args:
            content: Original content
            signature: Base64-encoded signature
            public_key_pem: Public key in PEM format
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            public_key = serialization.load_pem_public_key(public_key_pem.encode())
            
            import base64
            signature_bytes = base64.b64decode(signature)
            
            public_key.verify(
                signature_bytes,
                content.encode(),
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            return True
        except (InvalidSignature, Exception):
            return False

    @staticmethod
    def verify_chain_integrity(chain: list, content_hash: str) -> bool:
        """Verify chain of custody integrity.
        
        Args:
            chain: List of chain links
            content_hash: Original content hash
            
        Returns:
            True if chain is valid, False otherwise
        """
        if not chain:
            return True

        # First link must reference content hash
        if chain[0].previous_hash != content_hash:
            return False

        # Each link must reference previous
        for i in range(1, len(chain)):
            if chain[i].previous_hash != chain[i - 1].current_hash:
                return False

        return True

    @staticmethod
    def generate_uuid() -> str:
        """Generate UUID for manifest ID."""
        return f"urn:uuid:{uuid.uuid4()}"

    @staticmethod
    def canonical_json(data: dict) -> str:
        """Create canonical JSON representation for hashing."""
        return json.dumps(data, sort_keys=True, separators=(',', ':'))
