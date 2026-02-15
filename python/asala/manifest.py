"""Manifest builder for creating content provenance."""
from datetime import datetime
from typing import Dict, Any, List, Optional

from .crypto import CryptoUtils
from .types import (
    ContentManifest,
    SignatureInfo,
    Assertion,
    ChainLink,
    ContentType,
)


class ManifestBuilder:
    """Builder for creating content manifests."""

    def __init__(
        self,
        content_hash: str,
        content_type: ContentType,
        creator: str
    ):
        """Initialize manifest builder.
        
        Args:
            content_hash: Hash of the content
            content_type: Type of content
            creator: Creator identifier
        """
        self._id = CryptoUtils.generate_uuid()
        self._content_hash = content_hash
        self._content_type = content_type
        self._created_at = datetime.utcnow().isoformat()
        self._created_by = creator
        self._signatures: List[SignatureInfo] = []
        self._assertions: List[Assertion] = []
        self._chain: List[ChainLink] = []

    def add_assertion(self, assertion_type: str, data: Dict[str, Any]) -> "ManifestBuilder":
        """Add an assertion to the manifest.
        
        Args:
            assertion_type: Type of assertion
            data: Assertion data
            
        Returns:
            Self for chaining
        """
        assertion = Assertion(
            type=assertion_type,
            data=data,
            hash=CryptoUtils.hash_string(CryptoUtils.canonical_json(data))
        )
        self._assertions.append(assertion)
        return self

    def add_metadata(self, metadata: Dict[str, Any]) -> "ManifestBuilder":
        """Add metadata assertion.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Self for chaining
        """
        return self.add_assertion("stds.metadata", metadata)

    def add_creation_info(
        self,
        device: str,
        software: str,
        location: Optional[Dict[str, float]] = None
    ) -> "ManifestBuilder":
        """Add creation information.
        
        Args:
            device: Device name
            software: Software used
            location: Optional location coordinates
            
        Returns:
            Self for chaining
        """
        return self.add_assertion("c2pa.created", {
            "device": device,
            "software": software,
            "location": location,
            "timestamp": datetime.utcnow().isoformat()
        })

    def sign(self, private_key: str, signer: str) -> "ManifestBuilder":
        """Sign the manifest.
        
        Args:
            private_key: Private key in PEM format
            signer: Signer identifier
            
        Returns:
            Self for chaining
        """
        manifest_data = {
            "id": self._id,
            "content_hash": self._content_hash,
            "created_at": self._created_at,
            "created_by": self._created_by,
            "assertions": [
                {"type": a.type, "data": a.data}
                for a in self._assertions
            ]
        }
        
        signature = CryptoUtils.sign_content(
            CryptoUtils.canonical_json(manifest_data),
            private_key
        )
        
        # Extract public key from private key for verification
        public_key = CryptoUtils.get_public_key_from_private_key(private_key)
        
        signature_info = SignatureInfo(
            algorithm="RSA-SHA256",
            public_key=public_key,
            signature=signature,
            timestamp=datetime.utcnow().isoformat(),
            signer=signer
        )
        
        self._signatures.append(signature_info)
        return self

    def add_chain_link(
        self,
        action: str,
        actor: str,
        private_key: str
    ) -> "ManifestBuilder":
        """Add a chain link for edits/transformations.
        
        Args:
            action: Action performed
            actor: Actor performing the action
            private_key: Private key for signing
            
        Returns:
            Self for chaining
        """
        previous_hash = (
            self._chain[-1].current_hash
            if self._chain
            else self._content_hash
        )
        
        link_data = {
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
            "actor": actor,
            "previous_hash": previous_hash
        }
        
        current_hash = CryptoUtils.hash_string(CryptoUtils.canonical_json(link_data))
        signature = CryptoUtils.sign_content(
            CryptoUtils.canonical_json(link_data),
            private_key
        )
        
        chain_link = ChainLink(
            action=action,
            timestamp=link_data["timestamp"],
            actor=actor,
            previous_hash=previous_hash,
            current_hash=current_hash,
            signature=signature
        )
        
        self._chain.append(chain_link)
        return self

    def build(self) -> ContentManifest:
        """Build the final manifest.
        
        Returns:
            Complete ContentManifest
        """
        return ContentManifest(
            id=self._id,
            content_hash=self._content_hash,
            content_type=self._content_type,
            created_at=self._created_at,
            created_by=self._created_by,
            signatures=self._signatures,
            assertions=self._assertions,
            chain=self._chain
        )
