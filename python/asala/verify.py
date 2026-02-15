"""Main Asala implementation.

Asala (أصالة) means "Authenticity" in Arabic.
Cryptographic Content Authenticity Verification Library
"""
import io
from typing import Optional, Tuple

from .crypto import CryptoUtils
from .manifest import ManifestBuilder
from .types import (
    ContentManifest,
    ContentType,
    VerificationResult,
    VerificationStatus,
    LayerResult,
    VerificationOptions,
)


class Asala:
    """Main class for content verification."""

    def __init__(self):
        """Initialize Asala instance."""
        pass

    def verify(
        self,
        content: bytes,
        manifest: Optional[ContentManifest] = None,
        options: Optional[VerificationOptions] = None
    ) -> VerificationResult:
        """Verify content authenticity.
        
        Args:
            content: Content bytes to verify
            manifest: Optional manifest to verify against
            options: Verification options
            
        Returns:
            VerificationResult with status and details
        """
        if options is None:
            options = VerificationOptions()

        if manifest:
            return self._verify_with_manifest(content, manifest, options)
        
        return self._verify_without_manifest(content, options)

    def _verify_with_manifest(
        self,
        content: bytes,
        manifest: ContentManifest,
        options: VerificationOptions
    ) -> VerificationResult:
        """Verify content against a manifest."""
        layers = []
        warnings = []
        errors = []

        # Layer 0: Content Hash Verification
        content_hash = CryptoUtils.hash_content(content)
        if content_hash != manifest.content_hash:
            return VerificationResult(
                status=VerificationStatus.TAMPERED,
                confidence=0,
                manifest=manifest,
                warnings=[],
                errors=["Content hash does not match manifest"],
                layers=[
                    LayerResult(
                        name="Content Hash Verification",
                        passed=False,
                        score=0.0,
                        details={"expected": manifest.content_hash, "actual": content_hash}
                    )
                ]
            )

        layers.append(
            LayerResult(
                name="Content Hash Verification",
                passed=True,
                score=100.0,
                details={"hash": content_hash}
            )
        )

        # Layer 1: Signature Verification
        sig_layer = self._verify_signatures(manifest)
        layers.append(sig_layer)

        # Layer 2: Chain Integrity
        chain_layer = self._verify_chain(manifest)
        layers.append(chain_layer)

        # Layer 3: Trust Store (if provided)
        if options.trust_store:
            trust_layer = self._verify_trust(manifest, options.trust_store)
            layers.append(trust_layer)

        # Calculate overall status
        all_passed = all(layer.passed for layer in layers)
        any_failed = any(not layer.passed for layer in layers)

        if any_failed:
            status = VerificationStatus.TAMPERED
            confidence = 0
        elif all_passed:
            status = VerificationStatus.VERIFIED
            confidence = self._calculate_confidence(layers)
        else:
            status = VerificationStatus.UNVERIFIED
            confidence = 0

        return VerificationResult(
            status=status,
            confidence=confidence,
            manifest=manifest,
            warnings=warnings,
            errors=errors,
            layers=layers
        )

    def _verify_without_manifest(
        self,
        content: bytes,
        options: VerificationOptions
    ) -> VerificationResult:
        """Verify content without a manifest."""
        return VerificationResult(
            status=VerificationStatus.MISSING_PROVENANCE,
            confidence=0,
            warnings=["No embedded manifest found"],
            errors=["Content lacks C2PA provenance data"],
            layers=[]
        )

    def _verify_signatures(self, manifest: ContentManifest) -> LayerResult:
        """Verify all signatures in the manifest."""
        if not manifest.signatures:
            return LayerResult(
                name="Signature Verification",
                passed=False,
                score=0.0,
                details={"error": "No signatures found"}
            )

        valid_count = 0
        for sig in manifest.signatures:
            manifest_data = {
                "id": manifest.id,
                "content_hash": manifest.content_hash,
                "created_at": manifest.created_at,
                "created_by": manifest.created_by,
                "assertions": [{"type": a.type, "data": a.data} for a in manifest.assertions]
            }
            
            is_valid = CryptoUtils.verify_signature(
                CryptoUtils.canonical_json(manifest_data),
                sig.signature,
                sig.public_key
            )
            if is_valid:
                valid_count += 1

        score = (valid_count / len(manifest.signatures)) * 100
        
        return LayerResult(
            name="Signature Verification",
            passed=valid_count == len(manifest.signatures),
            score=score,
            details={
                "signatures_checked": len(manifest.signatures),
                "valid_signatures": valid_count
            }
        )

    def _verify_chain(self, manifest: ContentManifest) -> LayerResult:
        """Verify chain of custody integrity."""
        if not manifest.chain:
            return LayerResult(
                name="Chain Integrity",
                passed=True,
                score=100.0,
                details={"chain_length": 0}
            )

        is_valid = CryptoUtils.verify_chain_integrity(
            manifest.chain,
            manifest.content_hash
        )

        return LayerResult(
            name="Chain Integrity",
            passed=is_valid,
            score=100.0 if is_valid else 0.0,
            details={
                "chain_length": len(manifest.chain),
                "error": None if is_valid else "Chain integrity check failed"
            }
        )

    def _verify_trust(
        self,
        manifest: ContentManifest,
        trust_store: list
    ) -> LayerResult:
        """Verify signer is in trust store."""
        trusted_signers = [
            sig.signer for sig in manifest.signatures
            if sig.signer in trust_store or sig.public_key in trust_store
        ]

        return LayerResult(
            name="Trust Verification",
            passed=len(trusted_signers) > 0,
            score=100.0 if trusted_signers else 0.0,
            details={
                "trust_store_size": len(trust_store),
                "trusted_signers": trusted_signers
            }
        )

    def _calculate_confidence(self, layers: list) -> int:
        """Calculate overall confidence score."""
        if not layers:
            return 0
        total_score = sum(layer.score for layer in layers)
        return int(round(total_score / len(layers)))

    def sign_content(
        self,
        content: bytes,
        private_key: str,
        creator: str,
        content_type: Optional[ContentType] = None
    ) -> ContentManifest:
        """Sign content and create manifest.
        
        Args:
            content: Content to sign
            private_key: Private key in PEM format
            creator: Creator identifier
            content_type: Optional content type override
            
        Returns:
            Signed ContentManifest
        """
        content_hash = CryptoUtils.hash_content(content)
        detected_type = content_type or self._detect_content_type(content)

        manifest = ManifestBuilder(content_hash, detected_type, creator)
        manifest.add_creation_info("unknown-device", "asala/0.0.1")
        manifest.sign(private_key, creator)

        return manifest.build()

    def generate_key_pair(self) -> Tuple[str, str]:
        """Generate new key pair for signing.
        
        Returns:
            Tuple of (public_key, private_key) in PEM format
        """
        return CryptoUtils.generate_key_pair()

    def _detect_content_type(self, content: bytes) -> ContentType:
        """Detect content type from magic numbers."""
        if len(content) < 2:
            return ContentType.DOCUMENT

        # JPEG
        if content[0:2] == b'\xff\xd8':
            return ContentType.IMAGE

        # PNG
        if content[0:4] == b'\x89PNG':
            return ContentType.IMAGE

        # GIF
        if content[0:4] == b'GIF8':
            return ContentType.VIDEO

        # MP3 (ID3)
        if content[0:3] == b'ID3':
            return ContentType.AUDIO

        return ContentType.DOCUMENT
