"""
Asala - Python Implementation

Asala (أصالة) means "Authenticity" in Arabic.
Cryptographic Content Authenticity Verification Library
"""

from .verify import Asala
from .types import (
    ContentManifest,
    SignatureInfo,
    Assertion,
    ChainLink,
    ContentType,
    VerificationStatus,
    VerificationResult,
    LayerResult,
)
from .crypto import CryptoUtils
from .manifest import ManifestBuilder

__version__ = "0.0.1"
__all__ = [
    "Asala",
    "ContentManifest",
    "SignatureInfo",
    "Assertion",
    "ChainLink",
    "ContentType",
    "VerificationStatus",
    "VerificationResult",
    "LayerResult",
    "CryptoUtils",
    "ManifestBuilder",
]
