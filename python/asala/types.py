"""Type definitions for Asala."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional


class ContentType(Enum):
    """Types of content that can be verified."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"


class VerificationStatus(Enum):
    """Status of content verification."""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    TAMPERED = "tampered"
    INVALID_SIGNATURE = "invalid_signature"
    MISSING_PROVENANCE = "missing_provenance"


@dataclass
class SignatureInfo:
    """Information about a signature."""
    algorithm: str
    public_key: str
    signature: str
    timestamp: str
    signer: str


@dataclass
class Assertion:
    """An assertion about the content."""
    type: str
    data: Dict[str, Any]
    hash: str


@dataclass
class ChainLink:
    """A link in the chain of custody."""
    action: str
    timestamp: str
    actor: str
    previous_hash: str
    current_hash: str
    signature: str


@dataclass
class ContentManifest:
    """Manifest containing provenance information."""
    id: str
    content_hash: str
    content_type: ContentType
    created_at: str
    created_by: str
    signatures: List[SignatureInfo] = field(default_factory=list)
    assertions: List[Assertion] = field(default_factory=list)
    chain: List[ChainLink] = field(default_factory=list)


@dataclass
class LayerResult:
    """Result from a specific verification layer."""
    name: str
    passed: bool
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Complete verification result."""
    status: VerificationStatus
    confidence: int
    manifest: Optional[ContentManifest] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    layers: List[LayerResult] = field(default_factory=list)


@dataclass
class VerificationOptions:
    """Options for verification."""
    include_metadata: bool = True
    include_chain_analysis: bool = True
    include_physics_check: bool = False
    trust_store: List[str] = field(default_factory=list)
