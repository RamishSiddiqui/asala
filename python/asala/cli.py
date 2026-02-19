"""Command-line interface for Asala."""
import argparse
import json
import sys
from pathlib import Path

from .verify import Asala
from .crypto import CryptoUtils
from .types import VerificationOptions


def verify_command(args):
    """Verify content command."""
    asala = Asala(max_workers=getattr(args, "workers", 1))
    
    content_path = Path(args.file)
    if not content_path.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    content = content_path.read_bytes()
    
    # Load manifest if provided
    manifest = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"Error: Manifest not found: {args.manifest}", file=sys.stderr)
            sys.exit(1)
        
        from .types import ContentManifest, ContentType, SignatureInfo, Assertion, ChainLink
        
        manifest_data = json.loads(manifest_path.read_text())
        # Parse manifest from JSON
        manifest = ContentManifest(
            id=manifest_data["id"],
            content_hash=manifest_data["content_hash"],
            content_type=ContentType(manifest_data["content_type"]),
            created_at=manifest_data["created_at"],
            created_by=manifest_data["created_by"],
            signatures=[
                SignatureInfo(**sig) for sig in manifest_data.get("signatures", [])
            ],
            assertions=[
                Assertion(**assertion) for assertion in manifest_data.get("assertions", [])
            ],
            chain=[
                ChainLink(**link) for link in manifest_data.get("chain", [])
            ]
        )
    
    # Set up verification options
    options = VerificationOptions(
        include_metadata=True,
        include_chain_analysis=True,
        include_physics_check=args.physics or (manifest is None),  # Enable physics for unsigned content
        trust_store=args.trust if args.trust else []
    )
    
    result = asala.verify(content, manifest, options)
    
    if args.json:
        print(json.dumps({
            "status": result.status.value,
            "confidence": result.confidence,
            "warnings": result.warnings,
            "errors": result.errors,
            "layers": [
                {
                    "name": layer.name,
                    "passed": layer.passed,
                    "score": layer.score
                }
                for layer in result.layers
            ]
        }, indent=2))
    else:
        print(f"\n{'='*60}")
        print("  Content Verification Report")
        print(f"{'='*60}\n")
        print(f"File: {content_path.resolve()}")
        print(f"Status: {result.status.value.upper()}")
        print(f"Confidence: {result.confidence}%")
        
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  • {warning}")
        
        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  • {error}")
        
        if result.layers:
            print("\nVerification Layers:")
            for layer in result.layers:
                icon = "✓" if layer.passed else "✗"
                print(f"  {icon} {layer.name}: {layer.score:.0f}%")
        
        print(f"\n{'='*60}\n")
    
    sys.exit(0 if result.status.value == "verified" else 1)


def sign_command(args):
    """Sign content command."""
    asala = Asala()
    
    content_path = Path(args.file)
    if not content_path.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    
    key_path = Path(args.key)
    if not key_path.exists():
        print(f"Error: Private key not found: {args.key}", file=sys.stderr)
        sys.exit(1)
    
    content = content_path.read_bytes()
    private_key = key_path.read_text()
    creator = args.creator or "Unknown"
    device = args.device or "unknown-device"

    manifest = asala.sign_content(content, private_key, creator, device=device)
    
    # Determine output path
    output_path = args.output or f"{args.file}.manifest.json"
    
    # Save manifest
    manifest_dict = {
        "id": manifest.id,
        "content_hash": manifest.content_hash,
        "content_type": manifest.content_type.value,
        "created_at": manifest.created_at,
        "created_by": manifest.created_by,
        "signatures": [
            {
                "algorithm": sig.algorithm,
                "public_key": sig.public_key,
                "signature": sig.signature,
                "timestamp": sig.timestamp,
                "signer": sig.signer
            }
            for sig in manifest.signatures
        ],
        "assertions": [
            {
                "type": assertion.type,
                "data": assertion.data,
                "hash": assertion.hash
            }
            for assertion in manifest.assertions
        ],
        "chain": [
            {
                "action": link.action,
                "timestamp": link.timestamp,
                "actor": link.actor,
                "previous_hash": link.previous_hash,
                "current_hash": link.current_hash,
                "signature": link.signature
            }
            for link in manifest.chain
        ]
    }
    
    Path(output_path).write_text(json.dumps(manifest_dict, indent=2))
    
    print(f"\n✓ Content signed successfully!\n")
    print(f"Manifest saved to: {Path(output_path).resolve()}")
    print(f"Content hash: {manifest.content_hash}")
    print(f"Signatures: {len(manifest.signatures)}")


def keys_command(args):
    """Generate keys command."""
    asala = Asala()
    
    if args.generate:
        print("Generating key pair...")
        
        public_key, private_key = asala.generate_key_pair()
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        private_path = output_dir / "private.pem"
        public_path = output_dir / "public.pem"
        
        private_path.write_text(private_key)
        public_path.write_text(public_key)
        
        print(f"\n✓ Key pair generated successfully!\n")
        print(f"Private key: {private_path}")
        print(f"Public key: {public_path}")
        print("\n⚠ Important: Keep your private key secure and never share it!")
    else:
        print("Asala - Key Management\n")
        print("Generate a new key pair:")
        print("  asala keys --generate\n")
        print("Options:")
        print("  -g, --generate    Generate new key pair")
        print("  -o, --output      Output directory (default: ./keys)\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="asala",
        description="CLI tool for content authenticity verification"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify content authenticity")
    verify_parser.add_argument("file", help="File to verify")
    verify_parser.add_argument("-m", "--manifest", help="Path to manifest file")
    verify_parser.add_argument("-t", "--trust", nargs="+", help="Trusted public keys")
    verify_parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    verify_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    verify_parser.add_argument("-p", "--physics", action="store_true", help="Enable physics-based verification")
    verify_parser.add_argument("-w", "--workers", type=int, default=1, help="Number of parallel threads for analysis (default: 1)")
    verify_parser.set_defaults(func=verify_command)
    
    # Sign command
    sign_parser = subparsers.add_parser("sign", help="Sign content")
    sign_parser.add_argument("file", help="File to sign")
    sign_parser.add_argument("-k", "--key", required=True, help="Path to private key")
    sign_parser.add_argument("-o", "--output", help="Output file path")
    sign_parser.add_argument("-c", "--creator", help="Creator name")
    sign_parser.add_argument("-d", "--device", help="Device name")
    sign_parser.set_defaults(func=sign_command)
    
    # Keys command
    keys_parser = subparsers.add_parser("keys", help="Manage cryptographic keys")
    keys_parser.add_argument("-g", "--generate", action="store_true", help="Generate new key pair")
    keys_parser.add_argument("-o", "--output", default="./keys", help="Output directory")
    keys_parser.set_defaults(func=keys_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
