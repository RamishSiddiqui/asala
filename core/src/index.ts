export * from './types';
export * from './crypto/utils';
export * from './crypto/manifest';
export * from './verifiers/provenance';
export * from './verifiers/physics';
export * from './verifiers/audio';
export * from './verifiers/video';

import { CryptoUtils } from './crypto/utils';
import { ManifestBuilder } from './crypto/manifest';
import { ProvenanceVerifier } from './verifiers/provenance';
import { PhysicsVerifier } from './verifiers/physics';
import { VideoVerifier } from './verifiers/video';
import {
  VerificationResult,
  ContentManifest,
  VerificationOptions,
  ContentType,
  VerificationStatus
} from './types';

/**
 * Main Asala class
 * Provides easy-to-use interface for content verification
 */
export class Asala {
  private provenanceVerifier: ProvenanceVerifier;
  private physicsVerifier: PhysicsVerifier;
  private videoVerifier: VideoVerifier;

  constructor() {
    this.provenanceVerifier = new ProvenanceVerifier();
    this.physicsVerifier = new PhysicsVerifier();
    this.videoVerifier = new VideoVerifier();
  }

  /**
   * Verify content from a buffer
   * @param content - The content buffer
   * @param manifest - Optional manifest to verify against
   * @param options - Verification options
   * @returns Verification result
   */
  async verify(
    content: Buffer,
    manifest?: ContentManifest,
    options: VerificationOptions = {}
  ): Promise<VerificationResult> {
    // If manifest provided, verify it
    if (manifest) {
      // First verify content hash matches
      const contentHash = CryptoUtils.hashContent(content);
      if (contentHash !== manifest.contentHash) {
        return {
          status: VerificationStatus.TAMPERED,
          confidence: 0,
          manifest,
          warnings: [],
          errors: ['Content hash does not match manifest'],
          layers: [{
            name: 'Content Hash Verification',
            passed: false,
            score: 0,
            details: { expected: manifest.contentHash, actual: contentHash }
          }]
        };
      }

      const result = this.provenanceVerifier.verify(manifest, options);

      // Add content hash verification layer
      result.layers.unshift({
        name: 'Content Hash Verification',
        passed: true,
        score: 100,
        details: { hash: contentHash }
      });

      // Also add physics verification
      const contentType = this.detectContentType(content);
      let physicsLayer;

      if (contentType === ContentType.IMAGE) {
        physicsLayer = this.physicsVerifier.verifyImage(content);
      } else if (contentType === ContentType.AUDIO) {
        physicsLayer = this.physicsVerifier.verifyAudio(content);
      }

      if (physicsLayer) {
        result.layers.push(physicsLayer);
      }

      return result;
    }

    // No manifest - try to extract from content (if C2PA embedded)
    return this.verifyEmbedded(content, options);
  }

  /**
   * Sign content and create manifest
   * @param content - Content to sign
   * @param privateKey - Signer's private key
   * @param creator - Creator identifier
   * @returns Signed manifest
   */
  signContent(
    content: Buffer,
    privateKey: string,
    creator: string,
    contentType?: ContentType
  ): ContentManifest {
    const hash = CryptoUtils.hashContent(content);
    const type = contentType || this.detectContentType(content);

    const manifest = new ManifestBuilder(hash, type, creator)
      .addCreationInfo('unknown-device', 'asala/0.0.1')
      .sign(privateKey, creator)
      .build();

    return manifest;
  }

  /**
   * Generate new key pair for signing
   */
  generateKeyPair(): { publicKey: string; privateKey: string } {
    return CryptoUtils.generateKeyPair();
  }

  /**
   * Verify embedded C2PA manifest in content
   */
  private async verifyEmbedded(
    content: Buffer,
    options: VerificationOptions
  ): Promise<VerificationResult> {
    // TODO: Implement C2PA manifest extraction
    // For now, return unverified
    return {
      status: VerificationStatus.MISSING_PROVENANCE,
      confidence: 0,
      warnings: ['No embedded manifest found'],
      errors: ['Content lacks C2PA provenance data'],
      layers: []
    };
  }

  /**
   * Detect content type from buffer
   */
  private detectContentType(buffer: Buffer): ContentType {
    // Check magic numbers
    if (buffer[0] === 0xFF && buffer[1] === 0xD8) {
      return ContentType.IMAGE; // JPEG
    }
    if (buffer[0] === 0x89 && buffer[1] === 0x50) {
      return ContentType.IMAGE; // PNG
    }
    if (buffer[0] === 0x47 && buffer[1] === 0x49) {
      return ContentType.VIDEO; // GIF
    }
    if (buffer[0] === 0x49 && buffer[1] === 0x44) {
      return ContentType.AUDIO; // MP3 (ID3)
    }

    return ContentType.DOCUMENT;
  }
}

/** @deprecated Use Asala instead */


export default Asala;
