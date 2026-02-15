import { CryptoUtils } from '../crypto/utils';
import { 
  VerificationResult, 
  VerificationStatus, 
  VerificationOptions,
  ContentManifest,
  LayerResult 
} from '../types';

export class ProvenanceVerifier {
  /**
   * Verify content manifest and chain
   */
  verify(manifest: ContentManifest, options: VerificationOptions = {}): VerificationResult {
    const result: VerificationResult = {
      status: VerificationStatus.UNVERIFIED,
      confidence: 0,
      manifest,
      warnings: [],
      errors: [],
      layers: []
    };

    // Layer 1: Signature Verification
    const sigLayer = this.verifySignatures(manifest);
    result.layers.push(sigLayer);

    // Layer 2: Chain Integrity
    const chainLayer = this.verifyChain(manifest);
    result.layers.push(chainLayer);

    // Layer 3: Trust Store (if provided)
    if (options.trustStore && options.trustStore.length > 0) {
      const trustLayer = this.verifyTrust(manifest, options.trustStore);
      result.layers.push(trustLayer);
    }

    // Calculate overall status
    const allPassed = result.layers.every(layer => layer.passed);
    const anyFailed = result.layers.some(layer => !layer.passed);

    if (anyFailed) {
      result.status = VerificationStatus.TAMPERED;
      result.confidence = 0;
    } else if (allPassed) {
      result.status = VerificationStatus.VERIFIED;
      result.confidence = this.calculateConfidence(result.layers);
    }

    return result;
  }

  /**
   * Verify all signatures in the manifest
   */
  private verifySignatures(manifest: ContentManifest): LayerResult {
    const result: LayerResult = {
      name: 'Signature Verification',
      passed: true,
      score: 0,
      details: {
        signaturesChecked: manifest.signatures.length,
        validSignatures: 0
      }
    };

    if (manifest.signatures.length === 0) {
      result.passed = false;
      result.details.error = 'No signatures found';
      return result;
    }

    let validCount = 0;
    for (const signature of manifest.signatures) {
      const isValid = CryptoUtils.verifyManifestSignature(manifest, signature);
      if (isValid) {
        validCount++;
      }
    }

    result.details.validSignatures = validCount;
    result.score = (validCount / manifest.signatures.length) * 100;
    
    if (validCount < manifest.signatures.length) {
      result.passed = false;
      result.details.error = `${manifest.signatures.length - validCount} invalid signatures`;
    }

    return result;
  }

  /**
   * Verify chain of custody integrity
   */
  private verifyChain(manifest: ContentManifest): LayerResult {
    const result: LayerResult = {
      name: 'Chain Integrity',
      passed: true,
      score: 100,
      details: {
        chainLength: manifest.chain.length
      }
    };

    if (manifest.chain.length === 0) {
      // No chain is okay - means content hasn't been edited
      return result;
    }

    const isValid = CryptoUtils.verifyChainIntegrity(manifest.chain, manifest.contentHash);
    
    if (!isValid) {
      result.passed = false;
      result.score = 0;
      result.details.error = 'Chain integrity check failed - possible tampering';
    }

    return result;
  }

  /**
   * Verify signer is in trust store
   */
  private verifyTrust(manifest: ContentManifest, trustStore: string[]): LayerResult {
    const result: LayerResult = {
      name: 'Trust Verification',
      passed: false,
      score: 0,
      details: {
        trustStoreSize: trustStore.length,
        trustedSigners: []
      }
    };

    const trustedSigners = manifest.signatures
      .filter(sig => trustStore.includes(sig.signer) || trustStore.includes(sig.publicKey))
      .map(sig => sig.signer);

    result.details.trustedSigners = trustedSigners;

    if (trustedSigners.length > 0) {
      result.passed = true;
      result.score = 100;
    } else {
      result.details.warning = 'No trusted signers found';
    }

    return result;
  }

  /**
   * Calculate overall confidence score
   */
  private calculateConfidence(layers: LayerResult[]): number {
    if (layers.length === 0) return 0;
    const totalScore = layers.reduce((sum, layer) => sum + layer.score, 0);
    return Math.round(totalScore / layers.length);
  }
}
