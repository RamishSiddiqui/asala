import { createHash, createVerify, createSign, generateKeyPairSync, createPublicKey } from 'crypto';
import { SignatureInfo, ContentManifest } from '../types';

export class CryptoUtils {
  /**
   * Generate SHA-256 hash of content
   */
  static hashContent(content: Buffer): string {
    return createHash('sha256').update(content).digest('hex');
  }

  /**
   * Generate SHA-256 hash of string
   */
  static hashString(data: string): string {
    return createHash('sha256').update(data).digest('hex');
  }

  /**
   * Sign content with private key (RSA-SHA256)
   */
  static signContent(content: string, privateKey: string): string {
    const sign = createSign('RSA-SHA256');
    sign.update(content);
    sign.end();
    return sign.sign(privateKey, 'base64');
  }

  /**
   * Verify signature with public key
   */
  static verifySignature(content: string, signature: string, publicKey: string): boolean {
    try {
      const verify = createVerify('RSA-SHA256');
      verify.update(content);
      verify.end();
      return verify.verify(publicKey, signature, 'base64');
    } catch {
      return false;
    }
  }

  /**
   * Verify manifest signature
   */
  static verifyManifestSignature(manifest: ContentManifest, signature: SignatureInfo): boolean {
    const manifestString = JSON.stringify({
      id: manifest.id,
      contentHash: manifest.contentHash,
      createdAt: manifest.createdAt,
      createdBy: manifest.createdBy,
      assertions: manifest.assertions
    });
    return this.verifySignature(manifestString, signature.signature, signature.publicKey);
  }

  /**
   * Generate RSA key pair
   */
  static generateKeyPair(): { publicKey: string; privateKey: string } {
    const { publicKey, privateKey } = generateKeyPairSync('rsa', {
      modulusLength: 2048,
      publicKeyEncoding: { type: 'spki', format: 'pem' },
      privateKeyEncoding: { type: 'pkcs8', format: 'pem' }
    });
    return { publicKey, privateKey };
  }

  /**
   * Extract public key from private key
   */
  static getPublicKeyFromPrivateKey(privateKey: string): string {
    const key = createPublicKey(privateKey);
    return key.export({ type: 'spki', format: 'pem' }).toString();
  }

  /**
   * Create content manifest hash
   */
  static createManifestHash(manifest: Omit<ContentManifest, 'signatures'>): string {
    const canonical = JSON.stringify({
      id: manifest.id,
      contentHash: manifest.contentHash,
      contentType: manifest.contentType,
      createdAt: manifest.createdAt,
      createdBy: manifest.createdBy,
      assertions: manifest.assertions
    });
    return this.hashString(canonical);
  }

  /**
   * Verify chain integrity
   */
  static verifyChainIntegrity(chain: ContentManifest['chain'], contentHash: string): boolean {
    if (chain.length === 0) return true;

    // First link must match content hash
    if (chain[0].previousHash !== contentHash) {
      return false;
    }

    // Each link must reference previous
    for (let i = 1; i < chain.length; i++) {
      if (chain[i].previousHash !== chain[i - 1].currentHash) {
        return false;
      }
    }

    return true;
  }
}
