import { CryptoUtils } from './utils';
import { ContentManifest, SignatureInfo, Assertion, ChainLink, ContentType } from '../types';

export class ManifestBuilder {
  private manifest: Partial<ContentManifest> = {};

  constructor(contentHash: string, contentType: ContentType, creator: string) {
    this.manifest = {
      id: this.generateId(),
      contentHash,
      contentType,
      createdAt: new Date().toISOString(),
      createdBy: creator,
      signatures: [],
      assertions: [],
      chain: []
    };
  }

  /**
   * Add an assertion to the manifest
   */
  addAssertion(type: string, data: Record<string, unknown>): this {
    const assertion: Assertion = {
      type,
      data,
      hash: CryptoUtils.hashString(JSON.stringify(data))
    };
    this.manifest.assertions!.push(assertion);
    return this;
  }

  /**
   * Add metadata assertion
   */
  addMetadata(metadata: Record<string, unknown>): this {
    return this.addAssertion('stds.metadata', metadata);
  }

  /**
   * Add creation info
   */
  addCreationInfo(device: string, software: string, location?: { lat: number; lng: number }): this {
    return this.addAssertion('c2pa.created', {
      device,
      software,
      location,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Sign the manifest
   */
  sign(privateKey: string, signer: string): this {
    const manifestData = {
      id: this.manifest.id!,
      contentHash: this.manifest.contentHash!,
      createdAt: this.manifest.createdAt!,
      createdBy: this.manifest.createdBy!,
      assertions: this.manifest.assertions
    };

    const signature = CryptoUtils.signContent(JSON.stringify(manifestData), privateKey);
    
    // Extract public key from private key for verification
    const publicKey = CryptoUtils.getPublicKeyFromPrivateKey(privateKey);
    
    const signatureInfo: SignatureInfo = {
      algorithm: 'RSA-SHA256',
      publicKey,
      signature,
      timestamp: new Date().toISOString(),
      signer
    };

    this.manifest.signatures!.push(signatureInfo);
    return this;
  }

  /**
   * Add chain link for edits/transformations
   */
  addChainLink(action: string, actor: string, privateKey: string): this {
    const previousHash = this.manifest.chain!.length > 0
      ? this.manifest.chain![this.manifest.chain!.length - 1].currentHash
      : this.manifest.contentHash!;

    const linkData = {
      action,
      timestamp: new Date().toISOString(),
      actor,
      previousHash
    };

    const currentHash = CryptoUtils.hashString(JSON.stringify(linkData));
    const signature = CryptoUtils.signContent(JSON.stringify(linkData), privateKey);

    const chainLink: ChainLink = {
      ...linkData,
      currentHash,
      signature
    };

    this.manifest.chain!.push(chainLink);
    return this;
  }

  /**
   * Build the final manifest
   */
  build(): ContentManifest {
    return this.manifest as ContentManifest;
  }

  private generateId(): string {
    return `urn:uuid:${CryptoUtils.hashString(Date.now().toString() + Math.random())}`;
  }
}
