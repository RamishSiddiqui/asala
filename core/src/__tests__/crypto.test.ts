import { CryptoUtils } from '../crypto/utils';
import { ManifestBuilder } from '../crypto/manifest';
import { ContentType, VerificationStatus } from '../types';

describe('CryptoUtils', () => {
  describe('hashContent', () => {
    it('should generate consistent SHA-256 hashes', () => {
      const content = Buffer.from('test content');
      const hash1 = CryptoUtils.hashContent(content);
      const hash2 = CryptoUtils.hashContent(content);
      
      expect(hash1).toBe(hash2);
      expect(hash1).toHaveLength(64); // SHA-256 hex length
    });

    it('should generate different hashes for different content', () => {
      const content1 = Buffer.from('content 1');
      const content2 = Buffer.from('content 2');
      
      const hash1 = CryptoUtils.hashContent(content1);
      const hash2 = CryptoUtils.hashContent(content2);
      
      expect(hash1).not.toBe(hash2);
    });
  });

  describe('hashString', () => {
    it('should generate consistent hashes for strings', () => {
      const str = 'test string';
      const hash1 = CryptoUtils.hashString(str);
      const hash2 = CryptoUtils.hashString(str);
      
      expect(hash1).toBe(hash2);
    });
  });

  describe('generateKeyPair', () => {
    it('should generate valid RSA key pairs', () => {
      const { publicKey, privateKey } = CryptoUtils.generateKeyPair();
      
      expect(publicKey).toContain('BEGIN PUBLIC KEY');
      expect(publicKey).toContain('END PUBLIC KEY');
      expect(privateKey).toContain('BEGIN PRIVATE KEY');
      expect(privateKey).toContain('END PRIVATE KEY');
    });

    it('should generate unique key pairs each time', () => {
      const keys1 = CryptoUtils.generateKeyPair();
      const keys2 = CryptoUtils.generateKeyPair();
      
      expect(keys1.publicKey).not.toBe(keys2.publicKey);
      expect(keys1.privateKey).not.toBe(keys2.privateKey);
    });
  });

  describe('signContent and verifySignature', () => {
    it('should successfully sign and verify content', () => {
      const { publicKey, privateKey } = CryptoUtils.generateKeyPair();
      const content = 'Test message for signing';
      
      const signature = CryptoUtils.signContent(content, privateKey);
      const isValid = CryptoUtils.verifySignature(content, signature, publicKey);
      
      expect(signature).toBeDefined();
      expect(signature.length).toBeGreaterThan(0);
      expect(isValid).toBe(true);
    });

    it('should reject invalid signatures', () => {
      const { publicKey, privateKey } = CryptoUtils.generateKeyPair();
      const content = 'Test message';
      const wrongContent = 'Wrong message';
      
      const signature = CryptoUtils.signContent(content, privateKey);
      const isValid = CryptoUtils.verifySignature(wrongContent, signature, publicKey);
      
      expect(isValid).toBe(false);
    });

    it('should reject signatures with wrong public key', () => {
      const { privateKey } = CryptoUtils.generateKeyPair();
      const { publicKey: wrongPublicKey } = CryptoUtils.generateKeyPair();
      const content = 'Test message';
      
      const signature = CryptoUtils.signContent(content, privateKey);
      const isValid = CryptoUtils.verifySignature(content, signature, wrongPublicKey);
      
      expect(isValid).toBe(false);
    });
  });

  describe('verifyChainIntegrity', () => {
    it('should validate empty chain', () => {
      const contentHash = 'abc123';
      const result = CryptoUtils.verifyChainIntegrity([], contentHash);
      expect(result).toBe(true);
    });

    it('should validate correct chain', () => {
      const contentHash = 'abc123';
      const chain = [
        {
          action: 'edit',
          timestamp: new Date().toISOString(),
          actor: 'test',
          previousHash: contentHash,
          currentHash: 'def456',
          signature: 'sig1'
        }
      ];
      
      const result = CryptoUtils.verifyChainIntegrity(chain, contentHash);
      expect(result).toBe(true);
    });

    it('should detect broken chain', () => {
      const contentHash = 'abc123';
      const chain = [
        {
          action: 'edit',
          timestamp: new Date().toISOString(),
          actor: 'test',
          previousHash: 'wrong_hash',
          currentHash: 'def456',
          signature: 'sig1'
        }
      ];
      
      const result = CryptoUtils.verifyChainIntegrity(chain, contentHash);
      expect(result).toBe(false);
    });
  });
});

describe('ManifestBuilder', () => {
  const contentHash = 'abc123';
  const contentType = ContentType.IMAGE;
  const creator = 'Test Creator';

  describe('constructor', () => {
    it('should create manifest with required fields', () => {
      const builder = new ManifestBuilder(contentHash, contentType, creator);
      const manifest = builder.build();

      expect(manifest.id).toBeDefined();
      expect(manifest.contentHash).toBe(contentHash);
      expect(manifest.contentType).toBe(contentType);
      expect(manifest.createdBy).toBe(creator);
      expect(manifest.createdAt).toBeDefined();
      expect(manifest.signatures).toEqual([]);
      expect(manifest.assertions).toEqual([]);
      expect(manifest.chain).toEqual([]);
    });
  });

  describe('addAssertion', () => {
    it('should add assertion with hash', () => {
      const builder = new ManifestBuilder(contentHash, contentType, creator);
      builder.addAssertion('test.assertion', { key: 'value' });
      const manifest = builder.build();

      expect(manifest.assertions).toHaveLength(1);
      expect(manifest.assertions[0].type).toBe('test.assertion');
      expect(manifest.assertions[0].data).toEqual({ key: 'value' });
      expect(manifest.assertions[0].hash).toBeDefined();
    });
  });

  describe('addMetadata', () => {
    it('should add metadata assertion', () => {
      const builder = new ManifestBuilder(contentHash, contentType, creator);
      builder.addMetadata({ location: 'New York' });
      const manifest = builder.build();

      expect(manifest.assertions).toHaveLength(1);
      expect(manifest.assertions[0].type).toBe('stds.metadata');
    });
  });

  describe('sign', () => {
    it('should add signature to manifest', () => {
      const { privateKey } = CryptoUtils.generateKeyPair();
      const builder = new ManifestBuilder(contentHash, contentType, creator);
      builder.sign(privateKey, creator);
      const manifest = builder.build();

      expect(manifest.signatures).toHaveLength(1);
      expect(manifest.signatures[0].algorithm).toBe('RSA-SHA256');
      expect(manifest.signatures[0].signer).toBe(creator);
      expect(manifest.signatures[0].signature).toBeDefined();
      expect(manifest.signatures[0].timestamp).toBeDefined();
    });
  });

  describe('addChainLink', () => {
    it('should add chain link', () => {
      const { privateKey } = CryptoUtils.generateKeyPair();
      const builder = new ManifestBuilder(contentHash, contentType, creator);
      builder.addChainLink('resize', creator, privateKey);
      const manifest = builder.build();

      expect(manifest.chain).toHaveLength(1);
      expect(manifest.chain[0].action).toBe('resize');
      expect(manifest.chain[0].actor).toBe(creator);
      expect(manifest.chain[0].previousHash).toBe(contentHash);
      expect(manifest.chain[0].currentHash).toBeDefined();
      expect(manifest.chain[0].signature).toBeDefined();
    });

    it('should link to previous chain item', () => {
      const { privateKey } = CryptoUtils.generateKeyPair();
      const builder = new ManifestBuilder(contentHash, contentType, creator);
      
      builder.addChainLink('edit1', creator, privateKey);
      builder.addChainLink('edit2', creator, privateKey);
      
      const manifest = builder.build();

      expect(manifest.chain).toHaveLength(2);
      expect(manifest.chain[1].previousHash).toBe(manifest.chain[0].currentHash);
    });
  });
});
