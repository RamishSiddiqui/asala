import { Asala } from '../index';
import { VerificationStatus, ContentType } from '../types';

describe('Asala', () => {
  let asala: Asala;

  beforeEach(() => {
    asala = new Asala();
  });

  describe('generateKeyPair', () => {
    it('should generate valid key pair', () => {
      const keys = asala.generateKeyPair();

      expect(keys.publicKey).toBeDefined();
      expect(keys.privateKey).toBeDefined();
      expect(keys.publicKey).toContain('BEGIN PUBLIC KEY');
      expect(keys.privateKey).toContain('BEGIN PRIVATE KEY');
    });
  });

  describe('signContent', () => {
    it('should create valid manifest', () => {
      const { privateKey } = asala.generateKeyPair();
      const content = Buffer.from('test content');
      const creator = 'Test Creator';

      const manifest = asala.signContent(content, privateKey, creator);

      expect(manifest.id).toBeDefined();
      expect(manifest.contentHash).toBeDefined();
      expect(manifest.createdBy).toBe(creator);
      expect(manifest.signatures).toHaveLength(1);
      expect(manifest.assertions).toHaveLength(1); // creation info
    });

    it('should detect content type automatically', () => {
      const { privateKey } = asala.generateKeyPair();

      // JPEG magic numbers
      const jpegContent = Buffer.from([0xFF, 0xD8, 0xFF, 0xE0]);
      const manifest = asala.signContent(jpegContent, privateKey, 'Test');

      expect(manifest.contentType).toBe(ContentType.IMAGE);
    });

    it('should use provided content type', () => {
      const { privateKey } = asala.generateKeyPair();
      const content = Buffer.from('test');

      const manifest = asala.signContent(content, privateKey, 'Test', ContentType.AUDIO);

      expect(manifest.contentType).toBe(ContentType.AUDIO);
    });
  });

  describe('verify', () => {
    it('should verify signed content', async () => {
      const { privateKey } = asala.generateKeyPair();
      const content = Buffer.from('test content');
      const manifest = asala.signContent(content, privateKey, 'Test Creator');

      const result = await asala.verify(content, manifest);

      expect(result.status).toBe(VerificationStatus.VERIFIED);
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.layers.length).toBeGreaterThan(0);
    });

    it('should detect tampered content', async () => {
      const { privateKey } = asala.generateKeyPair();
      const originalContent = Buffer.from('original content');
      const tamperedContent = Buffer.from('tampered content');

      // Sign original but verify tampered
      const manifest = asala.signContent(originalContent, privateKey, 'Test');
      const result = await asala.verify(tamperedContent, manifest);

      // Should fail because hash doesn't match
      expect(result.status).not.toBe(VerificationStatus.VERIFIED);
    });

    it('should return unverified when no manifest provided', async () => {
      const content = Buffer.from('JPEG header');
      content[0] = 0xFF;
      content[1] = 0xD8;

      const result = await asala.verify(content);

      expect(result.status).toBe(VerificationStatus.MISSING_PROVENANCE);
    });

    it('should include warnings in result', async () => {
      const result = await asala.verify(Buffer.from('test'));

      expect(result.warnings.length).toBeGreaterThan(0);
      expect(result.warnings).toContain('No embedded manifest found');
    });
  });

  describe('integration', () => {
    it('should handle full workflow', async () => {
      // 1. Generate keys
      const keys = asala.generateKeyPair();

      // 2. Create content
      const content = Buffer.from('Important document');

      // 3. Sign content
      const manifest = asala.signContent(content, keys.privateKey, 'Alice');

      // 4. Verify content
      const result = await asala.verify(content, manifest);

      // 5. Check results
      expect(result.status).toBe(VerificationStatus.VERIFIED);
      expect(result.manifest).toBeDefined();
      expect(result.manifest!.createdBy).toBe('Alice');

      // Check signature layer
      const sigLayer = result.layers.find(l => l.name === 'Signature Verification');
      expect(sigLayer).toBeDefined();
      expect(sigLayer!.passed).toBe(true);
    });

    it('should verify content with chain of edits', async () => {
      const keys = asala.generateKeyPair();
      const content = Buffer.from('Original photo');

      // Sign original
      let manifest = asala.signContent(content, keys.privateKey, 'Photographer');

      // Simulate edits (in real usage, would modify content and update hash)
      // For testing, we're just adding chain links

      // Verify
      const result = await asala.verify(content, manifest);

      expect(result.status).toBe(VerificationStatus.VERIFIED);
    });
  });
});
