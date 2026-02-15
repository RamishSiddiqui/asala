export interface ContentManifest {
  id: string;
  contentHash: string;
  contentType: ContentType;
  createdAt: string;
  createdBy: string;
  signatures: SignatureInfo[];
  assertions: Assertion[];
  chain: ChainLink[];
}

export interface SignatureInfo {
  algorithm: string;
  publicKey: string;
  signature: string;
  timestamp: string;
  signer: string;
}

export interface Assertion {
  type: string;
  data: Record<string, unknown>;
  hash: string;
}

export interface ChainLink {
  action: string;
  timestamp: string;
  actor: string;
  previousHash: string;
  currentHash: string;
  signature: string;
}

export enum ContentType {
  IMAGE = 'image',
  VIDEO = 'video',
  AUDIO = 'audio',
  DOCUMENT = 'document'
}

export enum VerificationStatus {
  VERIFIED = 'verified',
  UNVERIFIED = 'unverified',
  TAMPERED = 'tampered',
  INVALID_SIGNATURE = 'invalid_signature',
  MISSING_PROVENANCE = 'missing_provenance'
}

export interface VerificationResult {
  status: VerificationStatus;
  confidence: number;
  manifest?: ContentManifest;
  warnings: string[];
  errors: string[];
  layers: LayerResult[];
}

export interface LayerResult {
  name: string;
  passed: boolean;
  score: number;
  details: Record<string, unknown>;
}

export interface VerificationOptions {
  includeMetadata?: boolean;
  includeChainAnalysis?: boolean;
  includePhysicsCheck?: boolean;
  trustStore?: string[];
}
