# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of Asala seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Reporting Process

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:

📧 **security@asala.org**

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

We will acknowledge receipt of your vulnerability report within **48 hours** and will send you regular updates about our progress.

- **Initial Response**: Within 48 hours
- **Assessment Complete**: Within 7 days
- **Fix Released**: Within 90 days (depending on severity)

### Security Measures

Asala implements the following security measures:

1. **Cryptographic Security**
   - RSA-2048 bit keys for signing
   - SHA-256 for hashing
   - Industry-standard cryptographic libraries

2. **Private Key Protection**
   - Private keys never leave the user's device
   - No cloud storage of keys
   - Keys generated locally

3. **Verification Security**
   - Content hash verification
   - Signature validation
   - Chain of custody integrity checks

### Security Best Practices

When using Asala:

1. **Protect Your Private Keys**
   - Never share your private key
   - Store backups securely (encrypted)
   - Use strong passwords for key protection

2. **Verify Signatures**
   - Always verify content before trusting it
   - Check the entire chain of custody
   - Verify signers are in your trust store

3. **Keep Software Updated**
   - Update to the latest version promptly
   - Monitor security advisories
   - Report suspicious activity

### Security Considerations

**What Asala Protects Against:**
- Content tampering after signing
- Signature forgery (computationally infeasible)
- Man-in-the-middle attacks on verification

**What Asala Does NOT Protect Against:**
- Compromised private keys (user responsibility)
- Social engineering attacks
- Malware on the user's device
- Compromised signing devices/hardware

### Third-Party Dependencies

We regularly audit our dependencies for security vulnerabilities:

- Node.js packages are scanned with `npm audit`
- Python packages are monitored with `safety`
- Cryptographic libraries follow industry best practices

### Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine the affected versions
2. Audit code to find any potential similar problems
3. Prepare fixes for all supported versions
4. Release new versions with patches
5. Notify users through security advisories

### Acknowledgments

We would like to thank the following security researchers who have responsibly disclosed vulnerabilities:

*(List will be updated as reports are received)*

### Contact

For any security-related questions or concerns, please contact:

📧 security@asala.org

For general questions, please use:
- GitHub Discussions: https://github.com/your-org/asala/discussions
- Discord: [Link]

---

**Last Updated**: February 15, 2024
