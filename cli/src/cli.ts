#!/usr/bin/env node

import { program } from 'commander';
import chalk from 'chalk';
import { verifyCommand } from './commands/verify';
import { signCommand } from './commands/sign';
import { keysCommand } from './commands/keys';
import { manifestCommand } from './commands/manifest';

const packageJson = require('../package.json');

program
  .name('asala')
  .description('Asala - CLI tool for content authenticity verification')
  .version(packageJson.version);

// Verify command
program
  .command('verify <file>')
  .description('Verify content authenticity')
  .option('-m, --manifest <path>', 'Path to manifest file')
  .option('-t, --trust <keys...>', 'Trusted public keys')
  .option('-j, --json', 'Output as JSON')
  .option('-v, --verbose', 'Verbose output')
  .action(verifyCommand);

// Sign command
program
  .command('sign <file>')
  .description('Sign content with provenance data')
  .requiredOption('-k, --key <path>', 'Path to private key')
  .option('-o, --output <path>', 'Output file path')
  .option('-c, --creator <name>', 'Creator name')
  .option('-d, --device <device>', 'Device name')
  .action(signCommand);

// Keys command
program
  .command('keys')
  .description('Manage cryptographic keys')
  .option('-g, --generate', 'Generate new key pair')
  .option('-o, --output <dir>', 'Output directory', './keys')
  .action(keysCommand);

// Manifest command
program
  .command('manifest <file>')
  .description('View or extract manifest from content')
  .option('-e, --extract', 'Extract manifest to file')
  .option('-o, --output <path>', 'Output path for extracted manifest')
  .action(manifestCommand);

// Global error handler
process.on('unhandledRejection', (error) => {
  console.error(chalk.red('Error:'), error);
  process.exit(1);
});

program.parse();
