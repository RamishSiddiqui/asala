import fs from 'fs-extra';
import path from 'path';
import chalk from 'chalk';
import ora from 'ora';
import { Asala } from '@asala/core';

interface SignOptions {
  key: string;
  output?: string;
  creator?: string;
  device?: string;
}

export async function signCommand(file: string, options: SignOptions) {
  const spinner = ora('Signing content...').start();

  try {
    // Check if file exists
    if (!await fs.pathExists(file)) {
      spinner.fail(`File not found: ${file}`);
      process.exit(1);
    }

    // Check if key exists
    if (!await fs.pathExists(options.key)) {
      spinner.fail(`Private key not found: ${options.key}`);
      process.exit(1);
    }

    // Read content and key
    const content = await fs.readFile(file);
    const privateKey = await fs.readFile(options.key, 'utf-8');

    // Create signer
    const asala = new Asala();

    spinner.text = 'Creating provenance manifest...';

    // Sign content
    const creator = options.creator || 'Unknown';
    const manifest = asala.signContent(content, privateKey, creator);

    // Add device info if provided
    if (options.device) {
      // This would modify the manifest in a real implementation
      spinner.text = 'Adding device information...';
    }

    spinner.stop();

    // Determine output path
    const outputPath = options.output || `${file}.manifest.json`;

    // Save manifest
    await fs.writeJson(outputPath, manifest, { spaces: 2 });

    console.log(chalk.green('\n✓ Content signed successfully!\n'));
    console.log(chalk.gray('Manifest saved to:'), path.resolve(outputPath));
    console.log(chalk.gray('Content hash:'), manifest.contentHash);
    console.log(chalk.gray('Signatures:'), manifest.signatures.length);
    console.log();

  } catch (error) {
    spinner.fail(`Signing failed: ${error.message}`);
    process.exit(1);
  }
}
