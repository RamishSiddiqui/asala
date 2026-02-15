import fs from 'fs-extra';
import path from 'path';
import chalk from 'chalk';
import ora from 'ora';
import { Asala } from '@asala/core';

interface VerifyOptions {
  manifest?: string;
  trust?: string[];
  json?: boolean;
  verbose?: boolean;
}

export async function verifyCommand(file: string, options: VerifyOptions) {
  const spinner = ora('Verifying content...').start();

  try {
    // Check if file exists
    if (!await fs.pathExists(file)) {
      spinner.fail(`File not found: ${file}`);
      process.exit(1);
    }

    // Read content
    const content = await fs.readFile(file);

    // Load manifest if provided
    let manifest = null;
    if (options.manifest) {
      if (!await fs.pathExists(options.manifest)) {
        spinner.fail(`Manifest not found: ${options.manifest}`);
        process.exit(1);
      }
      manifest = await fs.readJson(options.manifest);
    }

    // Create verifier
    const asala = new Asala();

    spinner.text = 'Running verification...';

    // Verify content
    const result = await asala.verify(content, manifest || undefined, {
      includeMetadata: true,
      includeChainAnalysis: true,
      trustStore: options.trust
    });

    spinner.stop();

    // Output results
    if (options.json) {
      console.log(JSON.stringify(result, null, 2));
    } else {
      displayResults(file, result, options.verbose);
    }

    // Exit with appropriate code
    process.exit(result.status === 'verified' ? 0 : 1);

  } catch (error) {
    spinner.fail(`Verification failed: ${error.message}`);
    process.exit(1);
  }
}

function displayResults(file: string, result: any, verbose: boolean = false) {
  console.log('\n' + chalk.bold('='.repeat(60)));
  console.log(chalk.bold('  Content Verification Report'));
  console.log(chalk.bold('='.repeat(60)) + '\n');

  console.log(chalk.gray('File:'), path.resolve(file));
  console.log(chalk.gray('Status:'), getStatusColor(result.status)(result.status.toUpperCase()));
  console.log(chalk.gray('Confidence:'), `${result.confidence}%`);

  if (result.warnings.length > 0) {
    console.log('\n' + chalk.yellow('⚠ Warnings:'));
    result.warnings.forEach((warning: string) => {
      console.log(chalk.yellow('  •'), warning);
    });
  }

  if (result.errors.length > 0) {
    console.log('\n' + chalk.red('✖ Errors:'));
    result.errors.forEach((error: string) => {
      console.log(chalk.red('  •'), error);
    });
  }

  if (verbose && result.layers.length > 0) {
    console.log('\n' + chalk.bold('Verification Layers:'));
    result.layers.forEach((layer: any) => {
      const icon = layer.passed ? chalk.green('✓') : chalk.red('✗');
      console.log(`  ${icon} ${layer.name}: ${layer.score}%`);
      if (verbose && layer.details) {
        Object.entries(layer.details).forEach(([key, value]) => {
          console.log(chalk.gray(`      ${key}: ${value}`));
        });
      }
    });
  }

  if (result.manifest) {
    console.log('\n' + chalk.bold('Provenance Data:'));
    console.log(chalk.gray('  ID:'), result.manifest.id);
    console.log(chalk.gray('  Creator:'), result.manifest.createdBy);
    console.log(chalk.gray('  Created:'), new Date(result.manifest.createdAt).toLocaleString());
    console.log(chalk.gray('  Signatures:'), result.manifest.signatures.length);
    if (result.manifest.chain.length > 0) {
      console.log(chalk.gray('  Edit History:'), `${result.manifest.chain.length} transformations`);
    }
  }

  console.log('\n' + chalk.bold('='.repeat(60)) + '\n');
}

function getStatusColor(status: string) {
  switch (status) {
    case 'verified':
      return chalk.green.bold;
    case 'tampered':
      return chalk.red.bold;
    case 'unverified':
      return chalk.yellow.bold;
    default:
      return chalk.gray;
  }
}
