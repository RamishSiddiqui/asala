import fs from 'fs-extra';
import path from 'path';
import chalk from 'chalk';
import ora from 'ora';
import inquirer from 'inquirer';
import { CryptoUtils } from '@asala/core';

interface KeysOptions {
  generate?: boolean;
  output: string;
}

export async function keysCommand(options: KeysOptions) {
  if (options.generate) {
    await generateKeys(options.output);
  } else {
    await showKeysHelp();
  }
}

async function generateKeys(outputDir: string) {
  const spinner = ora('Generating key pair...').start();

  try {
    // Ensure output directory exists
    await fs.ensureDir(outputDir);

    // Generate key pair
    const { publicKey, privateKey } = CryptoUtils.generateKeyPair();

    // Save keys
    const privateKeyPath = path.join(outputDir, 'private.pem');
    const publicKeyPath = path.join(outputDir, 'public.pem');

    await fs.writeFile(privateKeyPath, privateKey);
    await fs.writeFile(publicKeyPath, publicKey);

    spinner.stop();

    console.log(chalk.green('\n✓ Key pair generated successfully!\n'));
    console.log(chalk.gray('Private key:'), privateKeyPath);
    console.log(chalk.gray('Public key:'), publicKeyPath);
    console.log();
    console.log(chalk.yellow('⚠ Important: Keep your private key secure and never share it!'));
    console.log();

    // Ask about key backup
    const { backup } = await inquirer.prompt([{
      type: 'confirm',
      name: 'backup',
      message: 'Would you like to create an encrypted backup of your private key?',
      default: false
    }]);

    if (backup) {
      const { password } = await inquirer.prompt([{
        type: 'password',
        name: 'password',
        message: 'Enter a strong password for encryption:',
        mask: '*'
      }]);

      // In a real implementation, encrypt the private key
      console.log(chalk.gray('\nCreating encrypted backup...'));
      // TODO: Implement encryption
    }

  } catch (error) {
    spinner.fail(`Key generation failed: ${error.message}`);
    process.exit(1);
  }
}

async function showKeysHelp() {
  console.log(chalk.bold('\nAsala - Key Management\n'));
  console.log('Generate a new key pair:');
  console.log(chalk.cyan('  asala keys --generate\n'));
  console.log('Options:');
  console.log('  -g, --generate    Generate new key pair');
  console.log('  -o, --output      Output directory (default: ./keys)\n');
  console.log(chalk.gray('The generated keys will be used to sign content and verify authenticity.\n'));
}
