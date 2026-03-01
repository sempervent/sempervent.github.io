#!/usr/bin/env node
/**
 * render_mermaid_svg.mjs
 *
 * Renders .mmd Mermaid source files to .svg using @mermaid-js/mermaid-cli.
 *
 * Usage:
 *   # Render a single file:
 *   node render_mermaid_svg.mjs --input path/to/diagram.mmd
 *
 *   # Render all .mmd files under a directory (recursive):
 *   node render_mermaid_svg.mjs --input path/to/diagrams/
 *
 * Or via npm scripts from tools/diagrams/:
 *   npm run render:workflow
 *   npm run render:all
 */

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import {
  existsSync,
  mkdirSync,
  readdirSync,
  statSync,
} from 'node:fs';
import { resolve, join, dirname, basename, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const execFileAsync = promisify(execFile);

// ─── Resolve paths relative to this script's location ────────────────────────
const __dirname = dirname(fileURLToPath(import.meta.url));
const MMDC_BIN = resolve(__dirname, 'node_modules/.bin/mmdc');
const CONFIG   = resolve(__dirname, 'mermaid-config.json');

// ─── CLI argument parsing ─────────────────────────────────────────────────────
const args = process.argv.slice(2);
const inputIdx = args.indexOf('--input');
if (inputIdx === -1 || !args[inputIdx + 1]) {
  console.error('Usage: node render_mermaid_svg.mjs --input <file.mmd|directory>');
  process.exit(1);
}
const inputPath = resolve(process.cwd(), args[inputIdx + 1]);

// ─── Collect .mmd files ───────────────────────────────────────────────────────
function collectMmdFiles(target) {
  const stat = statSync(target);
  if (stat.isFile()) {
    return extname(target) === '.mmd' ? [target] : [];
  }
  // Directory: walk recursively
  const entries = readdirSync(target, { withFileTypes: true });
  return entries.flatMap((entry) => {
    const full = join(target, entry.name);
    if (entry.isDirectory()) return collectMmdFiles(full);
    if (entry.isFile() && extname(entry.name) === '.mmd') return [full];
    return [];
  });
}

// ─── Render a single .mmd → .svg ─────────────────────────────────────────────
async function renderOne(mmdFile) {
  const outFile = join(dirname(mmdFile), basename(mmdFile, '.mmd') + '.svg');

  // Ensure output directory exists (should already, but be safe)
  mkdirSync(dirname(outFile), { recursive: true });

  const mmdcArgs = [
    '--input',  mmdFile,
    '--output', outFile,
    '--configFile', CONFIG,
    '--backgroundColor', 'transparent',
    '--quiet',
  ];

  try {
    await execFileAsync(MMDC_BIN, mmdcArgs);
    return { file: outFile, ok: true };
  } catch (err) {
    return { file: outFile, ok: false, error: err.message };
  }
}

// ─── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  if (!existsSync(MMDC_BIN)) {
    console.error(`mmdc binary not found at ${MMDC_BIN}`);
    console.error('Run: cd tools/diagrams && npm install');
    process.exit(1);
  }

  const files = collectMmdFiles(inputPath);
  if (files.length === 0) {
    console.warn(`No .mmd files found under: ${inputPath}`);
    process.exit(0);
  }

  console.log(`\nRendering ${files.length} diagram(s)...\n`);

  const results = await Promise.all(files.map(renderOne));

  let ok = 0;
  let fail = 0;
  for (const r of results) {
    if (r.ok) {
      console.log(`  ✓  ${r.file}`);
      ok++;
    } else {
      console.error(`  ✗  ${r.file}`);
      console.error(`     ${r.error}`);
      fail++;
    }
  }

  console.log(`\nDone: ${ok} rendered, ${fail} failed.\n`);
  if (fail > 0) process.exit(1);
}

main();
