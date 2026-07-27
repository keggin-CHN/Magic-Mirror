#!/usr/bin/env node
// Verifies SERVER_PACKAGE_VERSION in src/services/server.ts matches the
// version in src-python/pyproject.toml. Exits 1 on mismatch.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

const serverTs = readFileSync(join(root, "src/services/server.ts"), "utf8");
const tsMatch = serverTs.match(
  /SERVER_PACKAGE_VERSION\s*=\s*["']([^"']+)["']/
);
if (!tsMatch) {
  console.error("check-version-sync: SERVER_PACKAGE_VERSION not found in src/services/server.ts");
  process.exit(1);
}

const pyproject = readFileSync(join(root, "src-python/pyproject.toml"), "utf8");
const pyMatch = pyproject.match(/^version\s*=\s*["']([^"']+)["']/m);
if (!pyMatch) {
  console.error("check-version-sync: version not found in src-python/pyproject.toml");
  process.exit(1);
}

const tsVersion = tsMatch[1];
const pyVersion = pyMatch[1];

if (tsVersion !== pyVersion) {
  console.error(
    `check-version-sync: version mismatch\n` +
    `  src/services/server.ts SERVER_PACKAGE_VERSION = ${tsVersion}\n` +
    `  src-python/pyproject.toml version              = ${pyVersion}`
  );
  process.exit(1);
}

console.log(`check-version-sync: OK (${tsVersion})`);
