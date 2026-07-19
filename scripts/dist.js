#!/usr/bin/env node

import fs from "fs/promises";
import { existsSync } from "fs";
import path from "path";

async function findFiles(dir, predicate) {
  let results = [];
  if (!existsSync(dir)) return results;

  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results = results.concat(await findFiles(fullPath, predicate));
    } else if (predicate(fullPath)) {
      results.push(fullPath);
    }
  }

  return results;
}

async function copyFile(from, to) {
  if (!existsSync(from)) {
    throw new Error(`Source file does not exist: ${from}`);
  }

  await fs.mkdir(path.dirname(to), { recursive: true });
  await fs.copyFile(from, to);
  console.log(`Copied ${from} -> ${to}`);
}

function usage() {
  console.error("Usage: node scripts/dist.js <target-triple> <app-name>");
}

function platformExtensions(platform) {
  switch (platform) {
    case "darwin":
      return [".dmg"];
    case "win32":
      return [".exe"];
    case "linux":
      return [".AppImage", ".deb"];
    default:
      throw new Error(
        `Unsupported platform: ${platform}. Supported platforms: darwin, win32, linux.`,
      );
  }
}

async function main() {
  const [target, appName] = process.argv.slice(2);

  if (!target || !appName) {
    usage();
    process.exit(2);
  }

  console.log(`Target: ${target}`);
  console.log(`AppName: ${appName}`);
  console.log(`Platform: ${process.platform}`);

  const targetRoot = path.resolve("src-tauri/target");
  const targetBase = existsSync(path.join(targetRoot, target))
    ? path.join(targetRoot, target)
    : targetRoot;
  const extensions = platformExtensions(process.platform);
  console.log(`Searching in: ${targetBase}`);

  const files = await findFiles(targetBase, (filePath) => (
    extensions.some((ext) => filePath.endsWith(ext)) &&
    filePath.includes("release") &&
    filePath.includes("bundle")
  ));

  if (files.length === 0) {
    console.error("No bundle files found.");

    const debugFiles = await findFiles(
      targetBase,
      (filePath) => filePath.includes("bundle"),
    );
    if (debugFiles.length > 0) {
      console.error("Files found under bundle directories:");
      console.error(debugFiles.join("\n"));
    }

    process.exit(1);
  }

  console.log("Found candidates:", files);

  const filesByExtension = new Map();
  for (const file of files) {
    const ext = path.extname(file);
    filesByExtension.set(ext, [...(filesByExtension.get(ext) || []), file]);
  }
  const duplicateExtensions = [...filesByExtension.entries()]
    .filter(([, candidates]) => candidates.length > 1);
  if (duplicateExtensions.length > 0) {
    for (const [ext, candidates] of duplicateExtensions) {
      console.error(`Multiple bundle candidates found for ${ext}:`);
      console.error(candidates.join("\n"));
    }
    process.exit(1);
  }

  for (const file of files) {
    const ext = path.extname(file);
    await copyFile(file, path.join("dist", `${appName}${ext}`));
    console.log(`Prepared ${appName}${ext}`);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
