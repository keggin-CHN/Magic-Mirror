#!/usr/bin/env node

import fs from "fs/promises";
import { existsSync, mkdirSync } from "fs";
import path from "path";

// Helper to recursively find files
async function findFiles(dir, predicate) {
  let results = [];
  if (!existsSync(dir)) return results;

  try {
    const list = await fs.readdir(dir, { withFileTypes: true });
    for (const dirent of list) {
      const fullPath = path.join(dir, dirent.name);
      if (dirent.isDirectory()) {
        results = results.concat(await findFiles(fullPath, predicate));
      } else if (predicate(fullPath)) {
        results.push(fullPath);
      }
    }
  } catch (err) {
    console.error(`Error reading ${dir}: ${err.message}`);
  }
  return results;
}

const copyFile = async (from, to) => {
  if (!existsSync(from)) {
    console.error(`Source file does not exist: ${from}`);
    return false;
  }

  const dirname = path.dirname(to);
  if (!existsSync(dirname)) {
    mkdirSync(dirname, { recursive: true });
  }

  await fs.copyFile(from, to);
  console.log(`Copied ${from} -> ${to}`);
};

async function main() {
  const args = process.argv.slice(2);
  const [target, appName] = args;

  console.log(`Target: ${target}`);
  console.log(`AppName: ${appName}`);
  console.log(`Platform: ${process.platform}`);

  const targetBase = path.resolve("src-tauri/target");
  console.log(`Searching in: ${targetBase}`);

  let extensions = [];
  switch (process.platform) {
    case "darwin":
      extensions = [".dmg"];
      break;
    case "win32":
      extensions = [".exe"];
      break;
    default:
      console.warn(`Unsupported platform: ${process.platform}`);
  }

  if (extensions.length === 0) return;

  // Find all files with matching extensions in target directory
  // that are also in a 'bundle' directory and 'release' directory
  const files = await findFiles(targetBase, (filePath) => {
    return extensions.some(ext => filePath.endsWith(ext)) &&
      filePath.includes("release") &&
      filePath.includes("bundle");
  });

  if (files.length === 0) {
    console.error("No bundle files found!");

    // Debug: List all files in bundle directories
    console.log("Listing all files in 'bundle' directories for debugging:");
    const debugFiles = await findFiles(targetBase, (p) => p.includes("bundle"));
    console.log(debugFiles.join("\n"));

    process.exit(1);
  }

  console.log("Found candidates:", files);

  // Copy found files
  for (const file of files) {
    const ext = path.extname(file);
    await copyFile(file, path.join("dist", appName + ext));
    console.log(`✅ ${appName + ext}`);
  }
}

main();
