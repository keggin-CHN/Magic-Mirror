import { convertFileSrc } from "@tauri-apps/api/core";
import { open as openExternal } from "@tauri-apps/plugin-shell";
import { isTauri } from "./runtime";

export function convertFileSrcSafe(path: string) {
  return isTauri() ? convertFileSrc(path) : path;
}

export async function openDialogSafe(options: {
  multiple?: boolean;
  directory?: boolean;
  filters?: { name: string; extensions: string[] }[];
}) {
  if (!isTauri()) {
    return null;
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  return open(options);
}

export async function openExternalSafe(url: string) {
  if (isTauri()) {
    return openExternal(url);
  }
  if (typeof window !== "undefined") {
    window.open(url, "_blank", "noopener");
  }
  return null;
}

export async function exitAppSafe() {
  if (!isTauri()) {
    return;
  }
  // Kill the sidecar server first: `exit(0)` terminates the app immediately,
  // so React effect cleanups never run and the server process would be orphaned.
  try {
    const { Server } = await import("./server");
    await Server.kill();
  } catch {
    // ignore, still exit
  }
  const { exit } = await import("@tauri-apps/plugin-process");
  exit(0);
}
