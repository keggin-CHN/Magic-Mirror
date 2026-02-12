import { convertFileSrc } from "@tauri-apps/api/core";
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
    const { open } = await import("@tauri-apps/plugin-shell");
    return open(url);
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
  const { exit } = await import("@tauri-apps/plugin-process");
  exit(0);
}