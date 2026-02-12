export function isTauri() {
  if (typeof window === "undefined") {
    return false;
  }
  const tauriGlobal = (window as any).__TAURI__;
  const tauriInternals = (window as any).__TAURI_INTERNALS__;
  return Boolean(tauriGlobal || tauriInternals);
}