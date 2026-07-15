export function timestamp() {
  return new Date().getTime();
}

export async function sleep(time: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, time));
}

export const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;
export const LONG_REQUEST_TIMEOUT_MS = 120_000;
export const UPLOAD_REQUEST_TIMEOUT_MS = 10 * 60_000;
export const PROGRESS_REQUEST_TIMEOUT_MS = 15_000;

export async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit = {},
  timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const externalSignal = init.signal;
  const abortFromExternal = () => controller.abort(externalSignal?.reason);
  if (externalSignal?.aborted) {
    abortFromExternal();
  } else {
    externalSignal?.addEventListener("abort", abortFromExternal, { once: true });
  }
  const timer = globalThis.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    globalThis.clearTimeout(timer);
    externalSignal?.removeEventListener("abort", abortFromExternal);
  }
}

const kVideoExtensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"];
const kImageExtensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".json"];

export function getFileExtension(filePath: string) {
  const index = filePath.lastIndexOf(".");
  if (index === -1) {
    return "";
  }
  return filePath.slice(index).toLowerCase();
}

export function isVideoFile(filePath: string) {
  const ext = getFileExtension(filePath);
  return kVideoExtensions.includes(ext);
}

export function isImageFile(filePath: string) {
  const ext = getFileExtension(filePath);
  return kImageExtensions.includes(ext);
}
