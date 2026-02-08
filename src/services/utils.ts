export function timestamp() {
  return new Date().getTime();
}

export async function sleep(time: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, time));
}

const kVideoExtensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"];
const kImageExtensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"];

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
