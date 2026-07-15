import type {
  DetectFacesResult,
  Region,
  Task,
  TaskResult,
  VideoGpuModesResult,
  VideoTask,
  VideoTaskProgress,
} from "./server";
import {
  PROGRESS_REQUEST_TIMEOUT_MS,
  UPLOAD_REQUEST_TIMEOUT_MS,
  fetchWithTimeout,
} from "./utils";

export interface UploadResult {
  fileId: string;
  url: string;
  type: "image" | "video";
  name: string;
}

export interface LibraryItem {
  id: string;
  name: string;
  url: string;
}

const kTokenKey = "web-token";

function toHeaderRecord(extra?: HeadersInit): Record<string, string> {
  const headers: Record<string, string> = {};
  if (!extra) {
    return headers;
  }
  if (extra instanceof Headers) {
    extra.forEach((value, key) => {
      headers[key] = value;
    });
    return headers;
  }
  if (Array.isArray(extra)) {
    for (const [key, value] of extra) {
      headers[key] = value;
    }
    return headers;
  }
  return { ...extra };
}

class WebServer {
  _baseURL = "/api";
  _token: string | null = null;
  _authExpiredListeners = new Set<() => void>();

  constructor() {
    if (typeof window !== "undefined") {
      this._token = window.localStorage.getItem(kTokenKey);
    }
  }

  get token() {
    return this._token;
  }

  set token(value: string | null) {
    this._token = value;
    if (typeof window === "undefined") {
      return;
    }
    if (value) {
      window.localStorage.setItem(kTokenKey, value);
    } else {
      window.localStorage.removeItem(kTokenKey);
    }
  }

  get isAuthed() {
    return Boolean(this._token);
  }

  _headers(extra?: HeadersInit) {
    const headers = toHeaderRecord(extra);
    if (this._token) {
      headers.Authorization = `Bearer ${this._token}`;
    }
    return headers;
  }

  onAuthExpired(listener: () => void) {
    this._authExpiredListeners.add(listener);
    return () => {
      this._authExpiredListeners.delete(listener);
    };
  }

  async _request(
    input: RequestInfo | URL,
    init: RequestInit = {},
    timeoutMs?: number
  ) {
    const res = await fetchWithTimeout(input, init, timeoutMs);
    if (res.status === 401 && this._token) {
      this.token = null;
      for (const listener of this._authExpiredListeners) {
        try {
          listener();
        } catch {
          // Authentication cleanup must not hide the original response.
        }
      }
    }
    return res;
  }

  async login(password: string) {
    try {
      const res = await fetchWithTimeout(`${this._baseURL}/login`, {
        method: "post",
        headers: { "Content-Type": "application/json;charset=UTF-8" },
        body: JSON.stringify({ password }),
      });
      if (!res.ok) {
        return null;
      }
      const data = await res.json();
      if (data?.token) {
        this.token = data.token;
        return data.token as string;
      }
      return null;
    } catch {
      return null;
    }
  }

  async updateCredential(password: string) {
    try {
      const res = await this._request(`${this._baseURL}/credential`, {
        method: "post",
        headers: this._headers({
          "Content-Type": "application/json;charset=UTF-8",
        }),
        body: JSON.stringify({ password }),
      });
      if (!res.ok) {
        const text = await res.text();
        return { success: false, error: text || "http-error" };
      }
      const data = await res.json();
      if (data?.token) {
        this.token = String(data.token);
      }
      return { success: Boolean(data?.success), error: data?.error };
    } catch {
      return { success: false, error: "network" };
    }
  }

  async uploadFile(
    file: File,
    signal?: AbortSignal
  ): Promise<UploadResult | null> {
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await this._request(
        `${this._baseURL}/upload`,
        { method: "post", headers: this._headers(), body: form, signal },
        UPLOAD_REQUEST_TIMEOUT_MS
      );
      if (!res.ok) {
        return null;
      }
      const data = (await res.json()) as UploadResult;
      if (data?.fileId) {
        data.url = this.buildFileUrl(data.fileId);
      }
      return data;
    } catch {
      return null;
    }
  }

  async uploadLibrary(
    file: File,
    signal?: AbortSignal
  ): Promise<LibraryItem | null> {
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await this._request(
        `${this._baseURL}/library/upload`,
        { method: "post", headers: this._headers(), body: form, signal },
        UPLOAD_REQUEST_TIMEOUT_MS
      );
      if (!res.ok) {
        return null;
      }
      const data = (await res.json()) as LibraryItem;
      if (data?.id) {
        data.url = this.buildLibraryUrl(data.id);
      }
      return data;
    } catch {
      return null;
    }
  }

  async listLibrary(): Promise<LibraryItem[]> {
    try {
      const res = await this._request(`${this._baseURL}/library`, {
        method: "get",
        headers: this._headers(),
      });
      if (!res.ok) {
        return [];
      }
      const data = await res.json();
      if (!Array.isArray(data?.items)) {
        return [];
      }
      return data.items.map((item: LibraryItem) => ({
        ...item,
        url: item.id ? this.buildLibraryUrl(item.id) : item.url,
      }));
    } catch {
      return [];
    }
  }

  async detectImageFaces(
    inputFileId: string,
    regions?: Region[]
  ): Promise<DetectFacesResult> {
    try {
      const res = await this._request(`${this._baseURL}/task/detect-faces`, {
        method: "post",
        headers: this._headers({
          "Content-Type": "application/json;charset=UTF-8",
        }),
        body: JSON.stringify({
          inputFileId,
          regions,
        }),
      });

      if (!res.ok) {
        const errorText = await res.text();
        try {
          const data = JSON.parse(errorText);
          return {
            regions: [],
            error: data?.error || `http-${res.status}`,
          };
        } catch {
          return {
            regions: [],
            error: `http-${res.status}`,
          };
        }
      }

      const data = await res.json();
      return {
        regions: Array.isArray(data?.regions) ? data.regions : [],
        error: data?.error,
      };
    } catch {
      return {
        regions: [],
        error: "network",
      };
    }
  }

  async detectVideoFaces(
    inputFileId: string,
    keyFrameMs: number,
    regions?: Region[]
  ): Promise<DetectFacesResult> {
    try {
      const res = await this._request(`${this._baseURL}/task/video/detect-faces`, {
        method: "post",
        headers: this._headers({
          "Content-Type": "application/json;charset=UTF-8",
        }),
        body: JSON.stringify({
          inputFileId,
          keyFrameMs,
          regions,
        }),
      });

      if (!res.ok) {
        const errorText = await res.text();
        try {
          const data = JSON.parse(errorText);
          return {
            regions: [],
            error: data?.error || `http-${res.status}`,
          };
        } catch {
          return {
            regions: [],
            error: `http-${res.status}`,
          };
        }
      }

      const data = await res.json();
      return {
        regions: Array.isArray(data?.regions) ? data.regions : [],
        frameWidth:
          data?.frameWidth !== undefined && data?.frameWidth !== null
            ? Number(data.frameWidth)
            : undefined,
        frameHeight:
          data?.frameHeight !== undefined && data?.frameHeight !== null
            ? Number(data.frameHeight)
            : undefined,
        frameIndex:
          data?.frameIndex !== undefined && data?.frameIndex !== null
            ? Number(data.frameIndex)
            : undefined,
        error: data?.error,
      };
    } catch {
      return {
        regions: [],
        error: "network",
      };
    }
  }

  async getVideoGpuModes(): Promise<VideoGpuModesResult> {
    try {
      const res = await this._request(`${this._baseURL}/task/video/gpu-modes`, {
        method: "get",
        headers: this._headers(),
      });

      if (!res.ok) {
        return {
          modes: [{ id: "cpu", name: "CPU" }],
          error: `http-${res.status}`,
        };
      }

      const data = await res.json();
      const modes = Array.isArray(data?.modes)
        ? data.modes
          .filter(
            (mode: any) =>
              mode &&
              typeof mode.id === "string" &&
              ["cpu", "directml", "cuda"].includes(mode.id)
          )
          .map((mode: any) => ({
            id: mode.id,
            name: String(mode.name || mode.id),
          }))
        : [];

      if (!modes.some((mode: any) => mode.id === "cpu")) {
        modes.unshift({ id: "cpu", name: "CPU" });
      }

      return {
        modes: modes.length ? modes : [{ id: "cpu", name: "CPU" }],
        availableProviders: Array.isArray(data?.availableProviders)
          ? data.availableProviders.map((item: any) => String(item))
          : [],
        error: data?.error,
      };
    } catch {
      return {
        modes: [{ id: "cpu", name: "CPU" }],
        error: "network",
      };
    }
  }

  async createTask(task: Omit<Task, "id"> & { id: string } & { inputFileId: string; targetFaceId?: string }) {
    try {
      const res = await this._request(`${this._baseURL}/task`, {
        method: "post",
        headers: this._headers({
          "Content-Type": "application/json;charset=UTF-8",
        }),
        body: JSON.stringify({
          id: task.id,
          inputFileId: task.inputFileId,
          targetFaceId: task.targetFaceId,
          targetFaces: task.targetFaces,
          deepSwapMode: task.deepSwapMode,
          regions: task.regions,
          faceSources: task.faceSources,
        }),
      });

      if (!res.ok) {
        const errorText = await res.text();
        try {
          const data = JSON.parse(errorText);
          if (data?.error) {
            return { result: null, error: data.error } as TaskResult;
          }
        } catch {
          // ignore
        }
        return { result: null, error: `http-${res.status}` } as TaskResult;
      }

      const data = await res.json();
      if (data.error) {
        return { result: null, error: data.error } as TaskResult;
      }

      return {
        result: data.resultFileId || null,
        error: data.error,
      } as TaskResult;
    } catch {
      return { result: null, error: "network" } as TaskResult;
    }
  }

  async createVideoTask(task: Omit<VideoTask, "id"> & { id: string } & { inputFileId: string; targetFaceId?: string }) {
    try {
      const res = await this._request(`${this._baseURL}/task/video`, {
        method: "post",
        headers: this._headers({
          "Content-Type": "application/json;charset=UTF-8",
        }),
        body: JSON.stringify({
          id: task.id,
          inputFileId: task.inputFileId,
          targetFaceId: task.targetFaceId,
          targetFaces: task.targetFaces,
          deepSwapMode: task.deepSwapMode,
          segmentDurationSec: task.segmentDurationSec,
          segmentOverlapFrames: task.segmentOverlapFrames,
          regions: task.regions,
          faceSources: task.faceSources,
          keyFrameMs: task.keyFrameMs,
          useGpu: task.useGpu,
          gpuProvider: task.gpuProvider,
          configId: task.configId,
          generateConfigId: task.generateConfigId,
          dryRunConfigOnly: task.dryRunConfigOnly,
        }),
      });

      if (res.status === 405) {
        return { result: null, error: "video-not-supported" } as TaskResult;
      }

      if (!res.ok) {
        const errorText = await res.text();
        try {
          const data = JSON.parse(errorText);
          if (data?.error) {
            return { result: null, error: data.error } as TaskResult;
          }
        } catch {
          // ignore
        }
        return { result: null, error: `http-${res.status}` } as TaskResult;
      }

      const data = await res.json();
      if (data.error) {
        return {
          result: null,
          error: data.error,
          configId: data.configId ?? null,
          status: data.status ?? null,
        } as TaskResult;
      }
      return {
        result: data.resultFileId || null,
        configId: data.configId ?? null,
        status: data.status ?? null,
      } as TaskResult;
    } catch {
      return { result: null, error: "network" } as TaskResult;
    }
  }

  async getVideoTaskProgress(taskId: string): Promise<VideoTaskProgress> {
    try {
      const res = await this._request(
        `${this._baseURL}/task/video/progress/${encodeURIComponent(taskId)}`,
        {
          method: "get",
          headers: this._headers(),
        },
        PROGRESS_REQUEST_TIMEOUT_MS
      );
      if (!res.ok) {
        let error = `http-${res.status}`;
        try {
          const data = await res.json();
          error = data?.error || error;
        } catch {
          // ignore
        }
        return { status: "idle", progress: 0, etaSeconds: null, error };
      }
      const data = await res.json();
      return {
        status: data.status ?? "idle",
        progress: Number.isFinite(data.progress) ? Number(data.progress) : 0,
        etaSeconds:
          data.etaSeconds === null || data.etaSeconds === undefined
            ? null
            : Number(data.etaSeconds),
        error: data.error ?? null,
        stage:
          data.stage === null || data.stage === undefined
            ? null
            : String(data.stage),
        result: data.resultFileId ?? null,
      };
    } catch {
      return { status: "idle", progress: 0, etaSeconds: null, error: "network" };
    }
  }

  async cancelTask(taskId: string): Promise<boolean> {
    try {
      const res = await this._request(
        `${this._baseURL}/task/${encodeURIComponent(taskId)}`,
        {
          method: "delete",
          headers: this._headers(),
        }
      );
      if (!res.ok) {
        return false;
      }
      const data = await res.json();
      return Boolean(data?.success);
    } catch {
      return false;
    }
  }

  _withTokenQuery(url: string) {
    if (!this._token) {
      return url;
    }
    const separator = url.includes("?") ? "&" : "?";
    return `${url}${separator}token=${encodeURIComponent(this._token)}`;
  }

  buildFileUrl(fileId: string) {
    return this._withTokenQuery(`${this._baseURL}/file/${encodeURIComponent(fileId)}`);
  }

  buildLibraryUrl(fileName: string) {
    return this._withTokenQuery(
      `${this._baseURL}/library/${encodeURIComponent(fileName)}`
    );
  }

  buildDownloadUrl(fileId: string) {
    return this._withTokenQuery(`${this._baseURL}/download/${encodeURIComponent(fileId)}`);
  }

  async downloadResult(fileId: string, filename?: string) {
    const url = this.buildDownloadUrl(fileId);
    try {
      const probe = await this._request(
        url,
        { method: "HEAD", headers: this._headers() },
        PROGRESS_REQUEST_TIMEOUT_MS
      );
      if (!probe.ok) {
        return false;
      }
      const a = document.createElement("a");
      a.href = url;
      a.download = filename || "result";
      document.body.appendChild(a);
      a.click();
      a.remove();
      return true;
    } catch {
      return false;
    }
  }
}

export const WebServerClient = new WebServer();
