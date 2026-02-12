import type {
  DetectFacesResult,
  Region,
  Task,
  TaskResult,
  VideoTask,
  VideoTaskProgress,
} from "./server";

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

class WebServer {
  _baseURL = "/api";
  _token: string | null = null;

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
    const headers: Record<string, string> = {
      ...(extra || {}),
    } as Record<string, string>;
    if (this._token) {
      headers.Authorization = `Bearer ${this._token}`;
    }
    return headers;
  }

  async login(password: string) {
    const res = await fetch(`${this._baseURL}/login`, {
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
  }

  async updateCredential(password: string) {
    const res = await fetch(`${this._baseURL}/credential`, {
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
    return { success: Boolean(data?.success), error: data?.error };
  }

  async uploadFile(file: File): Promise<UploadResult | null> {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${this._baseURL}/upload`, {
      method: "post",
      headers: this._headers(),
      body: form,
    });
    if (!res.ok) {
      return null;
    }
    return (await res.json()) as UploadResult;
  }

  async uploadLibrary(file: File): Promise<LibraryItem | null> {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${this._baseURL}/library/upload`, {
      method: "post",
      headers: this._headers(),
      body: form,
    });
    if (!res.ok) {
      return null;
    }
    return (await res.json()) as LibraryItem;
  }

  async listLibrary(): Promise<LibraryItem[]> {
    const res = await fetch(`${this._baseURL}/library`, {
      method: "get",
      headers: this._headers(),
    });
    if (!res.ok) {
      return [];
    }
    const data = await res.json();
    return Array.isArray(data?.items) ? data.items : [];
  }

  async detectImageFaces(
    inputFileId: string,
    regions?: Region[]
  ): Promise<DetectFacesResult> {
    try {
      const res = await fetch(`${this._baseURL}/task/detect-faces`, {
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
      const res = await fetch(`${this._baseURL}/task/video/detect-faces`, {
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

  async createTask(task: Omit<Task, "id"> & { id: string } & { inputFileId: string; targetFaceId?: string }) {
    try {
      const res = await fetch(`${this._baseURL}/task`, {
        method: "post",
        headers: this._headers({
          "Content-Type": "application/json;charset=UTF-8",
        }),
        body: JSON.stringify({
          id: task.id,
          inputFileId: task.inputFileId,
          targetFaceId: task.targetFaceId,
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
      const res = await fetch(`${this._baseURL}/task/video`, {
        method: "post",
        headers: this._headers({
          "Content-Type": "application/json;charset=UTF-8",
        }),
        body: JSON.stringify({
          id: task.id,
          inputFileId: task.inputFileId,
          targetFaceId: task.targetFaceId,
          regions: task.regions,
          faceSources: task.faceSources,
          keyFrameMs: task.keyFrameMs,
          useGpu: task.useGpu,
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
        return { result: null, error: data.error } as TaskResult;
      }
      return { result: data.resultFileId || null } as TaskResult;
    } catch {
      return { result: null, error: "network" } as TaskResult;
    }
  }

  async getVideoTaskProgress(taskId: string): Promise<VideoTaskProgress> {
    try {
      const res = await fetch(
        `${this._baseURL}/task/video/progress/${encodeURIComponent(taskId)}`,
        {
          method: "get",
          headers: this._headers(),
        }
      );
      if (!res.ok) {
        return { status: "idle", progress: 0, etaSeconds: null };
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
      return { status: "idle", progress: 0, etaSeconds: null };
    }
  }

  async cancelTask(taskId: string): Promise<boolean> {
    try {
      const res = await fetch(`${this._baseURL}/task/${taskId}`, {
        method: "delete",
        headers: this._headers(),
      });
      const data = await res.json();
      return data.success || false;
    } catch {
      return false;
    }
  }

  buildFileUrl(fileId: string) {
    return `${this._baseURL}/file/${encodeURIComponent(fileId)}`;
  }

  buildDownloadUrl(fileId: string) {
    return `${this._baseURL}/download/${encodeURIComponent(fileId)}`;
  }

  async downloadResult(fileId: string, filename?: string) {
    const url = this.buildDownloadUrl(fileId);
    const res = await fetch(url, {
      method: "get",
      headers: this._headers(),
    });
    if (!res.ok) {
      return false;
    }
    const blob = await res.blob();
    const objectUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = objectUrl;
    a.download = filename || "result";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(objectUrl);
    return true;
  }
}

export const WebServerClient = new WebServer();