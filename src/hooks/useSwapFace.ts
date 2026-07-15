import { useCallback, useEffect } from "react";
import { useXState } from "xsta";
import {
  Server,
  type FaceSource,
  type Region,
  type Task,
  type TaskResult,
  type VideoTask,
} from "../services/server";
import { WebServerClient } from "@/services/webServer";
import { isTauri } from "@/services/runtime";

const kSwapFaceRefs: {
  activeTaskId: string | null;
  cancel?: () => Promise<void>;
} = {
  activeTaskId: null,
  cancel: undefined,
};

function createTaskId() {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `${Date.now()}-${Math.random().toString(36).slice(2)}`
  );
}

type WebImageTask = {
  inputFileId: string;
  targetFaceId?: string;
  targetFaces?: FaceSource[];
  deepSwapMode?: boolean;
  regions?: Region[];
  faceSources?: FaceSource[];
};

type WebVideoTask = {
  inputFileId: string;
  targetFaceId?: string;
  targetFaces?: FaceSource[];
  deepSwapMode?: boolean;
  segmentDurationSec?: number;
  segmentOverlapFrames?: number;
  regions?: Region[];
  faceSources?: FaceSource[];
  keyFrameMs?: number;
  useGpu?: boolean;
  gpuProvider?: "auto" | "cpu" | "directml" | "cuda";
  configId?: string;
  generateConfigId?: boolean;
  dryRunConfigOnly?: boolean;
};

type AnyImageTask = Omit<Task, "id"> | WebImageTask;
type AnyVideoTask = Omit<VideoTask, "id"> | WebVideoTask;

export function useSwapFace() {
  const client = isTauri() ? Server : WebServerClient;
  const [isSwapping, setIsSwapping] = useXState("isSwapping", false);
  const [output, setOutput] = useXState<string | null>("swapOutput", null);
  const [error, setError] = useXState<string | null>("swapError", null);
  const [videoProgress, setVideoProgress] = useXState("videoSwapProgress", 0);
  const [videoEtaSeconds, setVideoEtaSeconds] = useXState<number | null>(
    "videoSwapEtaSeconds",
    null
  );
  const [videoStage, setVideoStage] = useXState<string | null>("videoSwapStage", null);
  const [videoTaskConfigId, setVideoTaskConfigId] = useXState<string | null>(
    "videoTaskConfigId",
    null
  );

  const isCurrentTask = useCallback(
    (taskId: string) => kSwapFaceRefs.activeTaskId === taskId,
    []
  );

  const runTask = useCallback(
    async (create: (taskId: string) => Promise<TaskResult>) => {
      await kSwapFaceRefs.cancel?.();

      setIsSwapping(true);
      setError(null);
      const taskId = createTaskId();
      kSwapFaceRefs.activeTaskId = taskId;
      kSwapFaceRefs.cancel = async () => {
        await client.cancelTask(taskId);
        if (isCurrentTask(taskId)) {
          setIsSwapping(false);
        }
      };

      try {
        const { result, error } = await create(taskId);
        if (!isCurrentTask(taskId)) {
          return null;
        }
        const finalError = result ? null : error ?? "unknown";
        setError(finalError);
        setOutput(result);
        setIsSwapping(false);
        return result;
      } finally {
        if (isCurrentTask(taskId)) {
          kSwapFaceRefs.activeTaskId = null;
          kSwapFaceRefs.cancel = undefined;
        }
      }
    }, [client, isCurrentTask, setError, setIsSwapping, setOutput]);

  const swapFace = useCallback(
    async (task: AnyImageTask) => {
      setVideoProgress(0);
      setVideoEtaSeconds(null);
      setVideoStage(null);
      setVideoTaskConfigId(null);
      return runTask((taskId: string) =>
        client.createTask({
          id: taskId,
          ...(task as any),
        })
      );
    },
    [client, runTask, setVideoEtaSeconds, setVideoProgress, setVideoStage, setVideoTaskConfigId]
  );

  const swapVideo = useCallback(
    async (task: AnyVideoTask): Promise<TaskResult> => {
      await kSwapFaceRefs.cancel?.();

      setIsSwapping(true);
      setError(null);
      setVideoProgress(0);
      setVideoEtaSeconds(null);
      setVideoStage(null);
      setVideoTaskConfigId(null);

      const taskId = createTaskId();
      kSwapFaceRefs.activeTaskId = taskId;
      const pollingControl: {
        shouldStop: boolean;
        timer: ReturnType<typeof setTimeout> | null;
        resolveWait: (() => void) | null;
      } = { shouldStop: false, timer: null, resolveWait: null };
      let cancelRequested = false;
      let taskCreated = false;
      let finalResult: string | null = null;

      const cancel = async () => {
        cancelRequested = true;
        pollingControl.shouldStop = true;
        if (pollingControl.timer !== null) {
          clearTimeout(pollingControl.timer);
          pollingControl.timer = null;
        }
        pollingControl.resolveWait?.();
        pollingControl.resolveWait = null;
        if (taskCreated) {
          await client.cancelTask(taskId);
        }
        if (isCurrentTask(taskId)) {
          setIsSwapping(false);
          setVideoEtaSeconds(null);
          setVideoStage("cancelled");
        }
      };
      kSwapFaceRefs.cancel = cancel;

      const waitForPoll = (delayMs: number) =>
        new Promise<void>((resolve) => {
          if (pollingControl.shouldStop) {
            resolve();
            return;
          }
          pollingControl.resolveWait = resolve;
          pollingControl.timer = setTimeout(() => {
            pollingControl.timer = null;
            pollingControl.resolveWait = null;
            resolve();
          }, delayMs);
        });

      const pollProgress = async () => {
        const maxDurationMs = 4 * 60 * 60 * 1000;
        const startedAt = Date.now();
        let consecutiveErrors = 0;
        const maxConsecutiveErrors = 8;
        const maxConsecutiveIdle = 8;
        let consecutiveIdle = 0;
        let lastProgress = -1;
        let stableCount = 0;

        while (!pollingControl.shouldStop && isCurrentTask(taskId)) {
          if (Date.now() - startedAt > maxDurationMs) {
            setError("polling-timeout");
            pollingControl.shouldStop = true;
            break;
          }

          try {
            const state = await client.getVideoTaskProgress(taskId);
            if (!isCurrentTask(taskId)) {
              break;
            }
            if (state.status === "failed") {
              setVideoEtaSeconds(null);
              setVideoStage(state.stage ?? "failed");
              setError(state.error ?? "unknown");
              pollingControl.shouldStop = true;
              break;
            }
            if (state.status === "cancelled") {
              setVideoEtaSeconds(null);
              setVideoStage(state.stage ?? "cancelled");
              pollingControl.shouldStop = true;
              break;
            }
            if (state.error) {
              throw new Error(state.error);
            }
            consecutiveErrors = 0;
            if (state.status === "idle") {
              consecutiveIdle += 1;
              if (consecutiveIdle >= maxConsecutiveIdle) {
                setError("task-not-found");
                pollingControl.shouldStop = true;
                break;
              }
            } else {
              consecutiveIdle = 0;
            }

            if (
              state.status === "queued" ||
              state.status === "running" ||
              state.status === "success"
            ) {
              setVideoProgress(state.progress ?? 0);
              setVideoEtaSeconds(state.etaSeconds ?? null);
              setVideoStage(state.stage ?? null);
              if (state.status === "success") {
                if (!state.result) {
                  throw new Error("missing-result");
                }
                finalResult = state.result;
                pollingControl.shouldStop = true;
                break;
              }
            }

            const currentProgress = state.progress ?? 0;
            if (currentProgress !== lastProgress) {
              lastProgress = currentProgress;
              stableCount = 0;
            } else {
              stableCount++;
            }
            const interval =
              currentProgress > 0
                ? Math.min(500 + stableCount * 200, 2000)
                : 1000;
            await waitForPoll(interval);
          } catch (pollError) {
            if (!isCurrentTask(taskId) || pollingControl.shouldStop) {
              break;
            }
            consecutiveErrors++;
            if (consecutiveErrors >= maxConsecutiveErrors) {
              setError("network-error");
              pollingControl.shouldStop = true;
              break;
            }
            const backoff = Math.min(
              1000 * Math.pow(2, consecutiveErrors - 1),
              30000
            );
            console.warn("[useSwapFace] video poll failed", pollError);
            await waitForPoll(backoff);
          }
        }
      };

      try {
        const { result, error, configId, status } = await client.createVideoTask({
          id: taskId,
          ...(task as any),
        });
        taskCreated = true;

        if (cancelRequested) {
          await client.cancelTask(taskId);
          return { result: null, error: "cancelled", status: "cancelled" };
        }
        if (!isCurrentTask(taskId)) {
          return { result: null, error: "cancelled", status: "cancelled" };
        }
        setVideoTaskConfigId(configId ?? null);

        if (result) {
          setVideoProgress(100);
          setVideoEtaSeconds(0);
          setVideoStage("done");
          setOutput(result);
          pollingControl.shouldStop = true;
        } else if (error === "network") {
          // The server may have accepted the task even if the POST response was
          // lost. Recover by polling the UUID instead of abandoning the job.
          setVideoStage("recovering");
          await pollProgress();
        } else if (error) {
          setError(error);
          setVideoStage("failed");
          pollingControl.shouldStop = true;
        } else if ((task as any).dryRunConfigOnly || status === "config-only") {
          setVideoStage("config-only");
          pollingControl.shouldStop = true;
        } else {
          await pollProgress();
        }

        if (!isCurrentTask(taskId)) {
          return { result: null, error: "cancelled", status: "cancelled" };
        }
        if (finalResult) {
          setVideoProgress(100);
          setVideoEtaSeconds(0);
          setVideoStage("done");
          setOutput(finalResult);
          setIsSwapping(false);
          return { result: finalResult, configId: configId ?? null, status: status ?? "success" };
        }

        setIsSwapping(false);
        return { result: result ?? null, error, configId: configId ?? null, status: status ?? null };
      } catch (taskError) {
        if (isCurrentTask(taskId)) {
          setError("network-error");
          setVideoStage("failed");
          setIsSwapping(false);
        }
        return { result: null, error: taskError instanceof Error ? taskError.message : "network" };
      } finally {
        if (pollingControl.timer !== null) {
          clearTimeout(pollingControl.timer);
          pollingControl.timer = null;
        }
        pollingControl.resolveWait?.();
        pollingControl.resolveWait = null;
        if (isCurrentTask(taskId)) {
          kSwapFaceRefs.activeTaskId = null;
          kSwapFaceRefs.cancel = undefined;
        }
      }
    },
    [
      client,
      isCurrentTask,
      setError,
      setIsSwapping,
      setOutput,
      setVideoEtaSeconds,
      setVideoProgress,
      setVideoStage,
      setVideoTaskConfigId,
    ]
  );

  useEffect(() => {
    return () => {
      void kSwapFaceRefs.cancel?.();
    };
  }, []);

  return {
    isSwapping,
    output,
    error,
    videoProgress,
    videoEtaSeconds,
    videoStage,
    videoTaskConfigId,
    swapFace,
    swapVideo,
    cancel: () => kSwapFaceRefs.cancel?.(),
  };
}
