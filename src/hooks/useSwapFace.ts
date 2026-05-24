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
  id: number;
  cancel?: () => Promise<void>;
} = {
  id: 1,
  cancel: undefined,
};

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
  const runTask = useCallback(
    async (create: (taskId: string) => Promise<TaskResult>) => {
      await kSwapFaceRefs.cancel?.();
      setIsSwapping(true);
      setError(null);
      const taskId = (kSwapFaceRefs.id++).toString();
      kSwapFaceRefs.cancel = async () => {
        const success = await client.cancelTask(taskId);
        if (success) {
          setIsSwapping(false);
        }
      };
      const { result, error } = await create(taskId);
      kSwapFaceRefs.cancel = undefined;
      const finalError = result ? null : error ?? "unknown";
      setError(finalError);
      setOutput(result);
      setIsSwapping(false);
      return result;
    },
    [client, setError, setIsSwapping, setOutput]
  );

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
    [
      client,
      runTask,
      setVideoEtaSeconds,
      setVideoProgress,
      setVideoStage,
      setVideoTaskConfigId,
    ]
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

      const taskId = (kSwapFaceRefs.id++).toString();
      // 使用对象引用来控制轮询，确保可以从外部停止
      const pollingControl = { shouldStop: false };
      let finalResult: string | null = null;

      const pollProgress = async () => {
        // 按真实经过时间限制（默认 4 小时），避免长任务被误杀
        const maxDurationMs = 4 * 60 * 60 * 1000;
        const startedAt = Date.now();
        // 连续网络错误指数退避控制
        let consecutiveErrors = 0;
        const maxConsecutiveErrors = 8;
        let lastProgress = -1;
        let stableCount = 0;

        while (!pollingControl.shouldStop) {
          if (Date.now() - startedAt > maxDurationMs) {
            console.error("[useSwapFace] polling timeout");
            setError("polling-timeout");
            pollingControl.shouldStop = true;
            break;
          }

          try {
            const state = await client.getVideoTaskProgress(taskId);
            consecutiveErrors = 0;

            if (
              state.status === "queued" ||
              state.status === "running" ||
              state.status === "success"
            ) {
              setVideoProgress(state.progress ?? 0);
              setVideoEtaSeconds(state.etaSeconds ?? null);
              setVideoStage(state.stage ?? null);
              if (state.status === "success" && state.result) {
                finalResult = state.result;
                pollingControl.shouldStop = true;
                break;
              }
            } else if (state.error) {
              throw new Error(state.error);
            } else if (state.status === "failed") {
              setVideoEtaSeconds(null);
              setVideoStage(state.stage ?? "failed");
              setError(state.error ?? "unknown");
              pollingControl.shouldStop = true;
              break;
            } else if (state.status === "cancelled") {
              setVideoEtaSeconds(null);
              setVideoStage(state.stage ?? "cancelled");
              pollingControl.shouldStop = true;
              break;
            }

            // 自适应轮询间隔：进度推进时 500ms，停滞时逐步放大到 2s
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
            await new Promise((resolve) => setTimeout(resolve, interval));
          } catch (err) {
            consecutiveErrors++;
            if (consecutiveErrors === 1 || consecutiveErrors >= maxConsecutiveErrors) {
              console.error(
                `[useSwapFace] poll failed (${consecutiveErrors}/${maxConsecutiveErrors}):`,
                err
              );
            }
            if (consecutiveErrors >= maxConsecutiveErrors) {
              setError("network-error");
              pollingControl.shouldStop = true;
              break;
            }
            // 指数退避: 1s, 2s, 4s, 8s, 16s, 封顶 30s
            const backoff = Math.min(
              1000 * Math.pow(2, consecutiveErrors - 1),
              30000
            );
            await new Promise((resolve) => setTimeout(resolve, backoff));
          }
        }
      };

      const pollPromise = pollProgress();

      kSwapFaceRefs.cancel = async () => {
        pollingControl.shouldStop = true;
        const success = await client.cancelTask(taskId);
        if (success) {
          setIsSwapping(false);
          setVideoEtaSeconds(null);
          setVideoStage("cancelled");
        }
      };

      const { result, error, configId, status } = await client.createVideoTask({
        id: taskId,
        ...(task as any),
      });

      setVideoTaskConfigId(configId ?? null);

      // If the backend returns immediately (queued), we don't have the result yet.
      // We rely on polling to get the result.
      if (result) {
        // If backend returned result immediately (old behavior or fast task)
        pollingControl.shouldStop = true;
        setVideoProgress(100);
        setVideoEtaSeconds(0);
        setVideoStage("done");
        setOutput(result);
      } else if (error) {
        // Immediate error
        setError(error);
        setVideoStage("failed");
        pollingControl.shouldStop = true; // Stop polling
      } else if ((task as any).dryRunConfigOnly || status === "config-only") {
        pollingControl.shouldStop = true;
        setVideoProgress(0);
        setVideoEtaSeconds(null);
        setVideoStage("config-only");
      }

      // Wait for polling to finish (it finishes when status is success/failed/cancelled)
      await pollPromise;

      kSwapFaceRefs.cancel = undefined;

      if (finalResult) {
        setVideoProgress(100);
        setVideoEtaSeconds(0);
        setVideoStage("done");
        setOutput(finalResult);
        setIsSwapping(false);
        return {
          result: finalResult,
          configId: configId ?? null,
          status: status ?? "success",
        };
      }

      setIsSwapping(false);
      return {
        result: result ?? null,
        error,
        configId: configId ?? null,
        status: status ?? null,
      };
    },
    [
      client,
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
      kSwapFaceRefs.cancel?.();
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
