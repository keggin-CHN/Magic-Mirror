import { useCallback, useEffect } from "react";
import { useXState } from "xsta";
import { Server, type Task, type TaskResult, type VideoTask } from "../services/server";

const kSwapFaceRefs: {
  id: number;
  cancel?: VoidFunction;
} = {
  id: 1,
  cancel: undefined,
};

export function useSwapFace() {
  const [isSwapping, setIsSwapping] = useXState("isSwapping", false);
  const [output, setOutput] = useXState<string | null>("swapOutput", null);
  const [error, setError] = useXState<string | null>("swapError", null);
  const [videoProgress, setVideoProgress] = useXState("videoSwapProgress", 0);
  const [videoEtaSeconds, setVideoEtaSeconds] = useXState<number | null>(
    "videoSwapEtaSeconds",
    null
  );
  const [videoStage, setVideoStage] = useXState<string | null>("videoSwapStage", null);
  const runTask = useCallback(
    async (create: (taskId: string) => Promise<TaskResult>) => {
      await kSwapFaceRefs.cancel?.();
      setIsSwapping(true);
      setError(null);
      const taskId = (kSwapFaceRefs.id++).toString();
      kSwapFaceRefs.cancel = async () => {
        const success = await Server.cancelTask(taskId);
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
    []
  );

  const swapFace = useCallback(
    async (task: Omit<Task, "id">) => {
      setVideoProgress(0);
      setVideoEtaSeconds(null);
      setVideoStage(null);
      return runTask((taskId: string) =>
        Server.createTask({
          id: taskId,
          ...task,
        })
      );
    },
    [runTask, setVideoEtaSeconds, setVideoProgress, setVideoStage]
  );

  const swapVideo = useCallback(
    async (task: Omit<VideoTask, "id">) => {
      await kSwapFaceRefs.cancel?.();

      setIsSwapping(true);
      setError(null);
      setVideoProgress(0);
      setVideoEtaSeconds(null);
      setVideoStage("queued");

      const taskId = (kSwapFaceRefs.id++).toString();
      let polling = true;
      let finalResult: string | null = null;

      const pollProgress = async () => {
        while (polling) {
          try {
            const state = await Server.getVideoTaskProgress(taskId);

            // 处理所有状态，包括 "queued"
            if (state.status === "queued" || state.status === "running" || state.status === "success") {
              setVideoProgress(state.progress ?? 0);
              setVideoEtaSeconds(state.etaSeconds ?? null);
              setVideoStage(state.stage ?? null);
              if (state.status === "success" && state.result) {
                finalResult = state.result;
                polling = false;
                break;
              }
            } else if (state.status === "failed") {
              setVideoEtaSeconds(null);
              setVideoStage(state.stage ?? "failed");
              setError(state.error ?? "unknown");
              polling = false;
              break;
            } else if (state.status === "cancelled") {
              setVideoEtaSeconds(null);
              setVideoStage(state.stage ?? "cancelled");
              polling = false;
              break;
            }

            // 轮询间隔优化：根据进度动态调整
            const interval = state.progress > 0 ? 500 : 1000;
            await new Promise((resolve) => setTimeout(resolve, interval));
          } catch (error) {
            console.error("[useSwapFace] 轮询进度失败:", error);
            // 网络错误时继续轮询，但增加间隔
            await new Promise((resolve) => setTimeout(resolve, 2000));
          }
        }
      };

      const pollPromise = pollProgress();

      kSwapFaceRefs.cancel = async () => {
        polling = false;
        const success = await Server.cancelTask(taskId);
        if (success) {
          setIsSwapping(false);
          setVideoEtaSeconds(null);
          setVideoStage("cancelled");
        }
      };

      const { result, error } = await Server.createVideoTask({
        id: taskId,
        ...task,
      });

      // If the backend returns immediately (queued), we don't have the result yet.
      // We rely on polling to get the result.
      if (result) {
        // If backend returned result immediately (old behavior or fast task)
        setVideoProgress(100);
        setVideoEtaSeconds(0);
        setVideoStage("done");
        setOutput(result);
      } else if (error) {
        // Immediate error
        setError(error);
        setVideoStage("failed");
        polling = false; // Stop polling
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
        return finalResult;
      }

      setIsSwapping(false);
      return result;
    },
    [
      setError,
      setIsSwapping,
      setOutput,
      setVideoEtaSeconds,
      setVideoProgress,
      setVideoStage,
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
    swapFace,
    swapVideo,
    cancel: () => kSwapFaceRefs.cancel?.(),
  };
}
