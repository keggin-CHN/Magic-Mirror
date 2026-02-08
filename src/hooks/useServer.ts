import { useCallback, useEffect } from "react";
import { useXState, XSta } from "xsta";
import { Server, ServerStatus } from "../services/server";
import { sleep } from "../services/utils";

const kStatusKey = "serverStatus";

// 防止重复启动的 Promise 缓存
let launchPromise: Promise<boolean> | null = null;

export function useServer() {
  const [status, setStatus] = useXState<ServerStatus>(kStatusKey, "idle");

  const launch = async () => {
    // 如果已经有启动任务在进行，返回同一个 Promise
    if (launchPromise) {
      return launchPromise;
    }

    if (status !== "idle") {
      console.warn(`[useServer] 服务器已在运行或正在启动，当前状态: ${status}`);
      return true;
    }

    launchPromise = (async () => {
      try {
        setStatus("launching");
        const launched = await Server.launch(() => {
          setStatus("idle");
        });
        if (!launched) {
          setStatus("idle");
          return false;
        }

        // 使用超时机制避免无限等待
        const maxWaitTime = 30000; // 30秒超时
        const startTime = Date.now();

        while (XSta.get(kStatusKey) === "launching") {
          if (Date.now() - startTime > maxWaitTime) {
            console.error("[useServer] 服务器启动超时");
            setStatus("idle");
            return false;
          }

          const status = await Server.status();
          if (status === "running") {
            break;
          }
          await sleep(200);
        }

        const prepared = await Server.prepare();
        if (prepared) {
          setStatus("running");
        } else {
          setStatus("idle");
        }
        return prepared;
      } finally {
        launchPromise = null;
      }
    })();

    return launchPromise;
  };

  const kill = useCallback(() => {
    setStatus("idle");
    Server.kill();
  }, []);

  useEffect(() => {
    return () => kill();
  }, [kill]);

  return { status, launch, kill };
}
