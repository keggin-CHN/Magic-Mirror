import { useCallback, useEffect } from "react";
import { useXState, XSta } from "xsta";
import { Server, ServerStatus } from "../services/server";
import { sleep } from "../services/utils";

const kStatusKey = "serverStatus";

export function useServer() {
  const [status, setStatus] = useXState<ServerStatus>(kStatusKey, "idle");

  const launch = async () => {
    if (status != "idle") {
      return true;
    }

    setStatus("launching");
    const launched = await Server.launch(() => {
      setStatus("idle");
    });
    if (!launched) {
      setStatus("idle");
      return false;
    }

    while (XSta.get(kStatusKey) === "launching") {
      const status = await Server.status();
      if (status === "running") {
        break;
      }
      await sleep(200);
    }

    const prepared = await Server.prepare();
    if (prepared) {
      setStatus("running");
    }
    return prepared;
  };

  const kill = useCallback(() => {
    setStatus("idle");
    Server.kill();
  }, []);

  useEffect(() => {
    return () => kill();
  }, []);

  return { status, launch, kill };
}
