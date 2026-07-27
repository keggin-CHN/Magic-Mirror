import banner from "@/assets/images/magic-mirror.svg";
import { ProgressBar } from "@/components/ProgressBar";
import { useDownload } from "@/hooks/useDownload";
import { useServer } from "@/hooks/useServer";
import { openExternalSafe } from "@/services/tauriBridge";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

const kLaunchTimeoutMs = 60000;

export default function LaunchDesktop() {
  const { t } = useTranslation();
  const { progress, download, status: downloadStatus } = useDownload();
  const { launch, status: launchingStatus } = useServer();
  const [launchFailed, setLaunchFailed] = useState(false);
  const [launchAttempt, setLaunchAttempt] = useState(0);

  const launchingStatusRef = useRef(launchingStatus);
  launchingStatusRef.current = launchingStatus;

  const navigate = useNavigate();

  useEffect(() => {
    download();
  }, [download]);

  useEffect(() => {
    if (downloadStatus !== "success") {
      return;
    }
    let cancelled = false;
    setLaunchFailed(false);
    launch().then((launched) => {
      if (!cancelled && !launched) {
        setLaunchFailed(true);
      }
    });
    const startedAt = Date.now();
    const checkInterval = window.setInterval(() => {
      if (cancelled) {
        return;
      }
      if (
        launchingStatusRef.current === "running" &&
        Date.now() - startedAt >= 3000
      ) {
        window.clearInterval(checkInterval);
        navigate("/mirror");
        return;
      }
      if (Date.now() - startedAt > kLaunchTimeoutMs) {
        window.clearInterval(checkInterval);
        setLaunchFailed(true);
      }
    }, 100);
    return () => {
      cancelled = true;
      window.clearInterval(checkInterval);
    };
  }, [downloadStatus, launch, navigate, launchAttempt]);

  const failed = launchFailed ? (
    <>
      <p className="c-#ff6b6b">{t("Failed to start the server. Please retry.")}</p>
      <button
        className="cursor-pointer bg-transparent c-blue border-none text-14px"
        onClick={() => setLaunchAttempt((attempt) => attempt + 1)}
      >
        {t("Retry")}
      </button>
    </>
  ) : null;

  const launching =
    !launchFailed &&
      (["idle", "success"].includes(downloadStatus) ||
        ["launching", "running"].includes(launchingStatus)) ? (
      <>
        <p>{t("Starting... First load may take longer, please wait.")}</p>
      </>
    ) : null;

  const downloading = ["downloading", "unzipping", "failed"].includes(
    downloadStatus
  ) ? (
    <>
      <p>
        {t("Downloading resources, please wait", {
          progress: progress.toFixed(2),
        })}
      </p>
      <ProgressBar progress={progress} />
      <p className="c-[rgba(255,255,255,0.6)] text-12px">
        {t(
          "*If the download is stuck or fails, please download and initialize manually. "
        )}
        <span
          className="c-blue cursor-pointer"
          onClick={() => openExternalSafe(t("downloadTutorial"))}
        >
          {t("View tutorial")}
        </span>
      </p>
    </>
  ) : null;

  return (
    <div
      data-tauri-drag-region
      style={{
        border: "1px solid rgba(0, 0, 0, 0.1)",
        boxShadow:
          "0 4px 10px rgba(0, 0, 0, 0.3), 0 8px 20px rgba(0, 0, 0, 0.3)",
      }}
      className="w-540px h-320px bg-#151515 color-white flex-col-c-c gap-8px p-10px"
    >
      <img
        src={banner}
        className="w-80% object-cover cursor-default pointer-events-none select-none"
      />
      {failed}
      {launching}
      {downloading}
    </div>
  );
}
