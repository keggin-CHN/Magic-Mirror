import banner from "@/assets/images/magic-mirror.svg";
import { ProgressBar } from "@/components/ProgressBar";
import { useDownload } from "@/hooks/useDownload";
import { useServer } from "@/hooks/useServer";
import { openExternalSafe } from "@/services/tauriBridge";
import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

export default function LaunchDesktop() {
  const { t } = useTranslation();
  const { progress, download, status: downloadStatus } = useDownload();
  const { launch, status: launchingStatus } = useServer();

  const launchingStatusRef = useRef(launchingStatus);
  launchingStatusRef.current = launchingStatus;

  const navigate = useNavigate();

  useEffect(() => {
    download();
  }, [download]);

  useEffect(() => {
    if (downloadStatus === "success") {
      launch();
      Promise.all([
        new Promise((resolve) => {
          setTimeout(resolve, 3000);
        }),
        new Promise((resolve) => {
          const checkInterval = setInterval(() => {
            if (launchingStatusRef.current === "running") {
              clearInterval(checkInterval);
              resolve(true);
            }
          }, 100);
        }),
      ]).then(() => {
        navigate("/mirror");
      });
    }
  }, [downloadStatus, launch, navigate]);

  const launching =
    ["idle", "success"].includes(downloadStatus) ||
      ["launching", "running"].includes(launchingStatus) ? (
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
      {launching}
      {downloading}
    </div>
  );
}