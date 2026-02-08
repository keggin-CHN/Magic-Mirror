import { ProgressBar } from "@/components/ProgressBar";
import { useDragDrop } from "@/hooks/useDragDrop";
import { useSwapFace } from "@/hooks/useSwapFace";
import { Server, type FaceSource, type Region } from "@/services/server";
import { getFileExtension, isImageFile, isVideoFile } from "@/services/utils";
import { convertFileSrc } from "@tauri-apps/api/core";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { exit } from "@tauri-apps/plugin-process";
import { open as openExternal } from "@tauri-apps/plugin-shell";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type PointerEvent,
  type WheelEvent,
} from "react";
import { useTranslation } from "react-i18next";

import "@/styles/mirror.css";

import iconMenu from "@/assets/images/menu.webp";
import background from "@/assets/images/mirror-bg.svg";
import mirrorInput from "@/assets/images/mirror-input.webp";
import mirrorMe from "@/assets/images/mirror-me.webp";
interface Asset {
  path: string;
  src: string;
  type?: "image" | "video";
}

interface FaceAsset extends FaceSource {
  src: string;
  locked?: boolean;
  name?: string;
}

type ResizeHandle = "nw" | "ne" | "sw" | "se";

const kDefaultFaceSourceId = "default-me";
const kMinSelectionZoom = 0.5;
const kMaxSelectionZoom = 4;

const kMirrorStates: {
  isMe: boolean;
  me?: Asset;
  input?: Asset;
  result?: Asset;
} = { isMe: true };

export function MirrorPage() {
  const [flag, setFlag] = useState(false);
  const rebuild = useRef<any>();
  rebuild.current = () => setFlag(!flag);

  const { i18n, t } = useTranslation();

  const {
    isSwapping,
    swapFace,
    swapVideo,
    error: swapError,
    videoProgress,
    videoEtaSeconds,
    videoStage,
  } = useSwapFace();
  const [success, setSuccess] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);
  const [regions, setRegions] = useState<Region[]>([]);
  const [draftRegion, setDraftRegion] = useState<Region | null>(null);
  const [inputSize, setInputSize] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const [isEditingRegions, setIsEditingRegions] = useState(false);
  const previewRef = useRef<HTMLDivElement | null>(null);
  const selectingRef = useRef(false);
  const startPointRef = useRef<{ x: number; y: number } | null>(null);
  const inputPathRef = useRef<string | null>(null);
  const [selectedRegionIndex, setSelectedRegionIndex] = useState<number | null>(
    null
  );
  const resizeRef = useRef<{
    index: number;
    handle: ResizeHandle;
    startX: number;
    startY: number;
    origin: Region;
  } | null>(null);
  const moveRef = useRef<{
    index: number;
    startX: number;
    startY: number;
    origin: Region;
  } | null>(null);
  const [isMultiFaceMode, setIsMultiFaceMode] = useState(false);
  const [faceSources, setFaceSources] = useState<FaceAsset[]>([]);
  const faceSourceInputRef = useRef<HTMLInputElement | null>(null);
  const previewVideoRef = useRef<HTMLVideoElement | null>(null);
  const autoDetectedImagePathRef = useRef<string | null>(null);
  const [videoDurationMs, setVideoDurationMs] = useState(0);
  const [videoKeyFrameMs, setVideoKeyFrameMs] = useState(0);
  const [isDetectingFaces, setIsDetectingFaces] = useState(false);
  const [selectionZoom, setSelectionZoom] = useState(1);
  const [activeFaceSourceId, setActiveFaceSourceId] = useState<string | null>(null);

  useEffect(() => {
    setTimeout(() => {
      if (kMirrorStates.me && !kMirrorStates.input) {
        kMirrorStates.isMe = false;
        rebuild.current();
      }
      if (kMirrorStates.me && isSwapping) {
        kMirrorStates.isMe = false;
        rebuild.current();
      }
    });
  }, [kMirrorStates.me, kMirrorStates.input, isSwapping]);

  useEffect(() => {
    const input = kMirrorStates.input;
    if (!input) {
      setInputSize(null);
      setRegions([]);
      setDraftRegion(null);
      setIsEditingRegions(false);
      setSelectedRegionIndex(null);
      setVideoDurationMs(0);
      setVideoKeyFrameMs(0);
      selectingRef.current = false;
      startPointRef.current = null;
      resizeRef.current = null;
      moveRef.current = null;
      setSelectionZoom(1);
      setActiveFaceSourceId(null);
      inputPathRef.current = null;
      autoDetectedImagePathRef.current = null;
      return;
    }

    const inputIdentity = `${input.type || "image"}:${input.path}`;
    if (inputPathRef.current === inputIdentity) {
      return;
    }

    inputPathRef.current = inputIdentity;
    autoDetectedImagePathRef.current = null;
    setRegions([]);
    setDraftRegion(null);
    setIsEditingRegions(true);
    setSelectedRegionIndex(null);
    setInputSize(null);
    setVideoDurationMs(0);
    setVideoKeyFrameMs(0);
    selectingRef.current = false;
    startPointRef.current = null;
    resizeRef.current = null;
    moveRef.current = null;
    setSelectionZoom(1);
    setActiveFaceSourceId(null);

    if (input.type === "image") {
      const img = new Image();
      img.onload = () => {
        setInputSize({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.src = convertFileSrc(input.path);
    }
  }, [kMirrorStates.input?.path, kMirrorStates.input?.type]);

  useEffect(() => {
    const me = kMirrorStates.me;
    if (!isMultiFaceMode || !me) {
      return;
    }
    setFaceSources((prev: FaceAsset[]) => {
      const others = prev.filter((item) => item.id !== kDefaultFaceSourceId);
      return [
        {
          id: kDefaultFaceSourceId,
          path: me.path,
          src: me.src,
          locked: true,
        },
        ...others,
      ];
    });
  }, [isMultiFaceMode, flag]);

  const clamp = (value: number, min: number, max: number) =>
    Math.max(min, Math.min(value, max));

  const isImageInput = kMirrorStates.input?.type === "image";
  const isVideoInput = kMirrorStates.input?.type === "video";
  const canSelect =
    !kMirrorStates.isMe &&
    (isImageInput || isVideoInput) &&
    isEditingRegions &&
    !isSwapping &&
    !!inputSize;
  const showSelection =
    (isImageInput || isVideoInput) &&
    isEditingRegions &&
    !kMirrorStates.isMe &&
    !!inputSize;
  const showToolbar = showSelection && !isSwapping;
  const canStartSwap =
    isVideoInput && !isMultiFaceMode ? true : regions.length > 0;
  const selectionObjectFit: "contain" | "cover" = showSelection
    ? "contain"
    : "cover";
  const selectionPadding = isMultiFaceMode
    ? isVideoInput
      ? "56px 56px 300px"
      : "56px 56px 240px"
    : isVideoInput
      ? "56px 56px 180px"
      : "56px 56px 120px";

  const mapMediaRegionsToScreen = useCallback(
    (mediaRegions: Region[], mediaWidth: number, mediaHeight: number): Region[] => {
      if (!previewRef.current || mediaWidth <= 0 || mediaHeight <= 0) {
        return [];
      }

      const rect = previewRef.current.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) {
        return [];
      }

      const scale = Math.min(rect.width / mediaWidth, rect.height / mediaHeight);
      const displayWidth = mediaWidth * scale;
      const displayHeight = mediaHeight * scale;
      const offsetX = (rect.width - displayWidth) / 2;
      const offsetY = (rect.height - displayHeight) / 2;

      return mediaRegions
        .map((region: Region) => {
          const x = clamp(region.x * scale + offsetX, 0, rect.width - 1);
          const y = clamp(region.y * scale + offsetY, 0, rect.height - 1);
          const width = clamp(region.width * scale, 1, rect.width - x);
          const height = clamp(region.height * scale, 1, rect.height - y);
          return {
            x: Math.round(x),
            y: Math.round(y),
            width: Math.round(width),
            height: Math.round(height),
          };
        })
        .filter((region: Region) => region.width > 1 && region.height > 1);
    },
    [clamp]
  );

  const toImageRegions = useCallback((): Region[] => {
    if (!previewRef.current || !inputSize) {
      console.log("[DEBUG] toImageRegions: previewRef 或 inputSize 为空", { previewRef: previewRef.current, inputSize });
      return [];
    }
    const rect = previewRef.current.getBoundingClientRect();

    // object-fit 的缩放计算（选区编辑态使用 contain，保证整图可见）
    const baseScale =
      selectionObjectFit === "cover"
        ? Math.max(rect.width / inputSize.width, rect.height / inputSize.height)
        : Math.min(rect.width / inputSize.width, rect.height / inputSize.height);

    const scale = baseScale * (showSelection ? selectionZoom : 1);

    const displayWidth = inputSize.width * scale;
    const displayHeight = inputSize.height * scale;
    const offsetX = (rect.width - displayWidth) / 2;
    const offsetY = (rect.height - displayHeight) / 2;

    console.log("[DEBUG] toImageRegions 转换参数:", {
      containerRect: { width: rect.width, height: rect.height },
      inputSize,
      scale,
      baseScale,
      selectionZoom,
      displaySize: { width: displayWidth, height: displayHeight },
      offset: { x: offsetX, y: offsetY },
      screenRegions: regions,
    });

    const imageRegions = regions
      .map((region: Region, idx: number) => {
        const x = Math.round((region.x - offsetX) / scale);
        const y = Math.round((region.y - offsetY) / scale);
        const width = Math.round(region.width / scale);
        const height = Math.round(region.height / scale);
        const clampedX = clamp(x, 0, inputSize.width - 1);
        const clampedY = clamp(y, 0, inputSize.height - 1);
        const clampedWidth = clamp(width, 1, inputSize.width - clampedX);
        const clampedHeight = clamp(height, 1, inputSize.height - clampedY);

        console.log(`[DEBUG] region[${idx}] 转换:`, {
          screen: region,
          raw: { x, y, width, height },
          clamped: { x: clampedX, y: clampedY, width: clampedWidth, height: clampedHeight },
        });

        return {
          x: clampedX,
          y: clampedY,
          width: clampedWidth,
          height: clampedHeight,
          faceSourceId: region.faceSourceId,
        };
      })
      .filter((region: Region) => region.width > 1 && region.height > 1);

    console.log("[DEBUG] toImageRegions 最终结果:", imageRegions);
    return imageRegions;
  }, [regions, inputSize, selectionObjectFit, showSelection, selectionZoom]);

  useEffect(() => {
    if (!showSelection || !isImageInput || !inputSize) {
      return;
    }
    const input = kMirrorStates.input;
    if (!input || input.type !== "image") {
      return;
    }
    if (autoDetectedImagePathRef.current === input.path) {
      return;
    }

    autoDetectedImagePathRef.current = input.path;
    let cancelled = false;

    (async () => {
      setIsDetectingFaces(true);
      const detected = await Server.detectImageFaces(input.path);
      if (cancelled) {
        return;
      }
      setIsDetectingFaces(false);

      if (detected.error) {
        return;
      }

      const screenRegions = mapMediaRegionsToScreen(
        detected.regions || [],
        inputSize.width,
        inputSize.height
      ).map((region: Region) => ({
        x: region.x,
        y: region.y,
        width: region.width,
        height: region.height,
      }));

      setRegions(screenRegions);
      setSelectedRegionIndex(screenRegions.length > 0 ? 0 : null);
      if (!screenRegions.length) {
        setNotice(t("No face detected in selected areas."));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [showSelection, isImageInput, inputSize, mapMediaRegionsToScreen, t]);

  useEffect(() => {
    if (!isVideoInput || !showSelection) {
      return;
    }
    const video = previewVideoRef.current;
    if (!video) {
      return;
    }
    const targetTime = Math.max(0, videoKeyFrameMs / 1000);
    if (Math.abs(video.currentTime - targetTime) > 0.05) {
      try {
        video.currentTime = targetTime;
      } catch {
        // ignore
      }
    }
    video.pause();
  }, [isVideoInput, showSelection, videoKeyFrameMs, kMirrorStates.input?.path]);

  // const remapRegionsForZoom = useCallback(
  //   (currentRegions: Region[], oldZoom: number, newZoom: number) => {
  //     if (!previewRef.current) return currentRegions;
  //     const rect = previewRef.current.getBoundingClientRect();
  //     const cx = rect.width / 2;
  //     const cy = rect.height / 2;
  //     const factor = newZoom / oldZoom;

  //     return currentRegions.map((r) => {
  //       const x = cx + (r.x - cx) * factor;
  //       const y = cy + (r.y - cy) * factor;
  //       const width = r.width * factor;
  //       const height = r.height * factor;
  //       return {
  //         ...r,
  //         x,
  //         y,
  //         width,
  //         height,
  //       };
  //     });
  //   },
  //   []
  // );

  const handleWheelZoom = useCallback(
    (event: WheelEvent<HTMLDivElement>) => {
      if (!showSelection || !previewRef.current) return;
      event.stopPropagation();

      const rect = previewRef.current.getBoundingClientRect();
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;

      const delta = -event.deltaY;
      const step = 0.1;
      const nextZoom = clamp(
        selectionZoom + (delta > 0 ? step : -step),
        kMinSelectionZoom,
        kMaxSelectionZoom
      );

      if (nextZoom !== selectionZoom) {
        // 计算缩放比例变化
        const scaleFactor = nextZoom / selectionZoom;

        // 计算新的区域位置，保持鼠标位置不变
        // 公式推导：
        // (x - mouseX) * scaleFactor + mouseX = newX
        // newX = x * scaleFactor + mouseX * (1 - scaleFactor)

        // 由于 regions 存储的是相对于原始图片尺寸的坐标，我们需要先转换到屏幕坐标系进行计算，然后再转换回去
        // 但这里我们直接修改 regions 的逻辑是：regions 存储的是屏幕坐标系下的位置 (根据 mapMediaRegionsToScreen 和 toImageRegions 的逻辑推断)
        // 实际上，regions 存储的是相对于 previewRef 容器的坐标。

        // 修正：regions 存储的是相对于 previewRef 容器的坐标。
        // 当 zoom 发生变化时，previewRef 的内容（img/video）会缩放，但 regions 是绝对定位的 div。
        // 为了让 regions 跟随图片缩放，我们需要调整 regions 的坐标和大小。

        // 之前的逻辑是围绕中心点缩放：
        // const cx = rect.width / 2;
        // const cy = rect.height / 2;
        // x = cx + (r.x - cx) * factor;

        // 现在改为围绕鼠标点缩放：
        // x = mouseX + (r.x - mouseX) * factor;

        const nextRegions = regions.map((r: Region) => {
          const x = mouseX + (r.x - mouseX) * scaleFactor;
          const y = mouseY + (r.y - mouseY) * scaleFactor;
          const width = r.width * scaleFactor;
          const height = r.height * scaleFactor;
          return {
            ...r,
            x,
            y,
            width,
            height,
          };
        });

        setRegions(nextRegions);
        setSelectionZoom(nextZoom);
      }
    },
    [showSelection, selectionZoom, regions, clamp]
  );

  const handlePointerDown = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (!canSelect || !previewRef.current) {
        return;
      }
      const rect = previewRef.current.getBoundingClientRect();
      const startX = clamp(event.clientX - rect.left, 0, rect.width);
      const startY = clamp(event.clientY - rect.top, 0, rect.height);
      previewRef.current.setPointerCapture(event.pointerId);
      resizeRef.current = null;
      moveRef.current = null;
      setSelectedRegionIndex(null);
      selectingRef.current = true;
      startPointRef.current = { x: startX, y: startY };
      setDraftRegion({ x: startX, y: startY, width: 0, height: 0 });
    },
    [canSelect, clamp]
  );

  const handlePointerMove = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (!previewRef.current) {
        return;
      }
      const rect = previewRef.current.getBoundingClientRect();
      const currentX = clamp(event.clientX - rect.left, 0, rect.width);
      const currentY = clamp(event.clientY - rect.top, 0, rect.height);

      if (resizeRef.current) {
        const { index, handle, startX, startY, origin } = resizeRef.current;
        const dx = currentX - startX;
        const dy = currentY - startY;
        const minSize = 8;
        let { x, y, width, height } = origin;
        let newX = x;
        let newY = y;
        let newW = width;
        let newH = height;

        switch (handle) {
          case "nw":
            newX = clamp(x + dx, 0, x + width - minSize);
            newY = clamp(y + dy, 0, y + height - minSize);
            newW = width - (newX - x);
            newH = height - (newY - y);
            break;
          case "ne":
            newY = clamp(y + dy, 0, y + height - minSize);
            newW = clamp(width + dx, minSize, rect.width - x);
            newH = height - (newY - y);
            break;
          case "sw":
            newX = clamp(x + dx, 0, x + width - minSize);
            newW = width - (newX - x);
            newH = clamp(height + dy, minSize, rect.height - y);
            break;
          case "se":
          default:
            newW = clamp(width + dx, minSize, rect.width - x);
            newH = clamp(height + dy, minSize, rect.height - y);
            break;
        }

        setRegions((prev: Region[]) =>
          prev.map((region, idx) =>
            idx === index
              ? {
                ...region,
                x: newX,
                y: newY,
                width: newW,
                height: newH,
              }
              : region
          )
        );
        return;
      }

      if (moveRef.current) {
        const { index, startX, startY, origin } = moveRef.current;
        const dx = currentX - startX;
        const dy = currentY - startY;
        const newX = clamp(origin.x + dx, 0, rect.width - origin.width);
        const newY = clamp(origin.y + dy, 0, rect.height - origin.height);

        setRegions((prev: Region[]) =>
          prev.map((region, idx) =>
            idx === index
              ? {
                ...region,
                x: newX,
                y: newY,
              }
              : region
          )
        );
        return;
      }

      if (!selectingRef.current || !startPointRef.current) {
        return;
      }
      const start = startPointRef.current;
      const x = Math.min(start.x, currentX);
      const y = Math.min(start.y, currentY);
      const width = Math.abs(currentX - start.x);
      const height = Math.abs(currentY - start.y);
      setDraftRegion({ x, y, width, height });
    },
    [clamp]
  );

  const finishSelection = useCallback(
    (event?: PointerEvent<HTMLDivElement>) => {
      if (event && previewRef.current) {
        try {
          previewRef.current.releasePointerCapture(event.pointerId);
        } catch {
          // ignore: might not have pointer capture
        }
      }
      if (resizeRef.current) {
        resizeRef.current = null;
        return;
      }
      if (moveRef.current) {
        moveRef.current = null;
        return;
      }
      if (!selectingRef.current) {
        return;
      }
      selectingRef.current = false;
      if (draftRegion && draftRegion.width > 4 && draftRegion.height > 4) {
        setRegions((prev: Region[]) => [...prev, { ...draftRegion }]);
        setSelectedRegionIndex(regions.length);
      }
      setDraftRegion(null);
      startPointRef.current = null;
      moveRef.current = null;
    },
    [draftRegion, regions.length, isMultiFaceMode, faceSources]
  );

  const handleSelectRegion = useCallback(
    (index: number) => (event: PointerEvent<HTMLDivElement>) => {
      if (!canSelect || !previewRef.current) {
        return;
      }
      event.stopPropagation();

      // 如果当前有选中的素材，则直接分配给该区域
      if (activeFaceSourceId) {
        setRegions((prev: Region[]) =>
          prev.map((region: Region, idx: number) =>
            idx === index ? { ...region, faceSourceId: activeFaceSourceId } : region
          )
        );
        setSelectedRegionIndex(index);
        // 分配后不进入移动模式，方便连续分配
        return;
      }

      const rect = previewRef.current.getBoundingClientRect();
      const startX = clamp(event.clientX - rect.left, 0, rect.width);
      const startY = clamp(event.clientY - rect.top, 0, rect.height);
      previewRef.current.setPointerCapture(event.pointerId);
      selectingRef.current = false;
      startPointRef.current = null;
      setDraftRegion(null);
      resizeRef.current = null;
      moveRef.current = {
        index,
        startX,
        startY,
        origin: regions[index],
      };
      setSelectedRegionIndex(index);
    },
    [canSelect, clamp, regions, activeFaceSourceId]
  );

  const handleResizePointerDown = useCallback(
    (index: number, handle: ResizeHandle) =>
      (event: PointerEvent<HTMLDivElement>) => {
        if (!canSelect || !previewRef.current) {
          return;
        }
        event.stopPropagation();
        const rect = previewRef.current.getBoundingClientRect();
        const startX = clamp(event.clientX - rect.left, 0, rect.width);
        const startY = clamp(event.clientY - rect.top, 0, rect.height);
        previewRef.current.setPointerCapture(event.pointerId);
        moveRef.current = null;
        resizeRef.current = {
          index,
          handle,
          startX,
          startY,
          origin: regions[index],
        };
        setSelectedRegionIndex(index);
      },
    [canSelect, clamp, regions]
  );

  const handleDeleteSelected = useCallback(() => {
    if (selectedRegionIndex === null) {
      return;
    }
    setRegions((prev: Region[]) =>
      prev.filter((_, index) => index !== selectedRegionIndex)
    );
    setSelectedRegionIndex(null);
  }, [selectedRegionIndex]);

  const handleClearRegions = useCallback(() => {
    setRegions([]);
    setDraftRegion(null);
    setNotice(null);
    setSelectedRegionIndex(null);
    setActiveFaceSourceId(null);
    selectingRef.current = false;
    startPointRef.current = null;
    resizeRef.current = null;
    moveRef.current = null;
  }, []);

  const handleEditRegions = useCallback(() => {
    if (!kMirrorStates.input || kMirrorStates.input.type !== "image") {
      return;
    }
    kMirrorStates.result = undefined;
    rebuild.current();
    setSelectedRegionIndex(null);
    setIsEditingRegions(true);
  }, []);

  const handleToggleMultiFaceMode = useCallback(() => {
    setIsMultiFaceMode((prev: boolean) => {
      const next = !prev;
      if (!next) {
        setFaceSources([]);
        setRegions((prevRegions: Region[]) =>
          prevRegions.map((region: Region) => ({
            x: region.x,
            y: region.y,
            width: region.width,
            height: region.height,
          }))
        );
      } else {
        const me = kMirrorStates.me;
        if (!me) {
          return next;
        }
        setFaceSources((prevSources: FaceAsset[]) => {
          const others = prevSources.filter(
            (item) => item.id !== kDefaultFaceSourceId
          );
          return [
            {
              id: kDefaultFaceSourceId,
              path: me.path,
              src: me.src,
              locked: true,
            },
            ...others,
          ];
        });
      }
      return next;
    });
    setNotice(null);
    setActiveFaceSourceId(null);
  }, []);

  const addFaceSourcesFromPaths = useCallback((paths: string[]) => {
    if (!paths.length) {
      return;
    }
    const additions = paths
      .filter((path: string) => {
        const ext = getFileExtension(path);
        return isImageFile(path) && ext !== ".heic" && ext !== ".heif";
      })
      .map((path: string, idx: number) => ({
        id: `face-${Date.now()}-${idx}-${Math.random().toString(36).slice(2, 8)}`,
        path,
        src: convertFileSrc(path),
      }));

    if (!additions.length) {
      return;
    }

    setFaceSources((prev: FaceAsset[]) => {
      const existed = new Set(prev.map((item) => item.path));
      const deduped = additions.filter((item) => !existed.has(item.path));
      if (!deduped.length) {
        return prev;
      }
      return [...prev, ...deduped];
    });
    setNotice(null);
  }, []);

  const handleOpenFaceSourcePicker = useCallback(async () => {
    try {
      const selected = await openDialog({
        multiple: true,
        directory: false,
        filters: [
          {
            name: "Images",
            extensions: ["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"],
          },
        ],
      });

      if (!selected) {
        return;
      }

      const paths = (Array.isArray(selected) ? selected : [selected]).filter(
        (path): path is string => typeof path === "string"
      );

      if (paths.length > 0) {
        addFaceSourcesFromPaths(paths);
      }
    } catch {
      // fallback: keep old file input way in web/dev environment
      faceSourceInputRef.current?.click();
    }
  }, [addFaceSourcesFromPaths]);

  const handleFaceSourceInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      // 修复：event.target.files 可能是 null，且类型转换可能存在问题
      // 另外，input type="file" 的 onChange 事件触发后，如果再次选择相同文件可能不会触发
      // 所以需要在处理完后清空 value

      const files = event.target.files;
      if (!files || files.length === 0) {
        return;
      }

      const fileList = Array.from(files);
      // 注意：浏览器出于安全考虑，通常不直接暴露完整文件路径 (file.path)
      // 但在 Tauri 环境中，如果配置了适当的权限，File 对象可能会包含 path 属性
      // 或者我们需要使用 Tauri 的 dialog API 来选择文件，而不是原生的 input

      // 尝试读取 path，如果不存在（比如在纯浏览器环境），则回退到使用 URL.createObjectURL (虽然这只对本次会话有效)
      // 但根据项目上下文，这是一个 Tauri 应用，且 Asset 接口定义了 path: string

      const paths = fileList
        .map((file) => (file as any).path) // 强制转换，Tauri 环境下 File 对象通常有 path 属性
        .filter((path): path is string => !!path);

      if (paths.length > 0) {
        addFaceSourcesFromPaths(paths);
      } else {
        // 如果获取不到 path (例如在非 Tauri 环境或权限问题)，尝试使用 createObjectURL
        // 但这里为了保持一致性，我们假设是在 Tauri 环境下运行
        // 如果 paths 为空，可能是因为 input 没能获取到 path
        console.warn("无法获取文件路径，请确保在 Tauri 环境中运行");
      }

      event.target.value = "";
    },
    [addFaceSourcesFromPaths]
  );

  const handleAssignSelectedRegionFaceSource = useCallback(
    (faceSourceId: string) => {
      if (selectedRegionIndex === null) {
        return;
      }
      setRegions((prev: Region[]) =>
        prev.map((region: Region, idx: number) =>
          idx === selectedRegionIndex ? { ...region, faceSourceId } : region
        )
      );
      setActiveFaceSourceId(faceSourceId || null);
    },
    [selectedRegionIndex]
  );

  const handleRenameFaceSource = useCallback(
    (faceSourceId: string) => {
      const source = faceSources.find((s: FaceAsset) => s.id === faceSourceId);
      if (!source || source.locked) return;

      const currentName = source.name || `Face Source ${faceSources.indexOf(source) + 1}`;
      const newName = window.prompt(t("Enter new name"), currentName);

      if (newName && newName.trim() !== "") {
        setFaceSources((prev: FaceAsset[]) =>
          prev.map((s: FaceAsset) =>
            s.id === faceSourceId ? { ...s, name: newName.trim() } : s
          )
        );
      }
    },
    [faceSources, t]
  );

  const handleRemoveFaceSource = useCallback(
    (faceSourceId: string) => {
      if (faceSourceId === kDefaultFaceSourceId) {
        return;
      }
      setFaceSources((prev: FaceAsset[]) =>
        prev.filter((source: FaceAsset) => source.id !== faceSourceId || source.locked)
      );
      setRegions((prev: Region[]) =>
        prev.map((region: Region) =>
          region.faceSourceId === faceSourceId
            ? { ...region, faceSourceId: undefined }
            : region
        )
      );
      setActiveFaceSourceId((prev: string | null) =>
        prev === faceSourceId ? null : prev
      );
    },
    []
  );

  const handleVideoTimelineChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const next = Number(event.target.value);
      setVideoKeyFrameMs(
        Number.isFinite(next) ? Math.max(0, Math.round(next)) : 0
      );
    },
    []
  );

  const handleDetectVideoFacesAtKeyFrame = useCallback(async () => {
    const input = kMirrorStates.input;
    if (!input || input.type !== "video" || !inputSize) {
      return;
    }

    setNotice(null);
    setSelectionZoom(1);
    setIsDetectingFaces(true);
    const detected = await Server.detectVideoFaces(
      input.path,
      Math.max(0, Math.round(videoKeyFrameMs))
    );
    setIsDetectingFaces(false);

    if (detected.error) {
      setNotice(t("Failed to detect faces at key frame."));
      return;
    }

    const frameWidth = detected.frameWidth || inputSize.width;
    const frameHeight = detected.frameHeight || inputSize.height;
    const screenRegions = mapMediaRegionsToScreen(
      detected.regions || [],
      frameWidth,
      frameHeight
    ).map((region: Region) => ({
      x: region.x,
      y: region.y,
      width: region.width,
      height: region.height,
    }));

    setRegions(screenRegions);
    setSelectedRegionIndex(screenRegions.length ? 0 : null);
    if (!screenRegions.length) {
      setNotice(t("No face detected in selected areas."));
    }
  }, [inputSize, mapMediaRegionsToScreen, t, videoKeyFrameMs]);

  const handleStartSwap = useCallback(async () => {
    const me = kMirrorStates.me;
    const input = kMirrorStates.input;
    if (!me || !input) {
      return;
    }

    const beginSwap = () => {
      setNotice(null);
      kMirrorStates.result = undefined;
      rebuild.current();
      setIsEditingRegions(false);
      setSelectedRegionIndex(null);
      setDraftRegion(null);
      selectingRef.current = false;
      startPointRef.current = null;
      resizeRef.current = null;
      moveRef.current = null;
    };

    if (input.type === "video") {
      // 询问用户是否使用 GPU 加速
      const useGpu = window.confirm(
        t("Enable GPU acceleration for faster processing?") + "\n\n" +
        t("GPU acceleration can speed up video processing by 10-50x, but requires compatible graphics drivers.") + "\n\n" +
        t("Click OK to enable GPU, or Cancel to use CPU only.")
      );

      if (isMultiFaceMode) {
        if (!regions.length) {
          setNotice(t("Please select at least one area."));
          return;
        }

        const videoRegions = toImageRegions();
        if (!videoRegions.length) {
          setNotice(t("Please select at least one area."));
          return;
        }

        if (!faceSources.length) {
          setNotice(t("Please add at least one face source."));
          return;
        }

        if (videoRegions.some((region: Region) => !region.faceSourceId)) {
          setNotice(t("Please assign a face source to each selected area."));
          return;
        }

        beginSwap();

        const result = await swapVideo({
          inputVideo: input.path,
          regions: videoRegions,
          faceSources: faceSources.map((item: FaceAsset) => ({
            id: item.id,
            path: item.path,
          })),
          keyFrameMs: Math.max(0, Math.round(videoKeyFrameMs)),
          useGpu: useGpu,
        });

        setSuccess(result != null);
        if (result) {
          kMirrorStates.result = {
            src: `${convertFileSrc(result)}?t=${Date.now()}`,
            path: result,
            type: "video",
          };
        } else {
          setIsEditingRegions(true);
        }
        rebuild.current();
        return;
      }

      beginSwap();

      const result = await swapVideo({
        inputVideo: input.path,
        targetFace: me.path,
        useGpu: useGpu,
      });

      setSuccess(result != null);
      if (result) {
        kMirrorStates.result = {
          src: `${convertFileSrc(result)}?t=${Date.now()}`,
          path: result,
          type: "video",
        };
      } else {
        setIsEditingRegions(true);
      }
      rebuild.current();
      return;
    }

    if (input.type !== "image") {
      return;
    }

    if (!regions.length) {
      setNotice(t("Please select at least one area."));
      return;
    }

    const imageRegions = toImageRegions();
    if (!imageRegions.length) {
      setNotice(t("Please select at least one area."));
      return;
    }

    if (isMultiFaceMode && !faceSources.length) {
      setNotice(t("Please add at least one face source."));
      return;
    }

    if (
      isMultiFaceMode &&
      imageRegions.some((region: Region) => !region.faceSourceId)
    ) {
      setNotice(t("Please assign a face source to each selected area."));
      return;
    }

    beginSwap();

    const result = await swapFace(
      isMultiFaceMode
        ? {
          inputImage: input.path,
          regions: imageRegions,
          faceSources: faceSources.map((item: FaceAsset) => ({
            id: item.id,
            path: item.path,
          })),
        }
        : {
          inputImage: input.path,
          targetFace: me.path,
          regions: imageRegions,
        }
    );

    setSuccess(result != null);
    if (result) {
      kMirrorStates.result = {
        src: `${convertFileSrc(result)}?t=${Date.now()}`,
        path: result,
        type: "image",
      };
    } else {
      setIsEditingRegions(true);
    }
    rebuild.current();
  }, [
    faceSources,
    isMultiFaceMode,
    regions,
    swapFace,
    swapVideo,
    t,
    toImageRegions,
    videoKeyFrameMs,
  ]);

  const { ref, isOverTarget } = useDragDrop(async (paths: string[]) => {
    if (!paths.length) {
      return;
    }

    const shouldAddFaceSources =
      !kMirrorStates.isMe &&
      isEditingRegions &&
      isMultiFaceMode &&
      !!kMirrorStates.input &&
      (kMirrorStates.input.type === "image" ||
        kMirrorStates.input.type === "video");

    // 修复：支持拖拽到素材池区域
    // 检查是否拖拽到了素材池区域
    // const isOverFaceSourcePool = (event?: DragEvent) => {
    //   // 由于 useDragDrop 钩子目前只提供了 paths，没有提供 event 对象或坐标
    //   // 我们需要修改 useDragDrop 或者在这里做一些假设
    //   // 但根据用户描述 "另外把图片从外部拖动到素材池也可以！"
    //   // 我们可以简单地认为，只要处于多脸模式且正在编辑区域，拖拽进来的图片就应该被视为添加素材
    //   // 除非用户明确是想替换主输入图片（但这通常需要拖拽到特定区域，或者在非编辑模式下）

    //   // 现有的逻辑是：如果处于多脸模式编辑状态，拖拽就认为是添加素材。
    //   // 这似乎符合逻辑。
    //   // 但用户反馈说 "把图片从外部拖动到素材池也可以"，暗示可能之前的逻辑有问题或者用户期望更明确的交互。

    //   // 让我们检查一下 shouldAddFaceSources 的条件：
    //   // !kMirrorStates.isMe (不是在设置自己的脸)
    //   // isEditingRegions (正在编辑区域)
    //   // isMultiFaceMode (多脸模式)
    //   // !!kMirrorStates.input (有输入图片/视频)

    //   // 这个逻辑看起来是正确的，只要满足这些条件，拖拽就会调用 addFaceSourcesFromPaths。
    //   // 可能是用户在操作时，某些条件不满足？
    //   // 或者用户希望即使不满足某些条件（比如不在编辑区域？），只要拖拽到素材池那个 UI 区域，也能添加？

    //   // 由于我们无法获取拖拽的具体坐标（useDragDrop 限制），我们只能依赖全局状态。
    //   // 现有的逻辑已经覆盖了 "在多脸模式下拖拽添加素材" 的需求。
    //   // 也许问题在于 useDragDrop 的实现，或者 isOverTarget 的判断。

    //   return true;
    // };

    if (shouldAddFaceSources) {
      addFaceSourcesFromPaths(paths);
      return;
    }

    const path = paths[0];
    const src = convertFileSrc(path);
    const isVideo = isVideoFile(path);
    const isImage = isImageFile(path);
    const ext = getFileExtension(path);
    const isHeic = ext === ".heic" || ext === ".heif";

    if (kMirrorStates.isMe) {
      if (!isImage) {
        setNotice(
          isHeic
            ? t("HEIC/HEIF is not supported. Please convert to JPG/PNG.")
            : t("Please use an image for your face photo.")
        );
        return;
      }
      kMirrorStates.me = {
        src,
        path,
        type: "image",
      };
      setNotice(null);
      rebuild.current();
    } else {
      if (!isImage && !isVideo) {
        setNotice(
          isHeic
            ? t("HEIC/HEIF is not supported. Please convert to JPG/PNG.")
            : t("Unsupported file type.")
        );
        return;
      }
      kMirrorStates.input = {
        src,
        path,
        type: isVideo ? "video" : "image",
      };
      setNotice(null);
      rebuild.current();
    }

    if (kMirrorStates.me && kMirrorStates.input) {
      kMirrorStates.result = undefined;
      rebuild.current();
      setIsEditingRegions(true);
    }
  });

  const isReady = kMirrorStates.me && kMirrorStates.input;
  const hasRegions = regions.length > 0;
  const selectionTips =
    isReady && !isSwapping && !kMirrorStates.isMe && isEditingRegions
      ? hasRegions
        ? t("Click Start to swap selected areas.")
        : isVideoInput
          ? t("Pick a key frame, then detect faces or draw boxes.")
          : t("Draw boxes to select areas.")
      : null;

  // Map error codes to user-friendly messages
  const getSwapErrorMessage = (error: string | null): string | null => {
    if (!error) return null;
    const errorMap: Record<string, string> = {
      "video-not-supported": t("Video face swap not supported. Please update server."),
      "unsupported-image-format": t("Unsupported image format. Please use JPG/PNG/WebP/BMP/TIFF."),
      "unsupported-video-format": t("Unsupported video format. Please use MP4/MOV/AVI/MKV/WEBM/M4V."),
      "image-decode-failed": t("Failed to read image file. Please try converting to JPG/PNG."),
      "file-not-found": t("File not found. Please reselect the file."),
      "no-face-detected": t("No face detected in the photo."),
      "no-face-in-selected-regions": t("No face detected in selected areas."),
      "output-write-failed": t("Failed to save output file."),
      "video-open-failed": t("Failed to open video file."),
      "video-write-failed": t("Failed to write output video file."),
      "video-output-missing": t("Video swap failed. Output file missing."),
      "missing-face-sources": t("Please add at least one face source."),
      "invalid-face-source-binding": t(
        "Please assign a face source to each selected area."
      ),
      "face-source-not-found": t(
        "Please assign a face source to each selected area."
      ),
    };
    return errorMap[error] || null;
  };

  const swapErrorMessage = getSwapErrorMessage(swapError);

  const formatEta = useCallback((seconds: number | null | undefined) => {
    if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
      return "--:--";
    }
    const safeSeconds = Math.max(0, Math.floor(seconds));
    const mm = Math.floor(safeSeconds / 60)
      .toString()
      .padStart(2, "0");
    const ss = (safeSeconds % 60).toString().padStart(2, "0");
    return `${mm}:${ss}`;
  }, []);

  const getVideoStageLabel = useCallback(
    (stage: string | null | undefined) => {
      if (!stage) {
        return t("Initializing...");
      }
      const stageMap: Record<string, string> = {
        queued: t("Queued"),
        "validating-input": t("Validating input"),
        "opening-video": t("Opening video"),
        "reading-video-metadata": t("Reading video metadata"),
        "extracting-target-face": t("Extracting target face"),
        "building-face-tracks": t("Building face tracks"),
        "processing-video-frames": t("Processing video frames"),
        "muxing-audio": t("Muxing audio"),
        finalizing: t("Finalizing output"),
        done: t("Completed"),
        failed: t("Failed"),
        cancelled: t("Cancelled"),
      };
      return stageMap[stage] || stage;
    },
    [t]
  );

  const tips = notice
    ? notice
    : kMirrorStates.isMe
      ? t("First, drag your front-facing photo into the mirror.")
      : !isReady
        ? t("Then, drag the photo you want to swap faces with into the mirror.")
        : isSwapping
          ? isVideoInput
            ? t("Video swapping stage", {
              stage: getVideoStageLabel(videoStage),
            })
            : t("Face swapping... This may take a few seconds, please wait.")
          : isDetectingFaces
            ? t("Detecting faces...")
            : selectionTips
              ? selectionTips
              : swapErrorMessage
                ? swapErrorMessage
                : success
                  ? isVideoInput
                    ? t("Face swap successful! Video saved locally.")
                    : t("Face swap successful! Image saved locally.")
                  : isVideoInput
                    ? t("Video face swap failed. Try a different video.")
                    : t("Face swap failed. Try a different image.");

  const previewSrc = kMirrorStates.isMe
    ? kMirrorStates.me?.src || background
    : isImageInput && isEditingRegions
      ? kMirrorStates.input?.src || background
      : kMirrorStates.result?.src || kMirrorStates.input?.src || background;

  const previewType = kMirrorStates.isMe
    ? "image"
    : isImageInput
      ? "image"
      : kMirrorStates.result?.type || kMirrorStates.input?.type || "image";

  return (
    <div data-tauri-drag-region className="w-100vw h-100vh p-40px">
      <input
        ref={faceSourceInputRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={handleFaceSourceInputChange}
      />
      <div ref={ref} className="relative w-full h-full">
        <div className="absolute top-[-40px] w-full flex-c-c c-white z-10">
          <p className="bg-black p-[4px_8px]">{tips}</p>
        </div>
        <div className="absolute top-50px right-50px z-10">
          <div className="relative dropdown">
            <img
              src={iconMenu}
              className="h-70px cursor-pointer pb-10px"
              draggable={false}
            />
            <div>
              <div className="dropdown-menu flex-col-c-c bg-black color-white">
                <div
                  onClick={() => {
                    i18n.changeLanguage(i18n.language === "en" ? "zh" : "en");
                  }}
                >
                  {t("Language")}
                </div>
                {kMirrorStates.me && (
                  <div
                    onClick={() => {
                      kMirrorStates.isMe = !kMirrorStates.isMe;
                      rebuild.current();
                    }}
                  >
                    {t("Switch")}
                  </div>
                )}
                {!kMirrorStates.isMe && (isImageInput || isVideoInput) && (
                  <div onClick={handleToggleMultiFaceMode}>
                    {isMultiFaceMode ? t("Single-Face Mode") : t("Multi-Face Swap")}
                  </div>
                )}
                <div onClick={() => openExternal(t("aboutLink"))}>{t("About")}</div>
                <div onClick={() => exit(0)}>{t("Quit")}</div>
              </div>
            </div>
          </div>
        </div>
        {!showSelection && (
          <img
            src={kMirrorStates.isMe ? mirrorMe : mirrorInput}
            className="mirror-frame absolute w-full h-full object-cover z-3"
            draggable={false}
            style={{
              maskImage:
                "radial-gradient(circle, rgba(0, 0, 0, 0) 30%, rgba(0, 0, 0, 1) 40%)",
              WebkitMaskImage:
                "radial-gradient(circle, rgba(0, 0, 0, 0) 30%, rgba(0, 0, 0, 1) 40%)",
            }}
          />
        )}
        <div
          className="w-full h-full flex-c-c"
          style={{
            padding: showSelection
              ? selectionPadding
              : kMirrorStates.isMe
                ? "120px"
                : "100px",
          }}
        >
          <div className={`mirror-preview ${isOverTarget ? "drop-over" : ""}`}>
            <div className={`mirror-clip ${showSelection ? "rect-edit-mode" : ""}`}>
              <div
                ref={previewRef}
                className={`preview-container ${showSelection ? "selection-active" : ""}`}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={finishSelection}
                onPointerCancel={finishSelection}
                onWheel={handleWheelZoom}
              >
                {previewType === "video" ? (
                  <video
                    ref={previewVideoRef}
                    src={previewSrc}
                    className={`preview-media ${showSelection ? "rect-edit-mode zoomable" : ""}`}
                    style={{
                      objectFit: selectionObjectFit,
                      transform: showSelection ? `scale(${selectionZoom})` : undefined,
                    }}
                    autoPlay={!showSelection}
                    loop={!showSelection}
                    muted
                    playsInline
                    onLoadedMetadata={(event) => {
                      const media = event.currentTarget;
                      if (media.videoWidth > 0 && media.videoHeight > 0) {
                        setInputSize({
                          width: media.videoWidth,
                          height: media.videoHeight,
                        });
                      }
                      const durationSeconds = Number.isFinite(media.duration)
                        ? Math.max(0, media.duration)
                        : 0;
                      const durationMs = Math.round(durationSeconds * 1000);
                      setVideoDurationMs(durationMs);
                      setVideoKeyFrameMs((prev: number) =>
                        Math.min(prev, durationMs)
                      );
                      if (showSelection) {
                        media.pause();
                      }
                    }}
                  />
                ) : (
                  <img
                    src={previewSrc}
                    className={`preview-media ${showSelection ? "rect-edit-mode zoomable" : ""}`}
                    style={{
                      objectFit: selectionObjectFit,
                      transform: showSelection ? `scale(${selectionZoom})` : undefined,
                    }}
                    draggable={false}
                  />
                )}
                {showSelection && (
                  <div className="selection-layer">
                    {regions.map((region: Region, index: number) => (
                      <div
                        key={`region-${index}`}
                        className={`selection-rect ${region.faceSourceId ? "assigned" : ""} ${selectedRegionIndex === index ? "selected" : ""}`}
                        style={{
                          left: region.x,
                          top: region.y,
                          width: region.width,
                          height: region.height,
                        }}
                        onPointerDown={handleSelectRegion(index)}
                      >
                        <span className="selection-index">{index + 1}</span>
                        {selectedRegionIndex === index && (
                          <>
                            <div
                              className="selection-handle nw"
                              onPointerDown={handleResizePointerDown(index, "nw")}
                            />
                            <div
                              className="selection-handle ne"
                              onPointerDown={handleResizePointerDown(index, "ne")}
                            />
                            <div
                              className="selection-handle sw"
                              onPointerDown={handleResizePointerDown(index, "sw")}
                            />
                            <div
                              className="selection-handle se"
                              onPointerDown={handleResizePointerDown(index, "se")}
                            />
                          </>
                        )}
                      </div>
                    ))}
                    {draftRegion && (
                      <div
                        className="selection-rect draft"
                        style={{
                          left: draftRegion.x,
                          top: draftRegion.y,
                          width: draftRegion.width,
                          height: draftRegion.height,
                        }}
                      />
                    )}
                  </div>
                )}
              </div>
            </div>
            {showSelection && isVideoInput && (
              <div
                className="video-timeline-panel"
                onPointerDown={(
                  event: PointerEvent<HTMLDivElement>
                ) => event.stopPropagation()}
              >
                <div className="video-timeline-row">
                  <span className="video-timeline-label">{t("Key Frame")}</span>
                  <input
                    className="video-timeline-slider"
                    type="range"
                    min={0}
                    max={Math.max(videoDurationMs, 0)}
                    step={40}
                    value={Math.min(videoKeyFrameMs, Math.max(videoDurationMs, 0))}
                    onChange={handleVideoTimelineChange}
                  />
                  <span className="video-timeline-value">
                    {(Math.max(videoKeyFrameMs, 0) / 1000).toFixed(2)}s
                  </span>
                  <div
                    className={`selection-btn ${isDetectingFaces || !inputSize ? "disabled" : ""}`}
                    onClick={handleDetectVideoFacesAtKeyFrame}
                  >
                    {isDetectingFaces
                      ? t("Detecting faces...")
                      : t("Detect Faces")}
                  </div>
                </div>
              </div>
            )}
            {showToolbar && (
              <div
                className="selection-toolbar"
                onPointerDown={(
                  event: PointerEvent<HTMLDivElement>
                ) => event.stopPropagation()}
              >
                <div
                  className={`selection-btn ${canStartSwap ? "" : "disabled"}`}
                  onClick={handleStartSwap}
                >
                  {t("Start Swap")}
                </div>
                <div
                  className={`selection-btn ${selectedRegionIndex === null ? "disabled" : ""}`}
                  onClick={handleDeleteSelected}
                >
                  {t("Delete Selected")}
                </div>
                <div
                  className={`selection-btn ${regions.length ? "" : "disabled"}`}
                  onClick={handleClearRegions}
                >
                  {t("Clear Selection")}
                </div>
              </div>
            )}
            {!showSelection && isImageInput && kMirrorStates.result && (
              <div
                className="selection-toolbar"
                onPointerDown={(
                  event: PointerEvent<HTMLDivElement>
                ) => event.stopPropagation()}
              >
                <div className="selection-btn" onClick={handleEditRegions}>
                  {t("Edit Selection")}
                </div>
              </div>
            )}
            {showSelection && isMultiFaceMode && (
              <div
                className={`face-source-panel ${isVideoInput ? "video-mode" : ""}`}
                onPointerDown={(
                  event: PointerEvent<HTMLDivElement>
                ) => event.stopPropagation()}
              >
                <div className="face-source-header">
                  <span>{t("Face Source Pool")}</span>
                  <div className="selection-btn" onClick={handleOpenFaceSourcePicker}>
                    {t("Add Face Sources")}
                  </div>
                </div>
                {selectedRegionIndex !== null && (
                  <div className="face-source-bind-row">
                    <span>
                      {t("Selected area")} #{selectedRegionIndex + 1}
                    </span>
                    <select
                      className="face-source-select"
                      value={regions[selectedRegionIndex]?.faceSourceId || ""}
                      onChange={(event) =>
                        handleAssignSelectedRegionFaceSource(event.target.value)
                      }
                    >
                      <option value="">{t("Select a face source")}</option>
                      {faceSources.map((source: FaceAsset, index: number) => (
                        <option key={source.id} value={source.id}>
                          {source.locked
                            ? t("Default Face")
                            : source.name || `${t("Face Source")} ${index + 1}`}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
                <div className="face-source-list">
                  {faceSources.map((source: FaceAsset, index: number) => {
                    const selectedFaceSourceId =
                      selectedRegionIndex === null
                        ? ""
                        : regions[selectedRegionIndex]?.faceSourceId || "";
                    const isAssignedToSelectedRegion = selectedFaceSourceId === source.id;
                    const isSelected =
                      activeFaceSourceId === source.id || isAssignedToSelectedRegion;
                    return (
                      <div
                        key={source.id}
                        className={`face-source-card ${isSelected ? "selected" : ""}`}
                        onClick={() =>
                          setActiveFaceSourceId((prev: string | null) =>
                            prev === source.id ? null : source.id
                          )
                        }
                        onContextMenu={(event) => {
                          if (source.locked) {
                            return;
                          }
                          event.preventDefault();
                          handleRenameFaceSource(source.id);
                        }}
                      >
                        <img
                          src={source.src}
                          className="face-source-thumb"
                          draggable={false}
                        />
                        <div className="face-source-meta">
                          <span className="face-source-title">
                            {source.locked
                              ? t("Default Face")
                              : source.name || `${t("Face Source")} ${index + 1}`}
                          </span>
                          {!source.locked && (
                            <div className="flex gap-2">
                              <button
                                type="button"
                                className="face-source-remove"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  handleRemoveFaceSource(source.id);
                                }}
                              >
                                {t("Remove")}
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
                {!faceSources.length && (
                  <div className="face-source-empty">
                    {t("No face sources yet. Add images to begin.")}
                  </div>
                )}
              </div>
            )}
            {isVideoInput && isSwapping && (
              <div className="video-progress-panel">
                <ProgressBar progress={videoProgress} width="360px" height="6px" />
                <div className="video-progress-meta">
                  <span>
                    {t("Video processing progress", {
                      progress: videoProgress.toFixed(1),
                    })}
                  </span>
                  <span>
                    {videoEtaSeconds !== null
                      ? t("Estimated remaining time", {
                        eta: formatEta(videoEtaSeconds),
                      })
                      : t("Estimating remaining time...")}
                  </span>
                  <span>
                    {t("Current stage", {
                      stage: getVideoStageLabel(videoStage),
                    })}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
