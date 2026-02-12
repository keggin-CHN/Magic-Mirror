import { DragDropEvent, getCurrentWebview } from "@tauri-apps/api/webview";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useCallback, useEffect, useRef, useState } from "react";
import { isTauri } from "@/services/runtime";

// 类型安全的 debounce 实现
function debounce<T extends (...args: any[]) => any>(
  func: T,
  delay = 100
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  const debouncedFn = (...args: Parameters<T>) => {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      func(...args);
      timeoutId = null;
    }, delay);
  };

  // 添加清理方法
  (debouncedFn as any).cancel = () => {
    if (timeoutId !== null) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }
  };

  return debouncedFn;
}

export function useDragDrop(
  onDrop: (payload: { paths: string[]; files: File[] }) => void
) {
  const ref = useRef<HTMLDivElement | null>(null);
  const onDropRef = useRef(onDrop);
  const [isOverTarget, setIsOverTarget] = useState(false);
  const debouncedDropRef = useRef<
    ((payload: { paths: string[]; files: File[] }) => void) & {
      cancel?: () => void;
    }
  >();

  useEffect(() => {
    onDropRef.current = onDrop;
  }, [onDrop]);

  useEffect(() => {
    // 创建 debounced 函数
    debouncedDropRef.current = debounce(
      (payload: { paths: string[]; files: File[] }) => {
        onDropRef.current(payload);
      },
      100
    );

    // 清理函数
    return () => {
      debouncedDropRef.current?.cancel?.();
    };
  }, []);

  const onDropped = useCallback((paths: string[], files: File[] = []) => {
    debouncedDropRef.current?.({ paths, files });
  }, []);

  useEffect(() => {
    if (!isTauri()) {
      return;
    }
    const checkIsInside = async (event: DragDropEvent) => {
      const targetRect = ref.current?.getBoundingClientRect();
      if (!targetRect || event.type === "leave") {
        return false;
      }
      const factor = await getCurrentWindow().scaleFactor();
      const position = event.position.toLogical(factor);
      const isInside =
        position.x >= targetRect.left &&
        position.x <= targetRect.right &&
        position.y >= targetRect.top &&
        position.y <= targetRect.bottom;
      return isInside;
    };

    const setupListener = async () => {
      const unlisten = await getCurrentWebview().onDragDropEvent(
        async (event: { payload: DragDropEvent }) => {
          const isInside = await checkIsInside(event.payload);
          if (event.payload.type === "over") {
            setIsOverTarget(isInside);
            return;
          }
          if (event.payload.type === "drop" && isInside) {
            onDropped(event.payload.paths);
          }
          setIsOverTarget(false);
        }
      );

      return unlisten;
    };

    let cleanup: (() => void) | undefined;

    setupListener().then((unlisten) => {
      cleanup = unlisten;
    });

    return () => {
      cleanup?.();
    };
  }, [onDropped]);

  useEffect(() => {
    if (isTauri()) {
      return;
    }
    const target = ref.current;
    if (!target) {
      return;
    }
    const handleDragOver = (event: DragEvent) => {
      event.preventDefault();
      setIsOverTarget(true);
    };
    const handleDragLeave = () => {
      setIsOverTarget(false);
    };
    const handleDrop = (event: DragEvent) => {
      event.preventDefault();
      setIsOverTarget(false);
      const files = Array.from(event.dataTransfer?.files || []);
      if (files.length) {
        onDropped([], files);
      }
    };
    target.addEventListener("dragover", handleDragOver);
    target.addEventListener("dragleave", handleDragLeave);
    target.addEventListener("drop", handleDrop);
    return () => {
      target.removeEventListener("dragover", handleDragOver);
      target.removeEventListener("dragleave", handleDragLeave);
      target.removeEventListener("drop", handleDrop);
    };
  }, [onDropped]);

  return { isOverTarget, ref };
}
