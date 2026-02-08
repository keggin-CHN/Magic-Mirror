import { DragDropEvent, getCurrentWebview } from "@tauri-apps/api/webview";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useCallback, useEffect, useRef, useState } from "react";

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

export function useDragDrop(onDrop: (paths: string[]) => void) {
  const ref = useRef<HTMLDivElement | null>(null);
  const onDropRef = useRef(onDrop);
  const [isOverTarget, setIsOverTarget] = useState(false);
  const debouncedDropRef = useRef<((paths: string[]) => void) & { cancel?: () => void }>();

  useEffect(() => {
    onDropRef.current = onDrop;
  }, [onDrop]);

  useEffect(() => {
    // 创建 debounced 函数
    debouncedDropRef.current = debounce((paths: string[]) => {
      onDropRef.current(paths);
    }, 100);

    // 清理函数
    return () => {
      debouncedDropRef.current?.cancel?.();
    };
  }, []);

  const onDropped = useCallback((paths: string[]) => {
    debouncedDropRef.current?.(paths);
  }, []);

  useEffect(() => {
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

  return { isOverTarget, ref };
}
