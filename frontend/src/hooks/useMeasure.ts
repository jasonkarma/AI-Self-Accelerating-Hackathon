import { useLayoutEffect, useRef, useState } from "react";

/**
 * Container width/height via ResizeObserver — charts stay crisp at any size.
 *
 * The first measurement is synchronous (`getBoundingClientRect` in a layout
 * effect) rather than waiting for the observer's first callback. Without it every
 * chart paints one frame at its minimum width and then jumps, which reads as a
 * layout bug even though the final frame is correct.
 */
export function useMeasure<T extends HTMLElement>(): [
  React.RefObject<T>,
  { width: number; height: number },
] {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useLayoutEffect(() => {
    const node = ref.current;
    if (!node) return;

    const apply = (width: number, height: number) => {
      setSize((previous) =>
        Math.abs(previous.width - width) < 0.5 && Math.abs(previous.height - height) < 0.5
          ? previous
          : { width, height },
      );
    };

    const rect = node.getBoundingClientRect();
    apply(rect.width, rect.height);

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      apply(entry.contentRect.width, entry.contentRect.height);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return [ref, size];
}
