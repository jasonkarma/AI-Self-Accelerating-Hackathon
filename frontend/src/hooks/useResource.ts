import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "../lib/api";

export type Resource<T> =
  | { status: "loading"; data: T | null; error: null }
  | { status: "ready"; data: T; error: null }
  | { status: "error"; data: T | null; error: ApiError | Error };

/**
 * One-shot fetch with a manual `reload`.
 *
 * The console polls nothing on a timer: every panel is a *report* of a frozen
 * artifact except the live state, and a report that changes under the reader is
 * worse than one that is 30 s stale. `reload` is the only way data moves, so a
 * screenshot and the screen always agree.
 */
export function useResource<T>(
  loader: (signal: AbortSignal) => Promise<T>,
  deps: readonly unknown[],
  options: { keepStale?: boolean } = {},
): Resource<T> & { reload: () => void } {
  const [state, setState] = useState<Resource<T>>({ status: "loading", data: null, error: null });
  const [nonce, setNonce] = useState(0);
  const controller = useRef<AbortController | null>(null);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  useEffect(() => {
    const abort = new AbortController();
    controller.current = abort;
    let cancelled = false;
    setState((previous) => {
      if (options.keepStale && previous.data !== null) {
        return previous;
      }
      return { status: "loading", data: previous.data, error: null };
    });
    loaderRef
      .current(abort.signal)
      .then((data) => {
        if (cancelled || abort.signal.aborted) return;
        setState({ status: "ready", data, error: null });
      })
      .catch((error: unknown) => {
        if (cancelled || abort.signal.aborted) return;
        if (error instanceof DOMException && error.name === "AbortError") return;
        setState({
          status: "error",
          data: null,
          error: error instanceof Error ? error : new Error(String(error)),
        });
      });
    return () => {
      cancelled = true;
      abort.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  const reload = useCallback(() => setNonce((value) => value + 1), []);
  return useMemo(() => ({ ...state, reload }), [state, reload]);
}

/** Theme is a document attribute so plain CSS tokens can flip, not just Tailwind. */
export type Theme = "dark" | "light";

export function useTheme(): { theme: Theme; toggle: () => void } {
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = window.localStorage.getItem("vl-theme");
    return stored === "light" ? "light" : "dark";
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("vl-theme", theme);
  }, [theme]);
  return { theme, toggle: () => setTheme((value) => (value === "dark" ? "light" : "dark")) };
}
