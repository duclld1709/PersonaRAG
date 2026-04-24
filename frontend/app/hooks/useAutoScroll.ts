"use client";

import { useEffect, useRef } from "react";

/**
 * Scroll a container to the bottom whenever `trigger` changes,
 * unless the user has scrolled up on their own.
 */
export function useAutoScroll<T extends HTMLElement>(trigger: unknown) {
  const ref = useRef<T | null>(null);
  const stickToBottom = useRef(true);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const handle = () => {
      const distance = el.scrollHeight - el.clientHeight - el.scrollTop;
      stickToBottom.current = distance < 80;
    };
    el.addEventListener("scroll", handle);
    return () => el.removeEventListener("scroll", handle);
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!el || !stickToBottom.current) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [trigger]);

  return ref;
}
