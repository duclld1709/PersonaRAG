"use client";

import clsx from "clsx";
import { Folder, Info, MessageSquarePlus, MoreHorizontal } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import {
  MouseEvent,
  ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties
} from "react";

import SidebarToggle from "./SidebarToggle";

export type SidebarListItem = {
  id: string;
  label: string;
  icon?: ReactNode;
};

type SidebarContentProps = {
  controlledId: string;
  heading: string;
  isOpen: boolean;
  isDesktop: boolean;
  items: SidebarListItem[];
  activeId?: string;
  onToggle: () => void;
  onNewChat: () => Promise<void> | void;
  onSelect: (id: string) => Promise<void> | void;
  onRename?: (id: string) => Promise<void> | void;
  onDelete?: (id: string) => Promise<void> | void;
  onShowAbout?: () => Promise<void> | void;
};

export default function SidebarContent({
  controlledId,
  heading,
  isOpen,
  isDesktop,
  items,
  activeId,
  onToggle,
  onNewChat,
  onSelect,
  onRename,
  onDelete,
  onShowAbout
}: SidebarContentProps) {
  const shouldRenderList = !isDesktop || isOpen;
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [menuFor, setMenuFor] = useState<string | null>(null);
  const prefersReducedMotion = useReducedMotion();
  const labelsVisible = !isDesktop || isOpen;
  const resolveLabelAnimation = useCallback(
    (visible: boolean) => {
      if (prefersReducedMotion) {
        return {
          opacity: visible ? 1 : 0,
          clipPath: visible ? "inset(0% 0% 0% 0%)" : "inset(0% 100% 0% 0%)",
          x: 0
        };
      }
      return {
        opacity: visible ? 1 : 0,
        clipPath: visible ? "inset(0% 0% 0% 0%)" : "inset(0% 100% 0% 0%)",
        x: visible ? 0 : -8
      };
    },
    [prefersReducedMotion]
  );
  const resolveLabelTransition = useCallback(
    (visible: boolean) =>
      prefersReducedMotion
        ? { duration: 0 }
        : { duration: 0.18, ease: "easeOut" as const, delay: visible ? 0.08 : 0 },
    [prefersReducedMotion]
  );
  const labelMotionStyle: CSSProperties | undefined = prefersReducedMotion
    ? undefined
    : { willChange: "opacity, clip-path, transform" };

  useEffect(() => {
    const handleClickOutside = (event: globalThis.MouseEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) {
        setMenuFor(null);
      }
    };

    const handleKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenuFor(null);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleKeydown);

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKeydown);
    };
  }, []);

  useEffect(() => {
    if (isDesktop && !isOpen) {
      setMenuFor(null);
    }
  }, [isDesktop, isOpen]);

  return (
    <div
      id={controlledId}
      ref={containerRef}
      className={clsx(
        "flex h-full flex-col border border-border bg-card text-text shadow-sm transition-colors duration-200",
        isDesktop ? "w-full" : "w-64"
      )}
    >
      <div
        className={clsx(
          "flex items-center gap-3 px-3 pb-3 pt-4 transition-all duration-200",
          isDesktop && !isOpen && "justify-end gap-0"
        )}
      >
        <SidebarToggle
          isOpen={isOpen}
          controlledId={controlledId}
          onClick={onToggle}
          className="flex-shrink-0"
        />
        {isDesktop && isOpen && (
          <div className="ml-2 flex flex-1 flex-col transition-all duration-200 ease-in-out">
            <span className="text-xs uppercase tracking-[0.18em] text-subtext">RAG Basic</span>
            <span className="text-sm font-semibold text-text">{heading}</span>
          </div>
        )}
      </div>

      <div className="px-3 pb-4">
        <button
          type="button"
          onClick={() => {
            void onNewChat();
          }}
          title="New chat"
          className={clsx(
            "flex w-full items-center rounded-2xl border border-border bg-primary text-sm font-medium text-primary-foreground shadow-sm transition hover:brightness-95 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",
            labelsVisible ? "gap-3" : "gap-0",
            "duration-200 ease-in-out",
            isDesktop && !isOpen ? "w-11 justify-center px-0 py-2" : "justify-start px-4 py-2"
          )}
        >
          <MessageSquarePlus className="h-4 w-4 flex-shrink-0" aria-hidden="true" />
          <motion.span
            className="inline-flex items-center gap-1 whitespace-nowrap overflow-hidden text-sm font-medium [text-wrap:nowrap]"
            animate={resolveLabelAnimation(labelsVisible)}
            initial={false}
            transition={resolveLabelTransition(labelsVisible)}
            aria-hidden={isDesktop && !isOpen}
            style={{
              ...labelMotionStyle,
              width: labelsVisible ? "auto" : 0
            }}
          >
            New chat
          </motion.span>
        </button>
      </div>

      {shouldRenderList && (
        <div className="flex-1 px-2 pb-6" aria-hidden={!isOpen}>
          <ul className="space-y-1">
            {items.length === 0 && (
              <li className="rounded-xl border border-border bg-muted px-3 py-2 text-sm text-subtext">
                No conversations yet.
              </li>
            )}
            {items.map((item) => {
              const isActive = item.id === activeId;
              return (
                <li key={item.id}>
                  <div
                    className={clsx(
                      "group relative flex items-start gap-2 rounded-2xl px-3 py-2 transition duration-200",
                      isActive
                        ? "border border-primary bg-muted text-text shadow-sm"
                        : "border border-transparent text-subtext hover:border-border hover:bg-muted hover:text-text"
                    )}
                  >
                    <button
                      type="button"
                      onClick={(event: MouseEvent<HTMLButtonElement>) => {
                        event.preventDefault();
                        setMenuFor(null);
                        void onSelect(item.id);
                      }}
                      className="flex flex-1 flex-col items-start gap-1 text-left"
                    >
                      <span className="flex items-center gap-2 overflow-hidden [contain:layout_paint]">
                        <span
                          className={clsx(
                            "flex h-8 w-8 items-center justify-center rounded-xl bg-white text-subtext transition-colors duration-200 group-hover:bg-muted group-hover:text-text",
                            isActive && "text-primary group-hover:text-primary"
                          )}
                        >
                          {item.icon ?? <Folder className="h-4 w-4" aria-hidden="true" />}
                        </span>
                        <motion.span
                          className="whitespace-nowrap overflow-hidden text-sm font-semibold leading-tight text-left [text-wrap:nowrap]"
                          animate={resolveLabelAnimation(labelsVisible)}
                          initial={false}
                          transition={resolveLabelTransition(labelsVisible)}
                          style={{
                            ...labelMotionStyle,
                            width: labelsVisible ? "auto" : 0,
                            maxWidth: labelsVisible ? "16ch" : 0
                          }}
                        >
                          {item.label}
                        </motion.span>
                      </span>
                    </button>
                    {(onRename || onDelete) && (isOpen || !isDesktop) && (
                      <>
                        <button
                          type="button"
                          className="ml-1 inline-flex h-8 w-8 items-center justify-center rounded-lg text-subtext transition hover:bg-muted hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                          aria-haspopup="true"
                          aria-expanded={menuFor === item.id}
                          aria-label="Conversation options"
                          onClick={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            setMenuFor((current) => (current === item.id ? null : item.id));
                          }}
                        >
                          <MoreHorizontal className="h-4 w-4" aria-hidden="true" />
                        </button>
                        {menuFor === item.id && (
                          <div
                            className="absolute right-2 top-12 z-20 w-40 rounded-xl border border-border bg-card p-1 shadow-lg shadow-black/10"
                            role="menu"
                          >
                            {onRename && (
                              <button
                                type="button"
                                className="flex w-full items-center rounded-lg px-3 py-2 text-left text-text transition hover:bg-muted"
                                onClick={(event) => {
                                  event.preventDefault();
                                  event.stopPropagation();
                                  setMenuFor(null);
                                  void onRename(item.id);
                                }}
                              >
                                Rename
                              </button>
                            )}
                            {onDelete && (
                              <button
                                type="button"
                                className="mt-1 flex w-full items-center rounded-lg px-3 py-2 text-left text-red-500 transition hover:bg-red-50"
                                onClick={(event) => {
                                  event.preventDefault();
                                  event.stopPropagation();
                                  setMenuFor(null);
                                  void onDelete(item.id);
                                }}
                              >
                                Delete
                              </button>
                            )}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {onShowAbout && (
        <div
          className={clsx(
            "mt-auto px-3 pb-4",
            isDesktop && !isOpen && "flex justify-center"
          )}
        >
          <button
            type="button"
            onClick={() => {
              void onShowAbout();
            }}
            title="About us"
            className={clsx(
              "flex w-full items-center rounded-2xl border border-border bg-card text-sm font-medium text-subtext transition hover:bg-muted hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",
              labelsVisible ? "gap-3" : "gap-0",
              isDesktop && !isOpen ? "w-11 justify-center px-0 py-2" : "px-4 py-2"
            )}
          >
            <Info className="h-4 w-4 flex-shrink-0" aria-hidden="true" />
            <motion.span
              className="inline-flex items-center gap-1 whitespace-nowrap overflow-hidden text-sm font-medium [text-wrap:nowrap]"
              animate={resolveLabelAnimation(labelsVisible)}
              initial={false}
              transition={resolveLabelTransition(labelsVisible)}
              aria-hidden={isDesktop && !isOpen}
              style={{
                ...labelMotionStyle,
                width: labelsVisible ? "auto" : 0
              }}
            >
              About us
            </motion.span>
          </button>
        </div>
      )}
    </div>
  );
}
