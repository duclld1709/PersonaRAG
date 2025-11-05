"use client";

import clsx from "clsx";
import type { KeyboardEvent } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ConversationSummary } from "../../lib/types";

type SidebarProps = {
  conversations: ConversationSummary[];
  selectedId: string | null;
  onSelectChat: (id: string) => void;
  onNewChat: () => Promise<void> | void;
};

export function Sidebar({ conversations, selectedId, onSelectChat, onNewChat }: SidebarProps) {
  const listRef = useRef<HTMLDivElement | null>(null);
  const initialIndex = useMemo(() => {
    const index = conversations.findIndex((item) => item.id === selectedId);
    return index >= 0 ? index : 0;
  }, [conversations, selectedId]);
  const [highlightIndex, setHighlightIndex] = useState(initialIndex);

  useEffect(() => {
    setHighlightIndex(initialIndex);
  }, [initialIndex]);

  const moveHighlight = useCallback(
    (direction: 1 | -1) => {
      if (conversations.length === 0) {
        return;
      }
      setHighlightIndex((current) => {
        const next = (current + direction + conversations.length) % conversations.length;
        return next;
      });
    },
    [conversations.length]
  );

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (conversations.length === 0) {
        return;
      }

      if (event.key === "ArrowDown") {
        event.preventDefault();
        moveHighlight(1);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        moveHighlight(-1);
      } else if (event.key === "Enter") {
        event.preventDefault();
        const target = conversations[highlightIndex];
        if (target) {
          void onSelectChat(target.id);
        }
      }
    },
    [conversations, highlightIndex, moveHighlight, onSelectChat]
  );

  useEffect(() => {
    if (!listRef.current) {
      return;
    }
    const activeChild = listRef.current.querySelector<HTMLButtonElement>(
      `[data-index="${highlightIndex}"]`
    );
    if (activeChild) {
      activeChild.scrollIntoView({ block: "nearest" });
    }
  }, [highlightIndex]);

  return (
    <aside className="sticky left-0 top-0 flex h-screen w-[280px] flex-shrink-0 flex-col border-r border-border bg-card text-text shadow-lg shadow-black/10">
      <div className="flex items-center justify-between gap-2 border-b border-border px-5 py-4">
        <div className="flex flex-col">
          <span className="text-xs uppercase tracking-[0.18em] text-subtext">RAG Basic</span>
          <span className="text-sm font-semibold text-text">Chat Workspace</span>
        </div>
        <button
          type="button"
          onClick={() => void onNewChat()}
          className="rounded-full border border-border bg-primary px-4 py-2 text-xs font-medium text-primary-foreground transition hover:brightness-95 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
        >
          New chat
        </button>
      </div>

      <div
        ref={listRef}
        tabIndex={0}
        onKeyDown={handleKeyDown}
        data-virtualized={conversations.length > 50}
        className={clsx(
          "flex-1 overflow-y-auto px-2 py-4 focus:outline-none",
          conversations.length > 50 ? "pr-1" : ""
        )}
        role="listbox"
        aria-activedescendant={conversations[highlightIndex]?.id}
      >
        {conversations.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border bg-muted px-4 py-6 text-center text-sm text-subtext">
            No conversations yet
          </div>
        ) : (
          <ul className="space-y-1">
            {conversations.map((conversation, index) => {
              const isActive = conversation.id === selectedId;
              const isHighlighted = index === highlightIndex;
              return (
                <li key={conversation.id}>
                  <button
                    type="button"
                    data-index={index}
                    onClick={() => void onSelectChat(conversation.id)}
                    className={clsx(
                      "relative flex w-full items-center gap-2 rounded-lg border border-transparent px-4 py-2 text-left text-sm text-subtext transition duration-150",
                      "hover:bg-muted hover:text-text",
                      isActive && "border-primary bg-muted text-text shadow-sm",
                      (isActive || isHighlighted) && "shadow-inner shadow-black/10"
                    )}
                    title={conversation.title}
                  >
                    {isActive && (
                      <span className="absolute inset-y-1 left-1 w-1 rounded-full bg-primary" aria-hidden />
                    )}
                    <span className="truncate pl-2">{conversation.title}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </aside>
  );
}

export default Sidebar;
