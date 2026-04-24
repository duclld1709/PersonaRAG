"use client";

import { BookOpen, FileText, MessageSquarePlus, Pencil, Trash2 } from "lucide-react";

import { cn } from "../lib/cn";
import type { ConversationSummary } from "../lib/types";

type TabKey = "chat" | "library";

type SidebarProps = {
  tab: TabKey;
  onTabChange: (tab: TabKey) => void;
  conversations: ConversationSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  onRename: (id: string) => void;
  onDelete: (id: string) => void;
  collapsed: boolean;
};

export default function Sidebar({
  tab,
  onTabChange,
  conversations,
  activeId,
  onSelect,
  onNewChat,
  onRename,
  onDelete,
  collapsed,
}: SidebarProps) {
  return (
    <aside
      className={cn(
        "flex h-full flex-col border-r border-border bg-panel transition-all duration-200",
        collapsed ? "w-0 overflow-hidden" : "w-[280px]"
      )}
      aria-label="Workspace sidebar"
    >
      <div className="flex items-center gap-2 border-b border-divider px-4 py-3">
        <BookOpen className="h-5 w-5 text-primary" aria-hidden />
        <div className="flex-1">
          <div className="text-sm font-semibold tracking-tight">PersonaRAG</div>
          <div className="text-[11px] text-subtext">Research Assistant</div>
        </div>
      </div>

      <nav className="flex gap-1 px-3 py-2" role="tablist" aria-label="Workspace tabs">
        <TabButton
          label="Chat"
          icon={<MessageSquarePlus className="h-4 w-4" aria-hidden />}
          active={tab === "chat"}
          onClick={() => onTabChange("chat")}
        />
        <TabButton
          label="Library"
          icon={<FileText className="h-4 w-4" aria-hidden />}
          active={tab === "library"}
          onClick={() => onTabChange("library")}
        />
      </nav>

      <div className="px-3 pb-2">
        <button type="button" className="btn-primary w-full" onClick={onNewChat}>
          <MessageSquarePlus className="h-4 w-4" aria-hidden />
          New chat
        </button>
      </div>

      <div className="px-3 pb-1 text-[11px] font-medium uppercase tracking-wide text-subtext">
        Recent
      </div>
      <ul className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-3" role="list">
        {conversations.length === 0 && (
          <li className="px-2 py-6 text-center text-xs text-subtext">
            No conversations yet.
          </li>
        )}
        {conversations.map((convo) => {
          const isActive = convo.id === activeId;
          return (
            <li key={convo.id} className="group relative">
              <button
                type="button"
                onClick={() => onSelect(convo.id)}
                className={cn(
                  "flex w-full flex-col items-start gap-0.5 rounded-lg px-2.5 py-2 text-left text-sm transition-colors",
                  isActive
                    ? "bg-primary-soft text-text"
                    : "text-text hover:bg-muted"
                )}
                aria-current={isActive ? "page" : undefined}
              >
                <span className="line-clamp-1 font-medium">{convo.title || "Untitled"}</span>
                {convo.lastMessagePreview && (
                  <span className="line-clamp-1 text-[11px] text-subtext">
                    {convo.lastMessagePreview}
                  </span>
                )}
              </button>
              <div className="pointer-events-none absolute right-1 top-1.5 flex opacity-0 transition-opacity group-hover:pointer-events-auto group-hover:opacity-100">
                <IconAction
                  label="Rename"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRename(convo.id);
                  }}
                >
                  <Pencil className="h-3.5 w-3.5" aria-hidden />
                </IconAction>
                <IconAction
                  label="Delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(convo.id);
                  }}
                  danger
                >
                  <Trash2 className="h-3.5 w-3.5" aria-hidden />
                </IconAction>
              </div>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}

function TabButton({
  label,
  icon,
  active,
  onClick,
}: {
  label: string;
  icon: React.ReactNode;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={cn(
        "flex flex-1 items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors",
        active
          ? "bg-surface text-text shadow-soft"
          : "text-subtext hover:bg-muted hover:text-text"
      )}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

function IconAction({
  children,
  onClick,
  label,
  danger,
}: {
  children: React.ReactNode;
  onClick: (e: React.MouseEvent<HTMLButtonElement>) => void;
  label: string;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      className={cn(
        "inline-flex h-6 w-6 items-center justify-center rounded-md text-subtext transition-colors hover:bg-surface",
        danger ? "hover:text-danger" : "hover:text-text"
      )}
    >
      {children}
    </button>
  );
}
