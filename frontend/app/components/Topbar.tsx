"use client";

import { BookOpen, ChevronLeft, Moon, PanelLeftClose, PanelLeftOpen, Settings, Sun } from "lucide-react";

import { cn } from "../lib/cn";
import CostMeter from "./CostMeter";

type TopbarProps = {
  title: string;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
  contextOpen: boolean;
  onToggleContext: () => void;
  theme: "light" | "dark";
  onToggleTheme: () => void;
  onOpenSettings: () => void;
  costUsd: number;
  latencyMs: number;
  cacheHit: boolean;
};

export default function Topbar({
  title,
  sidebarOpen,
  onToggleSidebar,
  contextOpen,
  onToggleContext,
  theme,
  onToggleTheme,
  onOpenSettings,
  costUsd,
  latencyMs,
  cacheHit,
}: TopbarProps) {
  return (
    <header className="flex h-14 shrink-0 items-center gap-2 border-b border-border bg-surface/70 px-3 backdrop-blur">
      <button
        type="button"
        onClick={onToggleSidebar}
        className="btn-ghost h-9 w-9 p-0"
        aria-label={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
        title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
      >
        {sidebarOpen ? (
          <PanelLeftClose className="h-4 w-4" aria-hidden />
        ) : (
          <PanelLeftOpen className="h-4 w-4" aria-hidden />
        )}
      </button>

      <div className="flex min-w-0 flex-1 items-center gap-2">
        <BookOpen className="h-4 w-4 text-primary" aria-hidden />
        <h1 className="line-clamp-1 text-sm font-semibold">{title}</h1>
      </div>

      <div className="hidden items-center md:flex">
        <CostMeter costUsd={costUsd} latencyMs={latencyMs} cacheHit={cacheHit} />
      </div>

      <button
        type="button"
        onClick={onToggleTheme}
        className="btn-ghost h-9 w-9 p-0"
        aria-label="Toggle theme"
        title="Toggle theme"
      >
        {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
      </button>

      <button
        type="button"
        onClick={onOpenSettings}
        className="btn-ghost h-9 w-9 p-0"
        aria-label="Open settings"
        title="Settings"
      >
        <Settings className="h-4 w-4" aria-hidden />
      </button>

      <button
        type="button"
        onClick={onToggleContext}
        className={cn(
          "btn-ghost h-9 w-9 p-0",
          contextOpen && "bg-muted text-text"
        )}
        aria-label={contextOpen ? "Hide context" : "Show context"}
        title={contextOpen ? "Hide context panel" : "Show context panel"}
      >
        <ChevronLeft
          className={cn("h-4 w-4 transition-transform", contextOpen ? "rotate-180" : "rotate-0")}
          aria-hidden
        />
      </button>
    </header>
  );
}
