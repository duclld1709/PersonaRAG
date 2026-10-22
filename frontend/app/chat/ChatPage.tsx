"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import CitationPanel from "../components/CitationPanel";
import Composer from "../components/Composer";
import DocumentPanel from "../components/DocumentPanel";
import MessageList from "../components/MessageList";
import SettingsDrawer from "../components/SettingsDrawer";
import Sidebar from "../components/Sidebar";
import Topbar from "../components/Topbar";
import { useTheme } from "../hooks/useTheme";
import {
  createConversation,
  deleteConversation,
  deleteDocument,
  getAppSettings,
  getConversation,
  getUsageSummary,
  listConversations,
  listDocuments,
  renameConversation,
  streamChat,
  updateAppSettings,
  uploadDocument,
} from "../lib/api";
import type {
  ChatMode,
  Citation,
  ConversationDetail,
  ConversationSummary,
  DocumentItem,
  DocumentTag,
  SettingsResponse,
  SettingsUpdatePayload,
  StreamStatus,
  TokenUsage,
  UsageSummary,
} from "../lib/types";

type Tab = "chat" | "library";

const EMPTY_USAGE: TokenUsage = {
  promptTokens: 0,
  completionTokens: 0,
  cachedTokens: 0,
  costUsd: 0,
};

export default function ChatPage() {
  const { theme, toggle: toggleTheme } = useTheme();

  const [tab, setTab] = useState<Tab>("chat");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [contextOpen, setContextOpen] = useState(true);

  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [conversation, setConversation] = useState<ConversationDetail | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);

  const [mode, setMode] = useState<ChatMode>("answer");
  const [composer, setComposer] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamAnswer, setStreamAnswer] = useState("");
  const [streamCitations, setStreamCitations] = useState<Citation[]>([]);
  const [status, setStatus] = useState<StreamStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUsage, setLastUsage] = useState<TokenUsage>(EMPTY_USAGE);
  const [lastLatency, setLastLatency] = useState(0);
  const [lastCacheHit, setLastCacheHit] = useState(false);
  const [lastBreakdown, setLastBreakdown] = useState<Record<string, number>>({});
  const [hoveredCitation, setHoveredCitation] = useState<Citation | null>(null);

  const [settings, setSettings] = useState<SettingsResponse | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [usage, setUsage] = useState<UsageSummary | null>(null);

  const [docBusy, setDocBusy] = useState(false);
  const [docError, setDocError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  const activeCitations = useMemo<Citation[]>(() => {
    if (isStreaming) return streamCitations;
    const lastAssistant = [...(conversation?.messages ?? [])]
      .reverse()
      .find((m) => m.role === "assistant");
    return lastAssistant?.citations ?? [];
  }, [conversation, isStreaming, streamCitations]);

  // Bootstrap
  useEffect(() => {
    let cancelled = false;
    const boot = async () => {
      try {
        const [convos, docs, app] = await Promise.all([
          listConversations(),
          listDocuments(),
          getAppSettings(),
        ]);
        if (cancelled) return;
        setConversations(convos);
        setDocuments(docs);
        setSettings(app);
        if (convos.length > 0) {
          const first = convos[0];
          setActiveId(first.id);
          setConversation(await getConversation(first.id));
        } else {
          const created = await createConversation();
          setConversation(created);
          setActiveId(created.id);
          setConversations([{
            id: created.id,
            title: created.title,
            updatedAt: created.updatedAt,
            lastMessagePreview: null,
          }]);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to initialise.");
      }
    };
    void boot();
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshConversations = useCallback(async () => {
    try {
      const convos = await listConversations();
      setConversations(convos);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh conversations.");
    }
  }, []);

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await listDocuments());
    } catch (err) {
      setDocError(err instanceof Error ? err.message : "Failed to load documents.");
    }
  }, []);

  const refreshUsage = useCallback(async () => {
    try {
      setUsage(await getUsageSummary());
    } catch {
      /* usage is best-effort */
    }
  }, []);

  const handleSelect = useCallback(async (id: string) => {
    abortRef.current?.abort();
    setActiveId(id);
    try {
      setConversation(await getConversation(id));
      setStreamAnswer("");
      setStreamCitations([]);
      setStatus(null);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load conversation.");
    }
  }, []);

  const handleNewChat = useCallback(async () => {
    try {
      const created = await createConversation();
      setConversation(created);
      setActiveId(created.id);
      setConversations((prev) => [
        { id: created.id, title: created.title, updatedAt: created.updatedAt, lastMessagePreview: null },
        ...prev.filter((c) => c.id !== created.id),
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start a new chat.");
    }
  }, []);

  const handleRename = useCallback(async (id: string) => {
    const current = conversations.find((c) => c.id === id);
    const next = window.prompt("Rename conversation", current?.title ?? "");
    if (!next || !next.trim() || next === current?.title) return;
    try {
      const detail = await renameConversation(id, next.trim());
      setConversation((prev) => (prev?.id === id ? detail : prev));
      void refreshConversations();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to rename.");
    }
  }, [conversations, refreshConversations]);

  const handleDelete = useCallback(async (id: string) => {
    if (!window.confirm("Delete this conversation?")) return;
    try {
      await deleteConversation(id);
      const convos = await listConversations();
      setConversations(convos);
      if (activeId === id) {
        if (convos.length > 0) {
          const next = convos[0];
          setActiveId(next.id);
          setConversation(await getConversation(next.id));
        } else {
          setActiveId(null);
          setConversation(null);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete.");
    }
  }, [activeId]);

  const handleSend = useCallback(async () => {
    if (!activeId) return;
    const question = composer.trim();
    if (!question) return;

    setComposer("");
    setError(null);
    setIsStreaming(true);
    setStreamAnswer("");
    setStreamCitations([]);
    setStatus("routing");

    const optimisticTimestamp = new Date().toISOString();
    setConversation((prev) =>
      prev
        ? {
            ...prev,
            messages: [
              ...prev.messages,
              { role: "user", content: question, createdAt: optimisticTimestamp },
            ],
          }
        : prev
    );

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat(activeId, question, mode, {
        onStatus: (s) => setStatus(s),
        onToken: (token) => setStreamAnswer((prev) => prev + token),
        onCitations: (c) => setStreamCitations(c),
        onFinal: ({ usage: finalUsage, latencyMs, cacheHit, costBreakdown }) => {
          setLastUsage(finalUsage);
          setLastLatency(latencyMs);
          setLastCacheHit(cacheHit);
          setLastBreakdown(costBreakdown);
        },
        onError: (message) => setError(message),
      }, controller.signal);
    } catch (err) {
      if ((err as { name?: string }).name !== "AbortError") {
        setError(err instanceof Error ? err.message : "Streaming failed.");
      }
    }

    try {
      const refreshed = await getConversation(activeId);
      setConversation(refreshed);
      void refreshConversations();
      void refreshUsage();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh conversation.");
    }
    setIsStreaming(false);
    setStatus("done");
    setStreamAnswer("");
    setStreamCitations([]);
    abortRef.current = null;
  }, [activeId, composer, mode, refreshConversations, refreshUsage]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsStreaming(false);
    setStatus(null);
  }, []);

  const handleSuggestion = useCallback((question: string, suggestMode: ChatMode) => {
    setMode(suggestMode);
    setComposer(question);
  }, []);

  const handleUpload = useCallback(async (file: File, tag: DocumentTag) => {
    setDocBusy(true);
    setDocError(null);
    try {
      await uploadDocument({ file, filename: file.name, tag });
      await refreshDocuments();
    } catch (err) {
      setDocError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setDocBusy(false);
    }
  }, [refreshDocuments]);

  const handleDeleteDoc = useCallback(async (id: string) => {
    if (!window.confirm("Delete this document?")) return;
    setDocError(null);
    try {
      await deleteDocument(id);
      await refreshDocuments();
    } catch (err) {
      setDocError(err instanceof Error ? err.message : "Delete failed.");
    }
  }, [refreshDocuments]);

  const handleOpenSettings = useCallback(async () => {
    setSettingsOpen(true);
    setSettingsError(null);
    try {
      const [s, u] = await Promise.all([getAppSettings(), getUsageSummary()]);
      setSettings(s);
      setUsage(u);
    } catch (err) {
      setSettingsError(err instanceof Error ? err.message : "Failed to load settings.");
    }
  }, []);

  const handleSaveSettings = useCallback(async (payload: SettingsUpdatePayload) => {
    setSettingsSaving(true);
    setSettingsError(null);
    try {
      const next = await updateAppSettings(payload);
      setSettings(next);
      setSettingsOpen(false);
    } catch (err) {
      setSettingsError(err instanceof Error ? err.message : "Failed to save settings.");
    } finally {
      setSettingsSaving(false);
    }
  }, []);

  return (
    <div className="flex h-dvh w-screen overflow-hidden bg-background text-text">
      <Sidebar
        tab={tab}
        onTabChange={setTab}
        conversations={conversations}
        activeId={activeId}
        onSelect={handleSelect}
        onNewChat={handleNewChat}
        onRename={handleRename}
        onDelete={handleDelete}
        collapsed={!sidebarOpen}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar
          title={conversation?.title ?? "Research Assistant"}
          sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
          contextOpen={contextOpen}
          onToggleContext={() => setContextOpen((prev) => !prev)}
          theme={theme}
          onToggleTheme={toggleTheme}
          onOpenSettings={handleOpenSettings}
          costUsd={lastUsage.costUsd}
          latencyMs={lastLatency}
          cacheHit={lastCacheHit}
          costBreakdown={lastBreakdown}
        />

        <div className="flex min-h-0 flex-1">
          <div className="flex min-w-0 flex-1 flex-col">
            {tab === "chat" ? (
              <>
                <MessageList
                  conversation={conversation}
                  streamingAnswer={streamAnswer}
                  streamingCitations={streamCitations}
                  isStreaming={isStreaming}
                  status={isStreaming ? status : null}
                  error={error}
                  onCitationHover={setHoveredCitation}
                  onSuggestion={handleSuggestion}
                />
                <Composer
                  value={composer}
                  onChange={setComposer}
                  onSend={handleSend}
                  onStop={handleStop}
                  isStreaming={isStreaming}
                  disabled={!activeId}
                  mode={mode}
                  onModeChange={setMode}
                />
              </>
            ) : (
              <DocumentPanel
                documents={documents}
                isBusy={docBusy}
                error={docError}
                onUpload={handleUpload}
                onDelete={handleDeleteDoc}
              />
            )}
          </div>

          {contextOpen && tab === "chat" && (
            <CitationPanel
              citations={activeCitations}
              hovered={hoveredCitation}
              isStreaming={isStreaming}
            />
          )}
        </div>
      </div>

      <SettingsDrawer
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        usage={usage}
        isSaving={settingsSaving}
        error={settingsError}
        onSave={handleSaveSettings}
      />
    </div>
  );
}
