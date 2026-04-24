"use client";

import type {
  AppSettingsOptions,
  ChatMode,
  Citation,
  ConversationDetail,
  ConversationMessage,
  ConversationSummary,
  DocumentDetail,
  DocumentItem,
  DocumentTag,
  SettingsResponse,
  SettingsUpdatePayload,
  StreamStatus,
  TokenUsage,
  UsageSummary,
} from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

async function http<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers as HeadersInit | undefined);
  if (!(init?.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(input, { ...init, headers });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

// ---------------------------------------------------------------------------
// Conversations
// ---------------------------------------------------------------------------

export async function listConversations(): Promise<ConversationSummary[]> {
  const data = await http<ConversationSummaryAPI[]>(`${API_BASE_URL}/conversations`);
  return data.map(mapConversationSummary);
}

export async function createConversation(title?: string): Promise<ConversationDetail> {
  const data = await http<ConversationDetailAPI>(`${API_BASE_URL}/conversations`, {
    method: "POST",
    body: JSON.stringify({ title }),
  });
  return mapConversationDetail(data);
}

export async function renameConversation(
  conversationId: string,
  title: string
): Promise<ConversationDetail> {
  const data = await http<ConversationDetailAPI>(
    `${API_BASE_URL}/conversations/${conversationId}`,
    { method: "PATCH", body: JSON.stringify({ title }) }
  );
  return mapConversationDetail(data);
}

export async function deleteConversation(conversationId: string): Promise<void> {
  await http<void>(`${API_BASE_URL}/conversations/${conversationId}`, { method: "DELETE" });
}

export async function getConversation(conversationId: string): Promise<ConversationDetail> {
  const data = await http<ConversationDetailAPI>(
    `${API_BASE_URL}/conversations/${conversationId}`
  );
  return mapConversationDetail(data);
}

// ---------------------------------------------------------------------------
// Chat streaming
// ---------------------------------------------------------------------------

export type StreamHandlers = {
  onToken?: (token: string) => void;
  onStatus?: (status: StreamStatus) => void;
  onCitations?: (citations: Citation[]) => void;
  onFinal?: (payload: {
    usage: TokenUsage;
    latencyMs: number;
    cacheHit: boolean;
    mode?: string;
  }) => void;
  onError?: (message: string) => void;
  onComplete?: () => void;
};

export async function streamChat(
  conversationId: string,
  question: string,
  mode: ChatMode,
  handlers: StreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ conversation_id: conversationId, question, mode }),
    signal,
  });
  if (!response.ok || !response.body) {
    const detail = await response.text();
    handlers.onError?.(detail || response.statusText);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completed = false;

  const processPayload = (raw: string) => {
    const trimmed = raw.trim();
    if (!trimmed) return;
    if (trimmed === "[DONE]") {
      completed = true;
      handlers.onComplete?.();
      return;
    }
    try {
      const parsed = JSON.parse(trimmed) as StreamPayloadAPI;
      if (parsed.error) {
        handlers.onError?.(parsed.error);
        completed = true;
        return;
      }
      if (parsed.status) {
        handlers.onStatus?.(parsed.status as StreamStatus);
      }
      if (typeof parsed.delta === "string" && parsed.delta) {
        handlers.onToken?.(parsed.delta);
      }
      if (parsed.citations) {
        handlers.onCitations?.(parsed.citations.map(mapCitation));
      }
      if (parsed.usage || parsed.latency_ms !== undefined || parsed.cache_hit !== undefined) {
        handlers.onFinal?.({
          usage: mapUsage(parsed.usage),
          latencyMs: parsed.latency_ms ?? 0,
          cacheHit: Boolean(parsed.cache_hit),
          mode: parsed.mode,
        });
      }
    } catch {
      handlers.onError?.("Received malformed streaming payload.");
      completed = true;
    }
  };

  const processChunk = (chunk: string) => {
    for (const line of chunk.split("\n")) {
      if (line.startsWith("data:")) processPayload(line.slice(5));
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let split = buffer.indexOf("\n\n");
      while (split !== -1) {
        const raw = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        processChunk(raw);
        split = buffer.indexOf("\n\n");
      }
    }
    buffer += decoder.decode();
    if (buffer.trim().length > 0) processChunk(buffer);
  } catch (error) {
    if ((error as { name?: string }).name === "AbortError") return;
    handlers.onError?.(error instanceof Error ? error.message : "Streaming interrupted.");
  } finally {
    reader.releaseLock();
    if (!completed) handlers.onComplete?.();
  }
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

export async function listDocuments(): Promise<DocumentItem[]> {
  const data = await http<DocumentAPI[]>(`${API_BASE_URL}/documents`);
  return data.map(mapDocument);
}

export async function uploadDocument(payload: {
  file: File;
  filename?: string;
  tag?: DocumentTag;
}): Promise<DocumentItem> {
  const formData = new FormData();
  formData.append("file", payload.file, payload.file.name);
  if (payload.filename) formData.append("filename", payload.filename);
  formData.append("tag", payload.tag ?? "other");
  const data = await http<DocumentAPI>(`${API_BASE_URL}/documents`, {
    method: "POST",
    body: formData,
  });
  return mapDocument(data);
}

export async function updateDocument(payload: {
  id: string;
  filename: string;
  content: string;
  tag: DocumentTag;
}): Promise<DocumentItem> {
  const data = await http<DocumentAPI>(`${API_BASE_URL}/documents/${payload.id}`, {
    method: "PUT",
    body: JSON.stringify({
      filename: payload.filename,
      content: payload.content,
      tag: payload.tag,
    }),
  });
  return mapDocument(data);
}

export async function getDocument(documentId: string): Promise<DocumentDetail> {
  const data = await http<DocumentDetailAPI>(`${API_BASE_URL}/documents/${documentId}`);
  return mapDocumentDetail(data);
}

export async function deleteDocument(documentId: string): Promise<void> {
  await http<void>(`${API_BASE_URL}/documents/${documentId}`, { method: "DELETE" });
}

// ---------------------------------------------------------------------------
// Settings & usage
// ---------------------------------------------------------------------------

export async function getAppSettings(): Promise<SettingsResponse> {
  const data = await http<SettingsResponseAPI>(`${API_BASE_URL}/settings`);
  return mapSettings(data);
}

export async function updateAppSettings(
  payload: SettingsUpdatePayload
): Promise<SettingsResponse> {
  const body: Record<string, unknown> = {};
  if (payload.chatModel !== undefined) body.chat_model = payload.chatModel;
  if (payload.utilityModel !== undefined) body.utility_model = payload.utilityModel;
  if (payload.queryVariations !== undefined) body.query_variations = payload.queryVariations;
  if (payload.retrievalK !== undefined) body.retrieval_k = payload.retrievalK;
  if (payload.rerankTopN !== undefined) body.rerank_top_n = payload.rerankTopN;
  if (payload.semanticCacheThreshold !== undefined)
    body.semantic_cache_threshold = payload.semanticCacheThreshold;
  if (payload.enableModeration !== undefined) body.enable_moderation = payload.enableModeration;
  const data = await http<SettingsResponseAPI>(`${API_BASE_URL}/settings`, {
    method: "PATCH",
    body: JSON.stringify(body),
  });
  return mapSettings(data);
}

export async function getUsageSummary(): Promise<UsageSummary> {
  const data = await http<UsageSummaryAPI>(`${API_BASE_URL}/usage/summary`);
  return {
    totalEvents: data.total_events,
    totalTokens: data.total_tokens,
    totalCostUsd: data.total_cost_usd,
    cacheHits: data.cache_hits,
    avgLatencyMs: data.avg_latency_ms,
    byModel: Object.fromEntries(
      Object.entries(data.by_model ?? {}).map(([k, v]) => [
        k,
        { tokens: v.tokens, costUsd: v.cost_usd, events: v.events },
      ])
    ),
  };
}

// ---------------------------------------------------------------------------
// Mapping + API shape definitions
// ---------------------------------------------------------------------------

type UsageAPI = {
  prompt_tokens?: number;
  completion_tokens?: number;
  cached_tokens?: number;
  cost_usd?: number;
};

type CitationAPI = {
  source: string;
  filename: string;
  page?: number | null;
  content: string;
  document_id?: string | null;
  chunk_id?: string | null;
  score?: number | null;
  tag?: string | null;
};

type StreamPayloadAPI = {
  delta?: string;
  citations?: CitationAPI[];
  status?: string;
  usage?: UsageAPI;
  latency_ms?: number;
  cache_hit?: boolean;
  mode?: string;
  intent?: string;
  error?: string;
};

type ConversationSummaryAPI = {
  id: string;
  title: string;
  last_message_preview?: string | null;
  updated_at: string;
};

type ConversationMessageAPI = {
  role: ConversationMessage["role"];
  content: string;
  created_at: string;
  citations?: CitationAPI[] | null;
  usage?: UsageAPI | null;
  latency_ms?: number | null;
  mode?: string | null;
};

type ConversationDetailAPI = {
  id: string;
  title: string;
  messages: ConversationMessageAPI[];
  created_at: string;
  updated_at: string;
};

type DocumentAPI = {
  id: string;
  filename: string;
  summary?: string | null;
  tag: DocumentTag;
  chunk_count: number;
  uploaded_at: string;
  updated_at: string;
};

type DocumentDetailAPI = DocumentAPI & { content: string };

type RangeAPI = { min: number; max: number; step: number };

type SettingsResponseAPI = {
  chat_model: string;
  utility_model: string;
  embedding_model: string;
  query_variations: number;
  retrieval_k: number;
  rerank_top_n: number;
  semantic_cache_threshold: number;
  enable_moderation: boolean;
  options: {
    chat_models: string[];
    utility_models: string[];
    query_variations: RangeAPI;
    retrieval_k: RangeAPI;
    rerank_top_n: RangeAPI;
    semantic_cache_threshold: RangeAPI;
  };
};

type UsageSummaryAPI = {
  total_events: number;
  total_tokens: number;
  total_cost_usd: number;
  cache_hits: number;
  avg_latency_ms: number;
  by_model: Record<string, { tokens: number; cost_usd: number; events: number }>;
};

function mapUsage(api: UsageAPI | undefined): TokenUsage {
  return {
    promptTokens: api?.prompt_tokens ?? 0,
    completionTokens: api?.completion_tokens ?? 0,
    cachedTokens: api?.cached_tokens ?? 0,
    costUsd: api?.cost_usd ?? 0,
  };
}

function mapCitation(api: CitationAPI): Citation {
  return {
    source: api.source,
    filename: api.filename,
    page: api.page ?? null,
    content: api.content,
    documentId: api.document_id ?? null,
    chunkId: api.chunk_id ?? null,
    score: api.score ?? null,
    tag: api.tag ?? null,
  };
}

function mapConversationSummary(api: ConversationSummaryAPI): ConversationSummary {
  return {
    id: api.id,
    title: api.title,
    lastMessagePreview: api.last_message_preview ?? null,
    updatedAt: api.updated_at,
  };
}

function mapConversationDetail(api: ConversationDetailAPI): ConversationDetail {
  return {
    id: api.id,
    title: api.title,
    createdAt: api.created_at,
    updatedAt: api.updated_at,
    messages: api.messages.map((msg) => ({
      role: msg.role,
      content: msg.content,
      createdAt: msg.created_at,
      citations: msg.citations ? msg.citations.map(mapCitation) : undefined,
      usage: msg.usage ? mapUsage(msg.usage) : null,
      latencyMs: msg.latency_ms ?? null,
      mode: msg.mode ?? null,
    })),
  };
}

function mapDocument(api: DocumentAPI): DocumentItem {
  return {
    id: api.id,
    filename: api.filename,
    summary: api.summary ?? null,
    tag: api.tag,
    chunkCount: api.chunk_count,
    uploadedAt: api.uploaded_at,
    updatedAt: api.updated_at,
  };
}

function mapDocumentDetail(api: DocumentDetailAPI): DocumentDetail {
  return { ...mapDocument(api), content: api.content };
}

function mapSettings(api: SettingsResponseAPI): SettingsResponse {
  const range = (r: RangeAPI) => ({ min: r.min, max: r.max, step: r.step });
  const options: AppSettingsOptions = {
    chatModels: api.options.chat_models,
    utilityModels: api.options.utility_models,
    queryVariations: range(api.options.query_variations),
    retrievalK: range(api.options.retrieval_k),
    rerankTopN: range(api.options.rerank_top_n),
    semanticCacheThreshold: range(api.options.semantic_cache_threshold),
  };
  return {
    chatModel: api.chat_model,
    utilityModel: api.utility_model,
    embeddingModel: api.embedding_model,
    queryVariations: api.query_variations,
    retrievalK: api.retrieval_k,
    rerankTopN: api.rerank_top_n,
    semanticCacheThreshold: api.semantic_cache_threshold,
    enableModeration: api.enable_moderation,
    options,
  };
}
