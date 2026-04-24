export type MessageRole = "user" | "assistant" | "system";

export type ChatMode = "answer" | "explain" | "summarize" | "compare" | "outline" | "quiz";

export type DocumentTag =
  | "lecture_notes"
  | "paper"
  | "textbook"
  | "assignment"
  | "reference"
  | "other";

export interface Citation {
  source: string;
  filename: string;
  page?: number | null;
  content: string;
  documentId?: string | null;
  chunkId?: string | null;
  score?: number | null;
  tag?: string | null;
}

export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  cachedTokens: number;
  costUsd: number;
}

export interface ConversationSummary {
  id: string;
  title: string;
  lastMessagePreview?: string | null;
  updatedAt: string;
}

export interface ConversationMessage {
  role: MessageRole;
  content: string;
  createdAt: string;
  citations?: Citation[];
  usage?: TokenUsage | null;
  latencyMs?: number | null;
  mode?: string | null;
}

export interface ConversationDetail {
  id: string;
  title: string;
  messages: ConversationMessage[];
  createdAt: string;
  updatedAt: string;
}

export interface DocumentItem {
  id: string;
  filename: string;
  summary?: string | null;
  tag: DocumentTag;
  chunkCount: number;
  uploadedAt: string;
  updatedAt: string;
}

export interface DocumentDetail extends DocumentItem {
  content: string;
}

export interface NumericRangeOption {
  min: number;
  max: number;
  step: number;
}

export interface AppSettingsOptions {
  chatModels: string[];
  utilityModels: string[];
  queryVariations: NumericRangeOption;
  retrievalK: NumericRangeOption;
  rerankTopN: NumericRangeOption;
  semanticCacheThreshold: NumericRangeOption;
}

export interface AppSettings {
  chatModel: string;
  utilityModel: string;
  embeddingModel: string;
  queryVariations: number;
  retrievalK: number;
  rerankTopN: number;
  semanticCacheThreshold: number;
  enableModeration: boolean;
}

export type SettingsResponse = AppSettings & { options: AppSettingsOptions };

export type SettingsUpdatePayload = Partial<
  Pick<
    AppSettings,
    | "chatModel"
    | "utilityModel"
    | "queryVariations"
    | "retrievalK"
    | "rerankTopN"
    | "semanticCacheThreshold"
    | "enableModeration"
  >
>;

export interface UsageModelBreakdown {
  tokens: number;
  costUsd: number;
  events: number;
}

export interface UsageSummary {
  totalEvents: number;
  totalTokens: number;
  totalCostUsd: number;
  cacheHits: number;
  avgLatencyMs: number;
  byModel: Record<string, UsageModelBreakdown>;
}

export type StreamStatus =
  | "routing"
  | "retrieving"
  | "reranking"
  | "generating"
  | "cache_hit"
  | "chitchat"
  | "refused"
  | "done";
