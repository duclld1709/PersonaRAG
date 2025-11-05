export type MessageRole = "user" | "assistant" | "system";

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
  uploadedAt: string;
  updatedAt: string;
}

export interface DocumentDetail extends DocumentItem {
  content: string;
}

export interface Citation {
  source: string;
  filename: string;
  page?: number | null;
  content: string;
}

export type EmbeddingProvider = "openai" | "huggingface";
export type LlmProvider = "openai" | "gemini";
export type OpenAiModel = "gpt-4o" | "gpt-4o-mini";
export type GeminiModel = "gemini-2.5-pro" | "gemini-2.5-flash" | "gemini-2.0-flash";

export interface NumericRangeOption {
  min: number;
  max: number;
  step: number;
}

export interface AppSettings {
  queryVariations: number;
  retrievalK: number;
  retrievalLambda: number;
  embeddingProvider: EmbeddingProvider;
  llmProvider: LlmProvider;
  chatModel: OpenAiModel;
  geminiChatModel: GeminiModel;
}

export interface AppSettingsOptions {
  embeddingProviders: EmbeddingProvider[];
  llmProviders: LlmProvider[];
  openaiModels: OpenAiModel[];
  geminiModels: GeminiModel[];
  queryVariations: NumericRangeOption;
  retrievalK: NumericRangeOption;
  retrievalLambda: NumericRangeOption;
}

export type SettingsResponse = AppSettings & { options: AppSettingsOptions };

export type SettingsUpdatePayload = Partial<
  Pick<
    AppSettings,
    | "queryVariations"
    | "retrievalK"
    | "retrievalLambda"
    | "embeddingProvider"
    | "llmProvider"
    | "chatModel"
    | "geminiChatModel"
  >
>;


