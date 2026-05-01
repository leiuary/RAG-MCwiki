export interface Source {
  title: string;
  section?: string;
  source_url: string;
  content: string;
  content_length?: number;
  content_preview?: string;
}

export interface StepDurations {
  [step: string]: number;
}

export interface LLMConfig {
  backend: string;
  base_url: string;
  model: string;
  temperature: number;
  streaming: boolean;
}

export interface TraceData {
  status?: string;
  user_input?: string;
  model_choice?: string;
  answer_detail?: string;
  search_terms?: string[];
  retrieved_chunk_count?: number;
  retrieve_time_ms?: number;
  context_total_chars?: number;
  first_context_ms?: number;
  system_prompt?: string;
  llm_config?: LLMConfig;
  embedding_model?: string;
  embedding_device?: string;
  step_durations_ms?: StepDurations;
  execution_mode?: string;
  fallback_used?: boolean;
  fallback_reason?: string;
  ttft_ms?: number;
  total_time_ms?: number;
  output_chars?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  token_estimated?: boolean;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  trace?: TraceData;
}

export interface EmbeddingModelInfo {
  id: string;
  name: string;
  description: string;
  dim: number;
  is_built: boolean;
  is_default: boolean;
  is_active: boolean;
}

export interface KnowledgeBaseStatus {
  status: 'ready' | 'empty' | string;
  version: string;
  embedding_model?: string;
  active_model_id?: string;
  active_model_name?: string;
  available_models?: EmbeddingModelInfo[];
}

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getKnowledgeBaseStatus() {
  return requestJson<KnowledgeBaseStatus>('/api/v1/knowledge_base/status');
}

export function getEmbeddingModels() {
  return requestJson<{ models: EmbeddingModelInfo[] }>('/api/v1/knowledge_base/models');
}

export function switchEmbeddingModel(modelId: string) {
  return requestJson<KnowledgeBaseStatus>('/api/v1/knowledge_base/switch_model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId }),
  });
}

export function getModels(provider: string, baseUrl?: string, apiKey?: string) {
  return requestJson<{ models: string[] }>('/api/v1/models', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider, base_url: baseUrl, api_key: apiKey }),
  });
}

export async function* streamChat(
  message: string,
  model_choice: string = 'local',
  api_key?: string,
  answer_detail: string = '标准',
  base_url?: string,
  model_name?: string
) {
  const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      model_choice,
      api_key,
      answer_detail,
      base_url,
      model_name
    }),
  });

  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new Error(
      response.status === 503
        ? '服务正在初始化，请稍后重试'
        : body || `请求失败 (${response.status})`
    );
  }

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const extractData = (rawEvent: string) => {
    const dataStr = rawEvent
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n')
      .trim();

    return dataStr;
  };

  const findBoundary = (text: string) => {
    const lf = text.indexOf('\n\n');
    const crlf = text.indexOf('\r\n\r\n');

    if (lf === -1 && crlf === -1) return null;
    if (lf === -1) return { index: crlf, length: 4 };
    if (crlf === -1) return { index: lf, length: 2 };

    return lf < crlf ? { index: lf, length: 2 } : { index: crlf, length: 4 };
  };

  if (!reader) return;

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundary = findBoundary(buffer);
    while (boundary) {
      const rawEvent = buffer.slice(0, boundary.index).trim();
      buffer = buffer.slice(boundary.index + boundary.length);

      if (rawEvent) {
        const dataStr = extractData(rawEvent);

        if (dataStr === '[DONE]') return;

        if (dataStr) {
          try {
            const payload = JSON.parse(dataStr);
            yield payload;
          } catch (e) {
            console.error('解析 SSE 数据出错', e);
          }
        }
      }

      boundary = findBoundary(buffer);
    }

    if (done) {
      const finalEvent = buffer.trim();
      if (finalEvent) {
        const dataStr = extractData(finalEvent);
        if (dataStr === '[DONE]') return;

        if (dataStr) {
          try {
            const payload = JSON.parse(dataStr);
            yield payload;
          } catch (e) {
            console.error('解析 SSE 尾帧出错', e);
          }
        }
      }
      break;
    }
  }
}
