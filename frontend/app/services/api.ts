export interface Source {
  title: string;
  section?: string;
  source_url: string;
  content: string;
}

export interface TraceData {
  user_input?: string;
  model_choice?: string;
  answer_detail?: string;
  search_terms?: string[];
  retrieved_chunk_count?: number;
  retrieve_time_ms?: number;
  system_prompt?: string;
  ttft_ms?: number;
  total_time_ms?: number;
  output_chars?: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  trace?: TraceData;
}

export interface KnowledgeBaseStatus {
  status: 'ready' | 'empty' | string;
  version: string;
  embedding_model?: string;
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

export function buildKnowledgeBase(model: string) {
  return requestJson<{ status: string; message: string; embedding_model?: string }>('/api/v1/knowledge_base/build', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ model }),
  });
}

export function cleanKnowledgeBase() {
  return requestJson<{ status: string; message: string }>('/api/v1/knowledge_base/clean', {
    method: 'POST',
  });
}

export function syncKnowledgeBase(model?: string) {
  return requestJson<{ status: string; message: string; embedding_model?: string }>('/api/v1/knowledge_base/sync', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ model }),
  });
}

export async function* streamChat(
  message: string,
  model_choice: string = 'local',
  api_key?: string,
  answer_detail: string = '标准'
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
    }),
  });

  if (!response.ok) {
    throw new Error('网络请求失败');
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
