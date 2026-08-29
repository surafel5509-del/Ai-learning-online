// Central API client. Reads JWT from localStorage; throws on non-2xx.
const BASE = '';

function token(): string | null {
  return localStorage.getItem('ai_token');
}

export function setToken(t: string) {
  localStorage.setItem('ai_token', t);
}
export function clearToken() {
  localStorage.removeItem('ai_token');
}

async function request<T>(method: string, path: string, body?: any, isForm = false): Promise<T> {
  const headers: Record<string, string> = {};
  if (token()) headers['Authorization'] = `Bearer ${token()}`;
  let payload: BodyInit | undefined;
  if (body !== undefined) {
    if (isForm) {
      payload = body as BodyInit;
    } else {
      headers['Content-Type'] = 'application/json';
      payload = JSON.stringify(body);
    }
  }
  const res = await fetch(`${BASE}${path}`, { method, headers, body: payload });
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let data: any = text ? JSON.parse(text) : undefined;
  if (!res.ok) {
    const msg = data?.detail || data?.message || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data as T;
}

export const api = {
  get: <T>(p: string) => request<T>('GET', p),
  post: <T>(p: string, body?: any) => request<T>('POST', p, body),
  postForm: <T>(p: string, form: FormData) => request<T>('POST', p, form, true),
  patch: <T>(p: string, body?: any) => request<T>('PATCH', p, body),
  del: <T>(p: string) => request<T>('DELETE', p),
};

// SSE stream helper for training updates & generation
export async function streamSSE(
  path: string,
  body: any,
  onEvent: (eventName: string, data: any) => void,
  signal?: AbortSignal,
) {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token()) headers['Authorization'] = `Bearer ${token()}`;
  const res = await fetch(`${BASE}${path}`, { method: 'POST', headers, body: JSON.stringify(body), signal });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() || '';
    for (const part of parts) {
      let event = 'message';
      let dataStr = '';
      for (const line of part.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataStr += line.slice(5).trim();
      }
      if (dataStr) {
        try { onEvent(event, JSON.parse(dataStr)); } catch { onEvent(event, dataStr); }
      }
    }
  }
}

// GET-based SSE (for training stream which uses GET)
export async function streamGET(
  path: string,
  onEvent: (eventName: string, data: any) => void,
  signal?: AbortSignal,
) {
  const headers: Record<string, string> = {};
  if (token()) headers['Authorization'] = `Bearer ${token()}`;
  const res = await fetch(`${BASE}${path}`, { method: 'GET', headers, signal });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() || '';
    for (const part of parts) {
      let event = 'message';
      let dataStr = '';
      for (const line of part.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataStr += line.slice(5).trim();
      }
      if (dataStr) {
        try { onEvent(event, JSON.parse(dataStr)); } catch { onEvent(event, dataStr); }
      }
    }
  }
}
