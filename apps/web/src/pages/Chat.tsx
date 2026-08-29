import { useEffect, useRef, useState } from 'react';
import { api, streamSSE } from '../api';
import { fmt } from '../ui';

export function Chat() {
  const [convs, setConvs] = useState<any[]>([]);
  const [activeConv, setActiveConv] = useState<string | null>(null);
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState('');
  const [models, setModels] = useState<any[]>([]);
  const [mvId, setMvId] = useState('');
  const [maxTok, setMaxTok] = useState(128);
  const [temp, setTemp] = useState(0.7);
  const [useRag, setUseRag] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [feedbackMsg, setFeedbackMsg] = useState('');
  const abortRef = useRef<AbortController | null>(null);

  async function refresh() {
    const [cs, mdls] = await Promise.all([api.get<any[]>('/chat/conversations'), api.get<any[]>('/models')]);
    setConvs(cs);
    const all = mdls.flatMap(m => m.versions.map((v: any) => ({ ...v, model_name: m.name }))).filter(v => v.status === 'production' || v.status === 'validated');
    setModels(all);
    const prod = all.find(v => v.status === 'production');
    if (prod && !mvId) setMvId(prod.id);
  }
  useEffect(() => { refresh(); }, []);

  async function loadConv(id: string) {
    setActiveConv(id);
    const c = await api.get<any>(`/chat/conversations/${id}`);
    setMessages(c.messages);
  }

  async function send() {
    if (!input.trim() || streaming) return;
    const userMsg = { role: 'user', content: input, id: 'tmp-' + Date.now() };
    setMessages(m => [...m, userMsg]);
    setInput(''); setStreaming(true); setFeedbackMsg('');
    setMessages(m => [...m, { role: 'assistant', content: '', id: 'tmp-a-' + Date.now() }]);
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await streamSSE('/chat/send/stream', {
        conversation_id: activeConv || undefined, message: userMsg.content,
        model_version_id: mvId || undefined, max_new_tokens: maxTok,
        temperature: temp, use_rag: useRag,
      }, (event, data) => {
        if (data.delta) {
          setMessages(m => {
            const copy = [...m];
            copy[copy.length - 1] = { ...copy[copy.length - 1], content: (copy[copy.length - 1].content || '') + data.delta };
            return copy;
          });
        }
        if (data.done) {
          setStreaming(false);
          if (data.conversation_id && !activeConv) { setActiveConv(data.conversation_id); refresh(); }
          setMessages(m => {
            const copy = [...m];
            copy[copy.length - 1] = { ...copy[copy.length - 1], latency_ms: data.latency_ms, tokens_generated: data.num_tokens };
            return copy;
          });
        }
      }, ctrl.signal);
    } catch (e: any) { setStreaming(false); }
  }

  async function rate(msgId: string, rating: number) {
    try {
      await api.post(`/chat/messages/${msgId}/feedback`, { rating });
      setFeedbackMsg('Thanks for the feedback!');
    } catch (e: any) { setFeedbackMsg(e.message); }
  }

  async function newConv() { setActiveConv(null); setMessages([]); }

  return (
    <div>
      <div className="page-header">
        <h1>Chat</h1>
        <p>Chat with the production model. Streaming, RAG, memory, feedback.</p>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: 16, height: 'calc(100vh - 140px)' }}>
        <div className="card" style={{ marginBottom: 0, overflow: 'auto' }}>
          <button className="secondary w-full mb-16" onClick={newConv}>+ New</button>
          {convs.map(c => (
            <div key={c.id} onClick={() => loadConv(c.id)} style={{ padding: 8, cursor: 'pointer', borderRadius: 6, background: activeConv === c.id ? 'var(--card-2)' : 'transparent', fontSize: 12, marginBottom: 4 }}>
              {c.title}
            </div>
          ))}
        </div>
        <div className="card" style={{ marginBottom: 0, display: 'flex', flexDirection: 'column' }}>
          <div className="flex gap-8 mb-16" style={{ flexShrink: 0 }}>
            <select value={mvId} onChange={e => setMvId(e.target.value)} style={{ width: 'auto' }}>
              {models.map(m => <option key={m.id} value={m.id}>{m.model_name} v{m.version} ({m.status})</option>)}
            </select>
            <label style={{ display: 'flex', alignItems: 'center', gap: 4, margin: 0 }}><input type="checkbox" style={{ width: 'auto' }} checked={useRag} onChange={e => setUseRag(e.target.checked)} /> RAG</label>
            <input type="number" value={maxTok} onChange={e => setMaxTok(+e.target.value)} style={{ width: 80 }} title="max tokens" />
            <input type="number" step="0.1" value={temp} onChange={e => setTemp(+e.target.value)} style={{ width: 70 }} title="temperature" />
          </div>
          <div className="chat-box" style={{ flex: 1 }}>
            {messages.length === 0 && <div className="muted">Start a conversation…</div>}
            {messages.map((m, i) => (
              <div key={m.id || i} className={`msg ${m.role}`}>
                <div className="role">{m.role}{m.latency_ms ? ` · ${fmt(m.latency_ms,0)}ms · ${m.tokens_generated} tok` : ''}</div>
                <div className="content">{m.content || (streaming && i === messages.length - 1 ? '…' : '')}</div>
                {m.role === 'assistant' && m.id && !m.id.startsWith('tmp') && (
                  <div className="mt-8" style={{ display: 'flex', gap: 8 }}>
                    <button className="ghost" style={{ padding: '2px 6px', fontSize: 11 }} onClick={() => rate(m.id, 1)}>👍</button>
                    <button className="ghost" style={{ padding: '2px 6px', fontSize: 11 }} onClick={() => rate(m.id, -1)}>👎</button>
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="flex gap-8 mt-16" style={{ flexShrink: 0 }}>
            <textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }} rows={1} style={{ resize: 'none' }} placeholder="Message…" />
            <button onClick={send} disabled={streaming || !input.trim()}>{streaming ? '…' : 'Send'}</button>
          </div>
          {feedbackMsg && <div className="success text-sm mt-8">{feedbackMsg}</div>}
        </div>
      </div>
    </div>
  );
}
