import { useEffect, useRef, useState } from 'react';
import { api, streamSSE } from '../api';
import { fmt } from '../ui';

export function TestLab() {
  const [models, setModels] = useState<any[]>([]);
  const [mvId, setMvId] = useState('');
  const [prompt, setPrompt] = useState('What is the capital of');
  const [maxTok, setMaxTok] = useState(64);
  const [temp, setTemp] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(40);
  const [repPen, setRepPen] = useState(1.15);
  const [sample, setSample] = useState(true);
  const [output, setOutput] = useState('');
  const [meta, setMeta] = useState<any>(null);
  const [streaming, setStreaming] = useState(false);
  // comparison
  const [mvIdB, setMvIdB] = useState('');
  const [cmpResult, setCmpResult] = useState<any>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function refresh() {
    const mdls = await api.get<any[]>('/models');
    const all = mdls.flatMap(m => m.versions.map((v: any) => ({ ...v, model_name: m.name })));
    setModels(all);
    if (!mvId && all.length) setMvId(all[0].id);
    if (!mvIdB && all.length > 1) setMvIdB(all[1].id);
  }
  useEffect(() => { refresh(); }, []);

  async function generate() {
    if (!mvId) return;
    setOutput(''); setMeta(null); setStreaming(true);
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await streamSSE('/inference/generate/stream', {
        prompt, model_version_id: mvId, max_new_tokens: maxTok,
        temperature: temp, top_p: topP, top_k: topK, repetition_penalty: repPen, do_sample: sample,
      }, (event, data) => {
        if (data.delta) setOutput(o => o + data.delta);
        if (data.done) { setMeta(data); setStreaming(false); }
      }, ctrl.signal);
    } catch (e: any) { setStreaming(false); setOutput('Error: ' + e.message); }
  }

  async function compare() {
    if (!mvId || !mvIdB) return;
    setCmpResult(null);
    try {
      const r = await api.post<any>('/inference/compare', { prompt, model_version_id_a: mvId, model_version_id_b: mvIdB, max_new_tokens: maxTok, temperature: temp });
      setCmpResult(r);
    } catch (e: any) { setCmpResult({ error: e.message }); }
  }

  return (
    <div>
      <div className="page-header">
        <h1>AI Test Lab</h1>
        <p>Test the trained model with real generation. Streaming, sampling controls, model comparison.</p>
      </div>

      <div className="card">
        <div className="card-title">Generation</div>
        <div className="form-row">
          <label>Model version</label>
          <select value={mvId} onChange={e => setMvId(e.target.value)}>
            {models.map(m => <option key={m.id} value={m.id}>{m.model_name} v{m.version} ({m.status})</option>)}
          </select>
        </div>
        <div className="form-row">
          <label>Prompt</label>
          <textarea value={prompt} onChange={e => setPrompt(e.target.value)} rows={3} />
        </div>
        <div className="grid grid-3">
          <div className="form-row"><label>Max tokens</label><input type="number" value={maxTok} onChange={e => setMaxTok(+e.target.value)} /></div>
          <div className="form-row"><label>Temperature</label><input type="number" step="0.05" value={temp} onChange={e => setTemp(+e.target.value)} /></div>
          <div className="form-row"><label>Top-p</label><input type="number" step="0.05" value={topP} onChange={e => setTopP(+e.target.value)} /></div>
          <div className="form-row"><label>Top-k</label><input type="number" value={topK} onChange={e => setTopK(+e.target.value)} /></div>
          <div className="form-row"><label>Repetition penalty</label><input type="number" step="0.05" value={repPen} onChange={e => setRepPen(+e.target.value)} /></div>
          <div className="form-row"><label>Sampling</label><select value={sample ? '1' : '0'} onChange={e => setSample(e.target.value === '1')}><option value="1">Sample</option><option value="0">Greedy</option></select></div>
        </div>
        <button onClick={generate} disabled={streaming || !mvId}>{streaming ? 'Generating…' : 'Ask AI'}</button>
        {output && (
          <div className="mt-16">
            <div className="card-title">Response</div>
            <div className="chat-box" style={{ height: 'auto', minHeight: 100 }}>{output}</div>
            {meta && (
              <div className="muted text-sm mt-8">
                Model v{meta.model_version} · {meta.num_tokens} tokens · {fmt(meta.latency_ms, 0)} ms · {fmt(meta.tokens_per_sec, 1)} tok/s
              </div>
            )}
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-title">Model Comparison</div>
        <p className="muted text-sm mb-16">Ask the same prompt to two model versions and compare.</p>
        <div className="grid grid-2">
          <div className="form-row"><label>Model A</label>
            <select value={mvId} onChange={e => setMvId(e.target.value)}>{models.map(m => <option key={m.id} value={m.id}>{m.model_name} v{m.version}</option>)}</select>
          </div>
          <div className="form-row"><label>Model B</label>
            <select value={mvIdB} onChange={e => setMvIdB(e.target.value)}>{models.map(m => <option key={m.id} value={m.id}>{m.model_name} v{m.version}</option>)}</select>
          </div>
        </div>
        <button onClick={compare} disabled={!mvId || !mvIdB}>Compare</button>
        {cmpResult && !cmpResult.error && (
          <div className="grid grid-2 mt-16">
            <div className="card" style={{ marginBottom: 0 }}>
              <div className="card-title">A — v{cmpResult.a.model_version}</div>
              <div className="chat-box" style={{ height: 'auto' }}>{cmpResult.a.text}</div>
              <div className="muted text-sm mt-8">{cmpResult.a.num_tokens} tokens · {fmt(cmpResult.a.tokens_per_sec,1)} tok/s</div>
            </div>
            <div className="card" style={{ marginBottom: 0 }}>
              <div className="card-title">B — v{cmpResult.b.model_version}</div>
              <div className="chat-box" style={{ height: 'auto' }}>{cmpResult.b.text}</div>
              <div className="muted text-sm mt-8">{cmpResult.b.num_tokens} tokens · {fmt(cmpResult.b.tokens_per_sec,1)} tok/s</div>
            </div>
          </div>
        )}
        {cmpResult?.error && <div className="error mt-8">{cmpResult.error}</div>}
      </div>
    </div>
  );
}
