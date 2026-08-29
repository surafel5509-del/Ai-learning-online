import { useEffect, useRef, useState } from 'react';
import { api, streamGET } from '../api';
import { StatusBadge, fmt, fmtInt, fmtTime } from '../ui';

export function Training() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [tokenizers, setTokenizers] = useState<any[]>([]);
  const [activeTok, setActiveTok] = useState<any>(null);
  const [hardware, setHardware] = useState<any>(null);
  const [tab, setTab] = useState<'tokenizer' | 'dataset' | 'train'>('tokenizer');

  // tokenizer form
  const [tokText, setTokText] = useState('');
  const [tokVocab, setTokVocab] = useState(400);
  const [tokMsg, setTokMsg] = useState('');
  // dataset form
  const [dsName, setDsName] = useState('');
  const [dsCat, setDsCat] = useState('General Knowledge');
  const [dsText, setDsText] = useState('');
  const [dsMsg, setDsMsg] = useState('');
  // train form
  const [selectedDvs, setSelectedDvs] = useState<string[]>([]);
  const [mode, setMode] = useState('fast');
  const [device, setDevice] = useState('auto');
  const [parentMv, setParentMv] = useState('');
  const [models, setModels] = useState<any[]>([]);
  const [hp, setHp] = useState({ epochs: 2, batch_size: 4, seq_len: 64, learning_rate: 0.0003, val_every: 10 });
  const [cfg, setCfg] = useState({ hidden_size: 96, num_layers: 2, num_heads: 4, num_kv_heads: 2, intermediate_size: 256, max_seq_len: 64 });
  const [trainMsg, setTrainMsg] = useState('');
  // live job
  const [liveJob, setLiveJob] = useState<any>(null);
  const [liveSteps, setLiveSteps] = useState<any[]>([]);
  const [plan, setPlan] = useState<any>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function refresh() {
    try {
      const [ds, tks, hw, mdls] = await Promise.all([
        api.get<any[]>('/datasets'),
        api.get<any[]>('/tokenizers'),
        api.get<any>('/training/hardware'),
        api.get<any[]>('/models'),
      ]);
      setDatasets(ds); setTokenizers(tks); setHardware(hw); setModels(mdls);
      const act = tks.find(t => t.is_active);
      setActiveTok(act);
    } catch { /* */ }
  }
  useEffect(() => { refresh(); }, []);

  async function trainTokenizer() {
    setTokMsg('Training tokenizer…');
    try {
      const texts = tokText.split('\n\n').filter(Boolean);
      const r = await api.post<any>('/tokenizers/train', { texts: texts.length ? texts : [tokText], target_vocab_size: tokVocab });
      setTokMsg(`✓ Trained tokenizer v${r.version} (vocab ${r.vocab_size}, ${r.num_merges} merges)`);
      setTokText(''); refresh();
    } catch (e: any) { setTokMsg('✗ ' + e.message); }
  }

  async function createDatasetVersion() {
    setDsMsg('Creating dataset…');
    try {
      const ds = await api.post<any>('/datasets', { name: dsName, knowledge_category: dsCat });
      const v = await api.post<any>(`/datasets/${ds.id}/versions/paste`, { text: dsText, filename: 'pasted.txt', deduplicate: true });
      setDsMsg(`✓ Dataset "${dsName}" v${v.version}: ${v.num_tokens} tokens, ${v.num_documents} docs`);
      setDsName(''); setDsText(''); refresh();
    } catch (e: any) { setDsMsg('✗ ' + e.message); }
  }

  async function computePlan() {
    if (!selectedDvs.length) { setPlan(null); return; }
    try {
      const p = await api.get<any>(`/training/plan?dataset_version_ids=${selectedDvs.join(',')}&mode=${mode}&seq_len=${hp.seq_len}&batch_size=${hp.batch_size}&epochs=${hp.epochs}`);
      setPlan(p);
    } catch { /* */ }
  }
  useEffect(() => { computePlan(); }, [selectedDvs, mode, hp]);

  async function startTraining() {
    if (!selectedDvs.length) { setTrainMsg('Select at least one dataset version'); return; }
    setTrainMsg('Starting…'); setLiveSteps([]);
    try {
      const body: any = {
        dataset_version_ids: selectedDvs, mode, device,
        hyperparams: hp, base_model_config: parentMv ? null : cfg,
      };
      if (parentMv) body.parent_model_version_id = parentMv;
      const job = await api.post<any>('/training/jobs', body);
      setTrainMsg(`✓ Job ${job.id.slice(0,8)} queued`);
      streamJob(job.id);
    } catch (e: any) { setTrainMsg('✗ ' + e.message); }
  }

  function streamJob(jobId: string) {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    streamGET(`/training/jobs/${jobId}/stream`, (event, data) => {
      if (event === 'step') setLiveSteps(s => [...s, data]);
      if (event === 'status') setLiveJob(data);
      if (event === 'done') { setLiveJob(data); }
    }, ctrl.signal);
  }

  const allDvs = datasets.flatMap(d => d.versions.map((v: any) => ({ ...v, dataset_name: d.name, dataset_id: d.id })));
  const allMvs = models.flatMap(m => m.versions.map((v: any) => ({ ...v, model_name: m.name })));

  return (
    <div>
      <div className="page-header">
        <h1>Training</h1>
        <p>Train a real Transformer model. Tokenizer → Dataset → Train.</p>
      </div>

      {hardware && (
        <div className="card mb-16">
          <div className="card-title">Hardware Detection (real)</div>
          <div className="flex gap-16">
            <span>Device: <b>{hardware.device}</b></span>
            <span>Name: <b>{hardware.name}</b></span>
            <span>CUDA: <b>{hardware.cuda_available ? 'Yes' : 'No'}</b></span>
            {hardware.cores && <span>Cores: <b>{hardware.cores}</b></span>}
            {hardware.vram_total_mb && <span>VRAM: <b>{fmtInt(hardware.vram_total_mb)} MB</b></span>}
          </div>
        </div>
      )}

      <div className="flex gap-8 mb-16">
        <button className={tab === 'tokenizer' ? '' : 'secondary'} onClick={() => setTab('tokenizer')}>1. Tokenizer</button>
        <button className={tab === 'dataset' ? '' : 'secondary'} onClick={() => setTab('dataset')}>2. Dataset</button>
        <button className={tab === 'train' ? '' : 'secondary'} onClick={() => setTab('train')}>3. Train</button>
      </div>

      {tab === 'tokenizer' && (
        <div className="card">
          <div className="card-title">Train Tokenizer <span className="muted text-sm">{activeTok ? `active: v${activeTok.version} (${activeTok.vocab_size})` : 'no active tokenizer'}</span></div>
          <div className="form-row">
            <label>Training text (paste raw text; the tokenizer learns byte-level BPE merges)</label>
            <textarea value={tokText} onChange={e => setTokText(e.target.value)} rows={6} placeholder="Paste text here. Supports Amharic/Ethiopic, English, any Unicode." />
          </div>
          <div className="form-row">
            <label>Target vocabulary size (min 260)</label>
            <input type="number" value={tokVocab} onChange={e => setTokVocab(+e.target.value)} min={260} />
          </div>
          <button onClick={trainTokenizer} disabled={!tokText.trim()}>Train Tokenizer</button>
          {tokMsg && <div className="mt-8">{tokMsg}</div>}
          {tokenizers.length > 0 && (
            <table className="mt-16">
              <thead><tr><th>Version</th><th>Vocab</th><th>Merges</th><th>Training tokens</th><th>Status</th></tr></thead>
              <tbody>
                {tokenizers.map(t => (
                  <tr key={t.id}><td>v{t.version}</td><td>{t.vocab_size}</td><td>{t.num_merges}</td><td>{fmtInt(t.training_tokens)}</td>
                    <td>{t.is_active ? <StatusBadge status="production" /> : 'inactive'}</td></tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {tab === 'dataset' && (
        <div className="card">
          <div className="card-title">Create Dataset + Version (paste text)</div>
          <div className="grid grid-2">
            <div className="form-row">
              <label>Dataset name</label>
              <input value={dsName} onChange={e => setDsName(e.target.value)} />
            </div>
            <div className="form-row">
              <label>Knowledge category</label>
              <select value={dsCat} onChange={e => setDsCat(e.target.value)}>
                {['General Knowledge','English','Amharic','Grammar','Technical Knowledge','Instructions','Conversation','User-provided Knowledge'].map(c => <option key={c}>{c}</option>)}
              </select>
            </div>
          </div>
          <div className="form-row">
            <label>Text content</label>
            <textarea value={dsText} onChange={e => setDsText(e.target.value)} rows={8} placeholder="Paste training text. Paragraphs (blank lines) become documents." />
          </div>
          <button onClick={createDatasetVersion} disabled={!dsName.trim() || !dsText.trim()}>Create & Process</button>
          {dsMsg && <div className="mt-8">{dsMsg}</div>}
          <div className="mt-16">
            <div className="card-title">Existing Datasets</div>
            <table>
              <thead><tr><th>Dataset</th><th>Category</th><th>Version</th><th>Tokens</th><th>Docs</th><th>Words (est.)</th></tr></thead>
              <tbody>
                {datasets.flatMap(d => d.versions.map((v: any) => (
                  <tr key={v.id}><td>{d.name}</td><td>{d.knowledge_category}</td><td>v{v.version}</td><td>{fmtInt(v.num_tokens)}</td><td>{v.num_documents}</td><td className="estimate">{fmtInt(v.estimated_words)}</td></tr>
                )))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === 'train' && (
        <>
          <div className="card">
            <div className="card-title">Configure Training Job</div>
            {!activeTok && <div className="error mb-8">No active tokenizer — train one in step 1 first.</div>}
            <div className="form-row">
              <label>Select dataset versions to train on (in order)</label>
              <div style={{ maxHeight: 160, overflow: 'auto', border: '1px solid var(--border)', borderRadius: 8, padding: 8 }}>
                {allDvs.map(v => (
                  <label key={v.id} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: 4, fontSize: 13, cursor: 'pointer' }}>
                    <input type="checkbox" style={{ width: 'auto' }}
                      checked={selectedDvs.includes(v.id)}
                      onChange={e => setSelectedDvs(s => e.target.checked ? [...s, v.id] : s.filter(x => x !== v.id))} />
                    {v.dataset_name} v{v.version} — {fmtInt(v.num_tokens)} tokens
                  </label>
                ))}
                {allDvs.length === 0 && <span className="muted">No datasets yet.</span>}
              </div>
            </div>
            <div className="grid grid-3">
              <div className="form-row">
                <label>Training mode</label>
                <select value={mode} onChange={e => setMode(e.target.value)}>
                  <option value="fast">FAST</option><option value="balanced">BALANCED</option><option value="deep">DEEP</option><option value="custom">CUSTOM</option>
                </select>
              </div>
              <div className="form-row">
                <label>Device</label>
                <select value={device} onChange={e => setDevice(e.target.value)}>
                  <option value="auto">AUTO</option><option value="cpu">CPU</option><option value="gpu">GPU</option>
                </select>
              </div>
              <div className="form-row">
                <label>Parent model (continual learning, optional)</label>
                <select value={parentMv} onChange={e => setParentMv(e.target.value)}>
                  <option value="">— Train from scratch —</option>
                  {allMvs.map(m => <option key={m.id} value={m.id}>{m.model_name} v{m.version} ({m.status})</option>)}
                </select>
              </div>
            </div>
            <div className="card-title mt-16">Hyperparameters</div>
            <div className="grid grid-3">
              {Object.entries(hp).map(([k, v]) => (
                <div className="form-row" key={k}>
                  <label>{k}</label>
                  <input type="number" value={v as number} onChange={e => setHp({...hp, [k]: +e.target.value})} />
                </div>
              ))}
            </div>
            {!parentMv && (
              <>
                <div className="card-title mt-16">Model Architecture (from scratch)</div>
                <div className="grid grid-3">
                  {Object.entries(cfg).map(([k, v]) => (
                    <div className="form-row" key={k}>
                      <label>{k}</label>
                      <input type="number" value={v as number} onChange={e => setCfg({...cfg, [k]: +e.target.value})} />
                    </div>
                  ))}
                </div>
              </>
            )}
            {plan && (
              <div className="mt-16 muted text-sm">
                Plan: {fmtInt(plan.total_train_tokens)} train tokens · {plan.total_steps} steps · est. {fmtTime(plan.estimated_seconds)} ·
                <span className="estimate"> ≈{fmtInt(plan.estimated_words)} words (estimate)</span>
              </div>
            )}
            <button className="mt-16" onClick={startTraining} disabled={!selectedDvs.length || !activeTok}>Start Training Job</button>
            {trainMsg && <div className="mt-8">{trainMsg}</div>}
          </div>

          {liveJob && (
            <div className="card">
              <div className="card-title">
                <span>Live Training — Job {liveJob.id?.slice(0,8)}</span>
                <StatusBadge status={liveJob.status} />
              </div>
              <div className="grid grid-4 mb-16">
                <div className="stat"><div className="label">Step</div><div className="value" style={{fontSize:18}}>{liveJob.current_step}/{liveJob.total_steps}</div></div>
                <div className="stat"><div className="label">Loss</div><div className="value mono" style={{fontSize:18}}>{fmt(liveJob.current_loss)}</div></div>
                <div className="stat"><div className="label">Tokens/sec</div><div className="value" style={{fontSize:18}}>{fmt(liveJob.tokens_per_sec,0)}</div></div>
                <div className="stat"><div className="label">Progress</div><div className="value" style={{fontSize:18}}>{fmt(liveJob.progress_pct,0)}%</div></div>
              </div>
              <div className="progress mb-16"><div className="bar" style={{ width: `${liveJob.progress_pct}%` }} /></div>
              {liveJob.final_perplexity != null && (
                <div className="muted text-sm">Final perplexity: {fmt(liveJob.final_perplexity)} · Retention: {fmt(liveJob.retention_score,2)} · Eval: {fmt(liveJob.evaluation_score,2)}</div>
              )}
              {liveSteps.length > 0 && (
                <div style={{ maxHeight: 200, overflow: 'auto' }}>
                  <table><thead><tr><th>Step</th><th>Epoch</th><th>Loss</th><th>LR</th><th>Tokens/s</th><th>Type</th></tr></thead>
                    <tbody>{liveSteps.slice(-30).reverse().map((s, i) => (
                      <tr key={i}><td>{s.step}</td><td>{s.epoch}</td><td className="mono">{fmt(s.loss)}</td><td className="mono">{fmt(s.learning_rate, 5)}</td><td>{fmt(s.tokens_per_sec,0)}</td><td>{s.is_validation ? <span className="badge blue">val</span> : 'train'}</td></tr>
                    ))}</tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
