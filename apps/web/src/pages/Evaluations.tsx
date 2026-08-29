import { useEffect, useState } from 'react';
import { api } from '../api';
import { fmt, fmtDate } from '../ui';

export function Evaluations() {
  const [evs, setEvs] = useState<any[]>([]);
  const [models, setModels] = useState<any[]>([]);
  const [selEv, setSelEv] = useState('');
  const [selMv, setSelMv] = useState('');
  const [results, setResults] = useState<any>(null);
  const [newTest, setNewTest] = useState({ question: '', expected: '', criteria: 'contains' });
  const [msg, setMsg] = useState('');

  async function refresh() {
    setEvs(await api.get<any[]>('/evaluations'));
    setModels((await api.get<any[]>('/models')).flatMap((m:any)=>m.versions.map((v:any)=>({...v, model_name:m.name}))));
  }
  useEffect(() => { refresh(); }, []);

  async function createBenchmark() {
    // create from first dataset version found
    const ds = await api.get<any[]>('/datasets');
    const dv = ds.flatMap(d=>d.versions).find((v:any)=>v);
    if (!dv) { setMsg('Create a dataset first'); return; }
    await api.post('/evaluations', { name: 'Benchmark '+new Date().toLocaleDateString(), kind: 'benchmark', source_dataset_version_id: dv.id });
    refresh();
  }

  async function createSuite() {
    await api.post('/evaluations', { name: 'Custom Suite '+new Date().toLocaleDateString(), kind: 'custom' });
    refresh();
  }

  async function addTest() {
    if (!selEv) return;
    await api.post(`/evaluations/${selEv}/tests`, newTest);
    setNewTest({ question: '', expected: '', criteria: 'contains' }); refresh();
  }

  async function run() {
    if (!selEv || !selMv) return;
    setMsg('Running evaluation (real generation)…');
    try {
      const r = await api.post<any>('/evaluations/run', { evaluation_id: selEv, model_version_id: selMv, max_new_tokens: 48 });
      setResults(r); setMsg(`✓ Mean score: ${r.mean_score.toFixed(2)} (${r.passed}/${r.num_tests} passed)`);
      setResults(await api.get<any>(`/evaluations/${selEv}/results/${selMv}`));
    } catch (e: any) { setMsg('✗ ' + e.message); }
  }

  const ev = evs.find(e => e.id === selEv);

  return (
    <div>
      <div className="page-header"><h1>Evaluations</h1><p>Benchmark suites (from datasets), custom tests, real model evaluation.</p></div>
      <div className="flex gap-8 mb-16">
        <button className="secondary" onClick={createBenchmark}>+ Benchmark from dataset</button>
        <button className="secondary" onClick={createSuite}>+ Custom suite</button>
      </div>
      <div className="grid" style={{ gridTemplateColumns: '300px 1fr' }}>
        <div className="card">
          <div className="card-title">Suites</div>
          {evs.map(e => (
            <div key={e.id} onClick={()=>setSelEv(e.id)} style={{padding:8, cursor:'pointer', borderRadius:6, background: selEv===e.id?'var(--card-2)':'transparent'}}>
              <div>{e.name}</div><div className="muted text-sm">{e.kind} · {e.num_tests} tests</div>
            </div>
          ))}
        </div>
        <div>
          {ev && (
            <div className="card">
              <div className="card-title">{ev.name} — Tests</div>
              <table>
                <thead><tr><th>Question</th><th>Expected</th><th>Criteria</th></tr></thead>
                <tbody>{ev.tests.map((t:any)=>(<tr key={t.id}><td>{t.question}</td><td>{t.expected}</td><td>{t.criteria}</td></tr>))}</tbody>
              </table>
              <div className="card-title mt-16">Add Custom Test</div>
              <div className="grid grid-3">
                <div className="form-row"><label>Question</label><input value={newTest.question} onChange={e=>setNewTest({...newTest, question:e.target.value})} /></div>
                <div className="form-row"><label>Expected answer</label><input value={newTest.expected} onChange={e=>setNewTest({...newTest, expected:e.target.value})} /></div>
                <div className="form-row"><label>Criteria</label><select value={newTest.criteria} onChange={e=>setNewTest({...newTest, criteria:e.target.value})}><option>contains</option><option>exact</option><option>similarity</option></select></div>
              </div>
              <button onClick={addTest} disabled={!newTest.question.trim()}>Add Test</button>
              <div className="card-title mt-16">Run Evaluation</div>
              <div className="flex gap-8">
                <select value={selMv} onChange={e=>setSelMv(e.target.value)} style={{width:'auto'}}><option value="">Select model…</option>{models.map(m=><option key={m.id} value={m.id}>{m.model_name} v{m.version}</option>)}</select>
                <button onClick={run} disabled={!selMv}>Run</button>
              </div>
              {msg && <div className="mt-8">{msg}</div>}
              {results?.results && (
                <table className="mt-16"><thead><tr><th>Question</th><th>Response</th><th>Expected</th><th>Score</th><th>Latency</th></tr></thead>
                  <tbody>{results.results.map((r:any,i:number)=>(<tr key={i}><td>{r.question}</td><td className="text-sm">{r.response?.slice(0,60)}</td><td>{r.expected_answer}</td><td>{r.score!=null?fmt(r.score,2):'—'}</td><td>{fmt(r.latency_ms,0)}ms</td></tr>))}</tbody>
                </table>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
