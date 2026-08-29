import { useEffect, useState } from 'react';
import { api } from '../api';
import { fmt, fmtInt, fmtDate } from '../ui';
import { LineChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid, AreaChart, Area } from 'recharts';

export function AIGrowth() {
  const [growth, setGrowth] = useState<any>(null);
  const [range, setRange] = useState('7d');
  const [charts, setCharts] = useState<any>(null);

  async function refresh() {
    try {
      const [g, c] = await Promise.all([
        api.get<any>('/dashboard/growth'),
        api.get<any>(`/dashboard/growth/charts?range=${range}`),
      ]);
      setGrowth(g); setCharts(c);
    } catch { /* */ }
  }
  useEffect(() => { refresh(); }, [range]);

  if (!growth) return <div className="muted">Loading…</div>;
  const loss = (charts?.training_loss || []).map((p: any, i: number) => ({ i, loss: p.loss }));
  const valLoss = (charts?.validation_loss || []).map((p: any, i: number) => ({ i, loss: p.loss, ppl: p.perplexity }));
  const tokens = (charts?.tokens_over_time || []).map((p: any, i: number) => ({ i, tokens: p.tokens }));
  const speed = (charts?.training_speed || []).map((p: any, i: number) => ({ i, tps: p.tps }));

  return (
    <div>
      <div className="page-header flex between items-center">
        <div><h1>AI Growth Dashboard</h1><p>Development of the model over time</p></div>
        <select value={range} onChange={e => setRange(e.target.value)} style={{ width: 'auto' }}>
          <option value="24h">24 hours</option><option value="7d">7 days</option><option value="30d">30 days</option><option value="all">All Time</option>
        </select>
      </div>

      <div className="grid grid-4 mb-16">
        <div className="stat"><div className="label">Current Model Version</div><div className="value" style={{fontSize:18}}>{growth.current_model_version || '—'}</div></div>
        <div className="stat"><div className="label">Parameters</div><div className="value">{fmtInt(growth.parameter_count)}</div></div>
        <div className="stat"><div className="label">Vocabulary Size</div><div className="value">{fmtInt(growth.vocab_size)}</div></div>
        <div className="stat"><div className="label">Total Training Tokens</div><div className="value">{fmtInt(growth.total_training_tokens)}</div><div className="sub estimate">≈ {fmtInt(growth.estimated_words)} words</div></div>
        <div className="stat"><div className="label">Datasets Learned</div><div className="value">{growth.datasets_learned}</div></div>
        <div className="stat"><div className="label">Completed Jobs</div><div className="value">{growth.completed_training_jobs}</div></div>
        <div className="stat"><div className="label">Training Hours</div><div className="value">{fmt(growth.training_hours, 2)}</div></div>
        <div className="stat"><div className="label">Growth Score</div><div className="value">{fmt(growth.growth_score, 1)}%</div><div className="sub estimate">composite</div></div>
      </div>

      <div className="grid grid-2">
        <div className="card">
          <div className="card-title">Training Loss</div>
          {loss.length ? <ResponsiveContainer width="100%" height={200}><LineChart data={loss}><CartesianGrid strokeDasharray="3 3" stroke="var(--border)" /><XAxis dataKey="i" stroke="var(--text-dim)" fontSize={11} /><YAxis stroke="var(--text-dim)" fontSize={11} /><Tooltip contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)' }} /><Line type="monotone" dataKey="loss" stroke="var(--accent)" dot={false} /></LineChart></ResponsiveContainer> : <div className="muted">No data</div>}
        </div>
        <div className="card">
          <div className="card-title">Validation Loss & Perplexity</div>
          {valLoss.length ? <ResponsiveContainer width="100%" height={200}><LineChart data={valLoss}><CartesianGrid strokeDasharray="3 3" stroke="var(--border)" /><XAxis dataKey="i" stroke="var(--text-dim)" fontSize={11} /><YAxis stroke="var(--text-dim)" fontSize={11} /><Tooltip contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)' }} /><Line type="monotone" dataKey="loss" stroke="var(--accent-2)" dot={false} /><Line type="monotone" dataKey="ppl" stroke="var(--yellow)" dot={false} /></LineChart></ResponsiveContainer> : <div className="muted">No validation data</div>}
        </div>
        <div className="card">
          <div className="card-title">Tokens Over Time</div>
          {tokens.length ? <ResponsiveContainer width="100%" height={200}><AreaChart data={tokens}><CartesianGrid strokeDasharray="3 3" stroke="var(--border)" /><XAxis dataKey="i" stroke="var(--text-dim)" fontSize={11} /><YAxis stroke="var(--text-dim)" fontSize={11} /><Tooltip contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)' }} /><Area type="monotone" dataKey="tokens" stroke="var(--green)" fill="rgba(56,211,159,0.2)" /></AreaChart></ResponsiveContainer> : <div className="muted">No data</div>}
        </div>
        <div className="card">
          <div className="card-title">Training Speed (tokens/sec)</div>
          {speed.length ? <ResponsiveContainer width="100%" height={200}><LineChart data={speed}><CartesianGrid strokeDasharray="3 3" stroke="var(--border)" /><XAxis dataKey="i" stroke="var(--text-dim)" fontSize={11} /><YAxis stroke="var(--text-dim)" fontSize={11} /><Tooltip contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)' }} /><Line type="monotone" dataKey="tps" stroke="var(--blue)" dot={false} /></LineChart></ResponsiveContainer> : <div className="muted">No data</div>}
        </div>
      </div>

      {growth.growth_breakdown && (
        <div className="card mt-16">
          <div className="card-title">AI Growth Score Breakdown</div>
          <p className="muted text-sm mb-16">{growth.growth_breakdown.note}</p>
          <div className="grid grid-3">
            {Object.entries(growth.growth_breakdown.components || {}).map(([k, v]: any) => (
              <div key={k} className="stat"><div className="label">{k}</div><div className="value" style={{fontSize:16}}>{fmt(v*100,1)}%</div><div className="sub">weight {fmt((growth.growth_breakdown.weights[k]||0)*100,0)}%</div></div>
            ))}
          </div>
        </div>
      )}

      <div className="card mt-16">
        <div className="card-title">Model Versions</div>
        <table>
          <thead><tr><th>Version</th><th>Created</th><th>Growth Score</th><th>Eval</th><th>Retention</th></tr></thead>
          <tbody>{(charts?.model_versions || []).map((m: any, i: number) => (
            <tr key={i}><td>v{m.version}</td><td>{fmtDate(m.t)}</td><td>{fmt(m.growth_score,1)}%</td><td>{fmt(m.eval,2)}</td><td>{m.retention != null ? fmt(m.retention,2) : '—'}</td></tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  );
}
