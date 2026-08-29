import { useEffect, useState } from 'react';
import { api } from '../api';
import { StatusBadge, fmt, fmtInt, fmtTime, useInterval } from '../ui';
import { LineChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from 'recharts';

export function Dashboard() {
  const [status, setStatus] = useState<any>(null);
  const [growth, setGrowth] = useState<any>(null);
  const [charts, setCharts] = useState<any>(null);
  const [jobs, setJobs] = useState<any[]>([]);

  async function refresh() {
    try {
      const [s, g, c, j] = await Promise.all([
        api.get<any>('/dashboard/status'),
        api.get<any>('/dashboard/growth'),
        api.get<any>('/dashboard/growth/charts?range=24h'),
        api.get<any[]>('/training/jobs'),
      ]);
      setStatus(s); setGrowth(g); setCharts(c); setJobs(j.slice(0, 5));
    } catch (e) { /* ignore */ }
  }
  useEffect(() => { refresh(); }, []);
  useInterval(refresh, 3000);

  if (!status || !growth) return <div className="muted">Loading…</div>;

  const hw = status.hardware || {};
  const lossData = (charts?.training_loss || []).map((p: any, i: number) => ({ i, loss: p.loss }));
  const runningJob = jobs.find(j => j.status === 'running');

  return (
    <div>
      <div className="page-header flex between items-center">
        <div>
          <h1>Dashboard</h1>
          <p>Real-time overview of the AI system</p>
        </div>
        <StatusBadge status={status.ai_status} />
      </div>

      <div className="grid grid-4 mb-16">
        <div className="stat">
          <div className="label">AI Status</div>
          <div className="value" style={{ fontSize: 18 }}><StatusBadge status={status.ai_status} /></div>
          <div className="sub">{status.current_model ? `Model v${status.current_model.version}` : 'No production model'}</div>
        </div>
        <div className="stat">
          <div className="label">Growth Score</div>
          <div className="value">{fmt(growth.growth_score, 1)}%</div>
          <div className="sub estimate">composite — not scientific intelligence</div>
        </div>
        <div className="stat">
          <div className="label">Total Training Tokens</div>
          <div className="value">{fmtInt(growth.total_training_tokens)}</div>
          <div className="sub estimate">≈ {fmtInt(growth.estimated_words)} words (estimate)</div>
        </div>
        <div className="stat">
          <div className="label">Current Model</div>
          <div className="value" style={{ fontSize: 18 }}>{growth.current_model_version || '—'}</div>
          <div className="sub">{fmtInt(growth.parameter_count)} params · {fmtInt(growth.vocab_size)} vocab</div>
        </div>
      </div>

      {runningJob && (
        <div className="card">
          <div className="card-title">
            <span>🔴 Live Training — {runningJob.dataset_versions?.[runningJob.current_dataset_index]?.dataset_name || '…'}</span>
            <StatusBadge status={runningJob.status} />
          </div>
          <div className="grid grid-4 mb-16">
            <div><div className="label">Step</div><div className="text-lg">{runningJob.current_step}/{runningJob.total_steps}</div></div>
            <div><div className="label">Loss</div><div className="text-lg mono">{fmt(runningJob.current_loss)}</div></div>
            <div><div className="label">Tokens/sec</div><div className="text-lg">{fmt(runningJob.tokens_per_sec, 0)}</div></div>
            <div><div className="label">Elapsed</div><div className="text-lg">{fmtTime(runningJob.elapsed_seconds)}</div></div>
          </div>
          <div className="progress"><div className="bar" style={{ width: `${runningJob.progress_pct}%` }} /></div>
        </div>
      )}

      <div className="grid grid-2">
        <div className="card">
          <div className="card-title">Training Loss (24h)</div>
          {lossData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="i" stroke="var(--text-dim)" fontSize={11} />
                <YAxis stroke="var(--text-dim)" fontSize={11} />
                <Tooltip contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)' }} />
                <Line type="monotone" dataKey="loss" stroke="var(--accent)" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : <div className="muted">No training data yet</div>}
        </div>
        <div className="card">
          <div className="card-title">System Resources (live)</div>
          <div className="grid grid-2">
            <div className="stat"><div className="label">CPU</div><div className="value" style={{fontSize:18}}>{fmt(hw.cpu_percent, 0)}%</div><div className="sub">{hw.cpu_count} cores</div></div>
            <div className="stat"><div className="label">Memory</div><div className="value" style={{fontSize:18}}>{fmt(hw.memory_percent, 0)}%</div><div className="sub">{fmt(hw.memory_used_mb, 0)}/{fmt(hw.memory_total_mb, 0)} MB</div></div>
            {hw.gpu_name && (
              <>
                <div className="stat"><div className="label">GPU</div><div className="value" style={{fontSize:14}}>{hw.gpu_name}</div><div className="sub">{fmt(hw.gpu_utilization_pct,0)}% util</div></div>
                <div className="stat"><div className="label">VRAM</div><div className="value" style={{fontSize:18}}>{fmt(hw.vram_used_mb,0)}/{fmt(hw.vram_total_mb,0)} MB</div></div>
              </>
            )}
          </div>
          <div className="mt-16">
            <div className="label mb-8">Queue</div>
            <div className="flex gap-16">
              <span>Queued: <b>{status.queued_count}</b></span>
              <span>Evaluating: <b>{status.evaluating_count}</b></span>
              <span>Failed: <b>{status.failed_count}</b></span>
            </div>
          </div>
        </div>
      </div>

      <div className="card mt-16">
        <div className="card-title">Recent Training Jobs</div>
        <table>
          <thead><tr><th>Job</th><th>Status</th><th>Step</th><th>Loss</th><th>Tokens</th><th>Tokens/s</th><th>Retention</th></tr></thead>
          <tbody>
            {jobs.map(j => (
              <tr key={j.id}>
                <td className="mono text-sm">{j.id.slice(0,8)}</td>
                <td><StatusBadge status={j.status} /></td>
                <td>{j.current_step}/{j.total_steps}</td>
                <td className="mono">{fmt(j.current_loss)}</td>
                <td>{fmtInt(j.tokens_processed)}</td>
                <td>{fmt(j.tokens_per_sec, 0)}</td>
                <td>{fmt(j.retention_score, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
