import { useEffect, useState } from 'react';
import { api } from '../api';
import { StatusBadge, fmt, fmtInt, fmtTime, fmtDate } from '../ui';
import { useInterval } from '../ui';

export function TrainingQueue() {
  const [jobs, setJobs] = useState<any[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [steps, setSteps] = useState<any[]>([]);

  async function refresh() { setJobs(await api.get<any[]>('/training/jobs')); }
  useEffect(() => { refresh(); }, []);
  useInterval(refresh, 3000);

  async function loadSteps(id: string) {
    setSel(id);
    setSteps(await api.get<any[]>(`/training/jobs/${id}/steps`));
  }

  async function control(id: string, action: string) {
    await api.post(`/training/jobs/${id}/control`, { action }); refresh();
  }

  return (
    <div>
      <div className="page-header"><h1>Training Queue</h1><p>Multi-dataset training jobs. Pause, resume, cancel.</p></div>
      <div className="grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <div className="card">
          <div className="card-title">Jobs</div>
          <table>
            <thead><tr><th>Status</th><th>Mode</th><th>Device</th><th>Step</th><th>Loss</th><th>Tokens/s</th><th>Actions</th></tr></thead>
            <tbody>{jobs.map(j => (
              <tr key={j.id} style={{ cursor: 'pointer', background: sel === j.id ? 'var(--card-2)' : '' }} onClick={()=>loadSteps(j.id)}>
                <td><StatusBadge status={j.status} /></td><td>{j.mode}</td><td>{j.device}</td>
                <td>{j.current_step}/{j.total_steps}</td><td className="mono">{fmt(j.current_loss)}</td><td>{fmt(j.tokens_per_sec,0)}</td>
                <td onClick={e=>e.stopPropagation()}>
                  {(j.status==='running'||j.status==='queued') && <button className="ghost" style={{padding:'2px 6px',fontSize:11,marginRight:4}} onClick={()=>control(j.id,'pause')}>Pause</button>}
                  {j.status==='paused' && <button className="ghost" style={{padding:'2px 6px',fontSize:11,marginRight:4}} onClick={()=>control(j.id,'resume')}>Resume</button>}
                  {(j.status==='running'||j.status==='queued'||j.status==='paused') && <button className="ghost" style={{padding:'2px 6px',fontSize:11}} onClick={()=>control(j.id,'cancel')}>Cancel</button>}
                </td>
              </tr>
            ))}</tbody>
          </table>
          {jobs.length === 0 && <div className="muted mt-8">No jobs yet.</div>}
        </div>
        <div className="card">
          <div className="card-title">Steps {sel ? `(${sel.slice(0,8)})` : ''}</div>
          {sel ? (
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table><thead><tr><th>Step</th><th>Loss</th><th>LR</th><th>Tokens/s</th><th>Type</th></tr></thead>
                <tbody>{steps.slice(-50).reverse().map((s,i)=>(
                  <tr key={i}><td>{s.step}</td><td className="mono">{fmt(s.loss)}</td><td className="mono">{fmt(s.learning_rate,5)}</td><td>{fmt(s.tokens_per_sec,0)}</td><td>{s.is_validation?'val':'train'}</td></tr>
                ))}</tbody>
              </table>
            </div>
          ) : <div className="muted">Select a job to view steps.</div>}
        </div>
      </div>
      {sel && <JobReport id={sel} />}
    </div>
  );
}

function JobReport({ id }: { id: string }) {
  const [j, setJ] = useState<any>(null);
  useEffect(() => { api.get<any>(`/training/jobs/${id}`).then(setJ); }, [id]);
  if (!j) return null;
  return (
    <div className="card mt-16">
      <div className="card-title">Training Report</div>
      <div className="grid grid-3">
        <div><div className="label">Dataset versions</div><div>{j.dataset_versions.length}</div></div>
        <div><div className="label">Tokens processed</div><div>{fmtInt(j.tokens_processed)}</div></div>
        <div><div className="label">Duration</div><div>{fmtTime(j.elapsed_seconds)}</div></div>
        <div><div className="label">Initial loss</div><div className="mono">{fmt(j.current_loss)}</div></div>
        <div><div className="label">Final loss</div><div className="mono">{fmt(j.final_loss)}</div></div>
        <div><div className="label">Final perplexity</div><div className="mono">{fmt(j.final_perplexity)}</div></div>
        <div><div className="label">Retention score</div><div>{fmt(j.retention_score,2)}</div></div>
        <div><div className="label">Evaluation score</div><div>{fmt(j.evaluation_score,2)}</div></div>
        <div><div className="label">Output model</div><div>{j.output_model_version ? 'v'+j.output_model_version.version : '—'}</div></div>
      </div>
      {j.error_message && <div className="error mt-8">{j.error_message}</div>}
    </div>
  );
}
