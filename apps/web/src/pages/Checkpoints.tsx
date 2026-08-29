import { useEffect, useState } from 'react';
import { api } from '../api';
import { fmt, fmtDate } from '../ui';

export function Checkpoints() {
  const [cps, setCps] = useState<any[]>([]);
  useEffect(() => { api.get<any[]>('/dashboard/checkpoints').then(setCps); }, []);
  return (
    <div>
      <div className="page-header"><h1>Checkpoints</h1><p>Saved model weights + optimizer/scheduler state. Resume, rollback, export.</p></div>
      <div className="card">
        <table>
          <thead><tr><th>Model Version</th><th>Step</th><th>Epoch</th><th>Val Loss</th><th>Val Perplexity</th><th>Latest</th><th>Best</th><th>Prev Prod</th><th>Created</th></tr></thead>
          <tbody>{cps.map(c => (
            <tr key={c.id}><td className="mono text-sm">{c.model_version_id.slice(0,8)}</td><td>{c.step}</td><td>{c.epoch}</td>
              <td className="mono">{fmt(c.val_loss)}</td><td className="mono">{fmt(c.val_perplexity)}</td>
              <td>{c.is_latest?'✓':''}</td><td>{c.is_best?'★':''}</td><td>{c.is_previous_production?'↩':''}</td>
              <td>{fmtDate(c.created_at)}</td></tr>
          ))}</tbody>
        </table>
        {cps.length === 0 && <div className="muted mt-8">No checkpoints yet.</div>}
      </div>
    </div>
  );
}
