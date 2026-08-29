import { useEffect, useState } from 'react';
import { api } from '../api';
import { StatusBadge, fmtInt, fmtDate } from '../ui';

export function Models() {
  const [models, setModels] = useState<any[]>([]);
  const [registry, setRegistry] = useState<any[]>([]);

  async function refresh() {
    setModels(await api.get<any[]>('/models'));
    setRegistry(await api.get<any[]>('/models/registry/all'));
  }
  useEffect(() => { refresh(); }, []);

  async function promote(id: string) {
    try { await api.post(`/models/${id}/promote`); refresh(); } catch (e: any) { alert(e.message); }
  }
  async function rollback(id: string) {
    try { await api.post(`/models/${id}/rollback`); refresh(); } catch (e: any) { alert(e.message); }
  }

  return (
    <div>
      <div className="page-header"><h1>Model Registry</h1><p>All model versions, lineage, statuses, promotion & rollback.</p></div>
      <div className="card">
        <div className="card-title">Model Versions</div>
        <table>
          <thead><tr><th>Model</th><th>Version</th><th>Parent</th><th>Params</th><th>Vocab</th><th>Tokens</th><th>Status</th><th>Growth</th><th>Created</th><th>Actions</th></tr></thead>
          <tbody>{registry.map(m => (
            <tr key={m.id}>
              <td>{m.model_name || m.id.slice(0,6)}</td><td>v{m.version}</td>
              <td className="muted">{m.parent_model_version_id ? '↗' : 'root'}</td>
              <td>{fmtInt(m.parameter_count)}</td><td>{fmtInt(m.vocab_size)}</td><td>{fmtInt(m.training_tokens)}</td>
              <td><StatusBadge status={m.status} /></td><td>{m.growth_score?.toFixed(1)}%</td>
              <td>{fmtDate(m.created_at)}</td>
              <td>
                {(m.status==='validated'||m.status==='candidate') && <button className="ghost" style={{padding:'2px 6px',fontSize:11}} onClick={()=>promote(m.id)}>Promote</button>}
                {m.status==='archived' && <button className="ghost" style={{padding:'2px 6px',fontSize:11}} onClick={()=>rollback(m.id)}>Rollback</button>}
              </td>
            </tr>
          ))}</tbody>
        </table>
        {registry.length === 0 && <div className="muted mt-8">No models yet. Train one first.</div>}
      </div>
      {models.map(m => (
        <div className="card" key={m.id}>
          <div className="card-title">{m.name} <span className="muted text-sm">{m.description}</span></div>
          <table>
            <thead><tr><th>Version</th><th>Status</th><th>Architecture</th><th>Promotion</th><th>Eval metrics</th><th>Retention</th></tr></thead>
            <tbody>{m.versions.map((v: any) => (
              <tr key={v.id}><td>v{v.version}</td><td><StatusBadge status={v.status} /></td>
                <td className="mono text-sm">{v.architecture?.num_layers}L · {v.architecture?.hidden_size}H · {v.architecture?.num_heads}/{v.architecture?.num_kv_heads} heads</td>
                <td className="text-sm">{v.promotion_passed ? '✓ passed' : '✗ ' + (v.promotion_reason||'')}</td>
                <td className="text-sm">{v.evaluation_metrics ? `score ${v.evaluation_metrics.mean_score?.toFixed(2)} (${v.evaluation_metrics.passed||0}/${v.evaluation_metrics.num_tests||0})` : '—'}</td>
                <td className="text-sm">{v.retention_metrics?.retention_score?.toFixed(2) ?? '—'}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      ))}
    </div>
  );
}
