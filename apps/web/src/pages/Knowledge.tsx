import { useEffect, useState } from 'react';
import { api } from '../api';
import { fmtInt, fmt, fmtDate } from '../ui';

export function Knowledge() {
  const [cats, setCats] = useState<any[]>([]);
  useEffect(() => { api.get<any[]>('/dashboard/knowledge').then(setCats); }, []);
  return (
    <div>
      <div className="page-header"><h1>Knowledge Tracker</h1><p>Per-category knowledge: datasets, tokens, evaluation, retention.</p></div>
      <div className="card">
        <table>
          <thead><tr><th>Category</th><th>Datasets</th><th>Tokens</th><th>Documents</th><th>Eval Score</th><th>Retention</th><th>Last Training</th><th>Model Version</th></tr></thead>
          <tbody>{cats.map(c => (
            <tr key={c.category}><td>{c.category}</td><td>{c.datasets}</td><td>{fmtInt(c.tokens)}</td><td>{c.documents}</td>
              <td>{fmt(c.evaluation_score,2)}</td><td>{c.retention != null ? fmt(c.retention,2) : '—'}</td>
              <td>{fmtDate(c.last_training)}</td><td>{c.model_version ? 'v'+c.model_version : '—'}</td></tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  );
}
