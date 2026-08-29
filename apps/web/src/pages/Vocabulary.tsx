import { useEffect, useState } from 'react';
import { api } from '../api';
import { fmtInt } from '../ui';

export function Vocabulary() {
  const [v, setV] = useState<any>(null);
  useEffect(() => { api.get<any>('/dashboard/vocabulary').then(setV); }, []);
  if (!v) return <div className="muted">Loading…</div>;
  const cov = v.unicode_coverage || {};
  return (
    <div>
      <div className="page-header"><h1>Vocabulary Tracker</h1><p>Tokenizer version, vocabulary size, Unicode coverage.</p></div>
      <div className="grid grid-4 mb-16">
        <div className="stat"><div className="label">Vocabulary Size</div><div className="value">{fmtInt(v.vocab_size)}</div></div>
        <div className="stat"><div className="label">Tokenizer Version</div><div className="value" style={{fontSize:18}}>{v.tokenizer_version || '—'}</div></div>
        <div className="stat"><div className="label">Num Merges</div><div className="value">{fmtInt(v.num_merges)}</div></div>
        <div className="stat"><div className="label">Training Tokens</div><div className="value">{fmtInt(v.training_tokens)}</div></div>
      </div>
      <div className="card">
        <div className="card-title">Unicode Block Coverage (real, from datasets)</div>
        <table>
          <thead><tr><th>Block</th><th>Characters seen</th><th>Share</th></tr></thead>
          <tbody>
            {Object.entries(cov).sort((a:any,b:any)=>b[1]-a[1]).map(([k, n]: any) => {
              const total = Object.values(cov).reduce((s:number,x:any)=>s+x,0) || 1;
              return <tr key={k}><td>{k}</td><td>{fmtInt(n)}</td><td><div className="progress" style={{width:120}}><div className="bar" style={{width:`${(n/total*100)}%`}} /></div></td></tr>;
            })}
          </tbody>
        </table>
        <div className="mt-16 muted text-sm">Ethiopic (Amharic) characters seen: <b>{fmtInt(v.ethiopic_coverage)}</b></div>
      </div>
    </div>
  );
}
