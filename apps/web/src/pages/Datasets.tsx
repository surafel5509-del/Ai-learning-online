import { useEffect, useState } from 'react';
import { api } from '../api';
import { fmtInt, fmtDate, StatusBadge } from '../ui';

export function Datasets() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [name, setName] = useState('');
  const [cat, setCat] = useState('General Knowledge');
  const [msg, setMsg] = useState('');

  async function refresh() { setDatasets(await api.get<any[]>('/datasets')); }
  useEffect(() => { refresh(); }, []);

  async function create() {
    try { await api.post('/datasets', { name, knowledge_category: cat }); setName(''); setMsg('✓ Dataset created'); refresh(); }
    catch (e: any) { setMsg('✗ ' + e.message); }
  }

  async function del(id: string) {
    if (!confirm('Delete this dataset and all its versions?')) return;
    await api.del(`/datasets/${id}`); refresh();
  }

  return (
    <div>
      <div className="page-header"><h1>Datasets</h1><p>Training library: datasets, versions, files, tokens.</p></div>
      <div className="card">
        <div className="card-title">New Dataset</div>
        <div className="grid grid-2">
          <div className="form-row"><label>Name</label><input value={name} onChange={e => setName(e.target.value)} /></div>
          <div className="form-row"><label>Category</label>
            <select value={cat} onChange={e => setCat(e.target.value)}>{['General Knowledge','English','Amharic','Grammar','Technical Knowledge','Instructions','Conversation','User-provided Knowledge'].map(c=><option key={c}>{c}</option>)}</select>
          </div>
        </div>
        <button onClick={create} disabled={!name.trim()}>Create Dataset</button>
        {msg && <div className="mt-8">{msg}</div>}
        <p className="muted text-sm mt-8">After creating, use the Training page to paste text into a version (or upload files via API).</p>
      </div>
      <div className="card">
        <div className="card-title">All Datasets</div>
        <table>
          <thead><tr><th>Dataset</th><th>Category</th><th>Version</th><th>Files</th><th>Documents</th><th>Tokens</th><th>Words (est.)</th><th>Created</th><th></th></tr></thead>
          <tbody>
            {datasets.flatMap(d => d.versions.length ? d.versions.map((v: any) => (
              <tr key={v.id}><td>{d.name}</td><td>{d.knowledge_category}</td><td>v{v.version}</td><td>{v.num_files}</td><td>{v.num_documents}</td><td>{fmtInt(v.num_tokens)}</td><td className="estimate">{fmtInt(v.estimated_words)}</td><td>{fmtDate(v.created_at)}</td><td><button className="ghost" style={{padding:'2px 6px',fontSize:11}} onClick={()=>del(d.id)}>🗑</button></td></tr>
            )) : [<tr key={d.id}><td>{d.name}</td><td>{d.knowledge_category}</td><td colSpan={6} className="muted">no versions</td><td><button className="ghost" style={{padding:'2px 6px',fontSize:11}} onClick={()=>del(d.id)}>🗑</button></td></tr>])}
          </tbody>
        </table>
      </div>
    </div>
  );
}
