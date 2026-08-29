import { useEffect, useState } from 'react';
import { api } from '../api';

export function Memory() {
  const [docs, setDocs] = useState<any[]>([]);
  const [mems, setMems] = useState<any[]>([]);
  const [corr, setCorr] = useState<any[]>([]);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [cat, setCat] = useState('General Knowledge');
  const [q, setQ] = useState('');
  const [retrieval, setRetrieval] = useState<any>(null);
  const [corrections, setCorrections] = useState({ question: '', incorrect: '', correct: '' });

  async function refresh() {
    setDocs(await api.get<any[]>('/memory/documents'));
    setMems(await api.get<any[]>('/memory/memories'));
    setCorr(await api.get<any[]>('/memory/corrections'));
  }
  useEffect(() => { refresh(); }, []);

  async function addDoc() {
    await api.post('/memory/documents', { title, content, knowledge_category: cat });
    setTitle(''); setContent(''); refresh();
  }
  async function retrieve() {
    setRetrieval(await api.post<any>('/memory/retrieve', { query: q, top_k: 5 }));
  }
  async function addCorrection() {
    await api.post('/memory/corrections', corrections);
    setCorrections({ question: '', incorrect: '', correct: '' }); refresh();
  }

  return (
    <div>
      <div className="page-header"><h1>Memory & Knowledge</h1><p>Explicit memory, vector documents (RAG), corrections.</p></div>
      <div className="grid grid-2">
        <div className="card">
          <div className="card-title">Add Knowledge Document</div>
          <div className="form-row"><label>Title</label><input value={title} onChange={e=>setTitle(e.target.value)} /></div>
          <div className="form-row"><label>Content</label><textarea value={content} onChange={e=>setContent(e.target.value)} rows={5} /></div>
          <div className="form-row"><label>Category</label>
            <select value={cat} onChange={e=>setCat(e.target.value)}>{['General Knowledge','English','Amharic','Grammar','Technical Knowledge','Instructions'].map(c=><option key={c}>{c}</option>)}</select>
          </div>
          <button onClick={addDoc} disabled={!content.trim()}>Add Document</button>
        </div>
        <div className="card">
          <div className="card-title">RAG Retrieval Test</div>
          <div className="form-row"><label>Query</label><input value={q} onChange={e=>setQ(e.target.value)} placeholder="Ask a question to retrieve context" /></div>
          <button onClick={retrieve} disabled={!q.trim()}>Retrieve</button>
          {retrieval && (
            <div className="mt-16">
              <div className="label mb-8">Retrieved context</div>
              <div className="chat-box" style={{height:'auto', minHeight:60, marginBottom:12}}>{retrieval.context || '(no results)'}</div>
              <div className="label mb-8">Hits</div>
              {retrieval.hits.map((h:any,i:number)=>(
                <div key={i} className="text-sm mb-8" style={{padding:8, background:'var(--bg-2)', borderRadius:6}}>
                  <div className="muted">score {h.score.toFixed(3)} · {h.title||'doc'}</div>
                  {h.text.slice(0,150)}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className="card">
        <div className="card-title">Submit Correction</div>
        <p className="muted text-sm mb-16">When the AI is wrong, submit a correction. Stored as structured data; does NOT modify weights immediately.</p>
        <div className="grid grid-3">
          <div className="form-row"><label>Question</label><input value={corrections.question} onChange={e=>setCorrections({...corrections, question:e.target.value})} /></div>
          <div className="form-row"><label>Incorrect answer</label><input value={corrections.incorrect} onChange={e=>setCorrections({...corrections, incorrect:e.target.value})} /></div>
          <div className="form-row"><label>Correct answer</label><input value={corrections.correct} onChange={e=>setCorrections({...corrections, correct:e.target.value})} /></div>
        </div>
        <button onClick={addCorrection} disabled={!corrections.question.trim()||!corrections.correct.trim()}>Submit Correction</button>
      </div>
      <div className="grid grid-2">
        <div className="card"><div className="card-title">Documents ({docs.length})</div>
          <table><thead><tr><th>Title</th><th>Category</th><th>Chunks</th></tr></thead>
            <tbody>{docs.map(d=><tr key={d.id}><td>{d.title||'—'}</td><td>{d.knowledge_category}</td><td>{d.num_chunks}</td></tr>)}</tbody>
          </table>
        </div>
        <div className="card"><div className="card-title">Corrections ({corr.length})</div>
          <table><thead><tr><th>Question</th><th>Correct</th><th>Status</th></tr></thead>
            <tbody>{corr.map(c=><tr key={c.id}><td>{c.question}</td><td>{c.correct_answer}</td><td>{c.status}</td></tr>)}</tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
