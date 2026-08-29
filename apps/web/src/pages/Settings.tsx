import { useEffect, useState } from 'react';
import { api } from '../api';
import { fmtDate } from '../ui';

export function Settings() {
  const [sched, setSched] = useState<any>(null);
  const [msg, setMsg] = useState('');

  async function refresh() {
    try { setSched(await api.get<any>('/schedules/active')); } catch { /* */ }
  }
  useEffect(() => { refresh(); }, []);

  async function updateSched(s: any) {
    await api.post('/schedules', { ...s });
    setMsg('✓ Schedule updated'); refresh();
  }

  return (
    <div>
      <div className="page-header"><h1>Settings</h1><p>Automatic learning, scheduled training, system configuration.</p></div>
      <div className="card">
        <div className="card-title">Automatic Learning</div>
        <p className="muted text-sm mb-16">When ON, new datasets → validation → queue → train → replay → evaluate → retention test → candidate → promote if it passes gates.</p>
        <div className="flex items-center gap-16">
          <label style={{display:'flex', alignItems:'center', gap:8, margin:0}}><input type="checkbox" style={{width:'auto'}} checked={sched?.auto_learning||false} onChange={e=>updateSched({...sched, auto_learning:e.target.checked})} /> Auto Learning: {sched?.auto_learning?'ON':'OFF'}</label>
        </div>
      </div>
      <div className="card">
        <div className="card-title">Scheduled Learning</div>
        <div className="form-row"><label>Interval</label>
          <select value={sched?.interval||'off'} onChange={e=>updateSched({...sched, interval:e.target.value})}>
            <option value="off">Off</option><option value="30m">30 minutes</option><option value="1h">1 hour</option><option value="6h">6 hours</option><option value="daily">Daily</option><option value="custom">Custom</option>
          </select>
        </div>
        <div className="form-row"><label>Retention threshold (promotion gate)</label>
          <input type="number" step="0.05" min="0" max="1" value={sched?.retention_threshold??0.7} onChange={e=>updateSched({...sched, retention_threshold:+e.target.value})} />
        </div>
        {sched?.last_run && <div className="muted text-sm mt-8">Last run: {fmtDate(sched.last_run)}</div>}
        {msg && <div className="success mt-8">{msg}</div>}
      </div>
      <div className="card">
        <div className="card-title">AI Limitation Notice</div>
        <p className="text-sm">This is an independent Transformer-based system. A small model trained on limited data cannot become a frontier model. The system is designed to scale through larger datasets, better data, larger models, and more compute. All metrics shown are real; composite scores (Growth Score) are labeled as estimates, not scientific intelligence measurements.</p>
      </div>
    </div>
  );
}
