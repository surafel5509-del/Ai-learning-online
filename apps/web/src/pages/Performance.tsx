import { useEffect, useState } from 'react';
import { api } from '../api';
import { fmt, fmtInt } from '../ui';

export function Performance() {
  const [hw, setHw] = useState<any>(null);
  const [jobs, setJobs] = useState<any[]>([]);
  useEffect(() => {
    const id = setInterval(async () => {
      setHw(await api.get('/dashboard/hardware'));
    }, 2000);
    api.get<any[]>('/training/jobs').then(js => setJobs(js.filter(j=>j.status==='completed').slice(0,20)));
    return () => clearInterval(id);
  }, []);
  return (
    <div>
      <div className="page-header"><h1>Performance</h1><p>Live hardware metrics and training throughput (real).</p></div>
      <div className="grid grid-4 mb-16">
        <div className="stat"><div className="label">CPU</div><div className="value">{hw?fmt(hw.cpu_percent,0):'—'}%</div><div className="sub">{hw?.cpu_count} cores</div></div>
        <div className="stat"><div className="label">Memory</div><div className="value">{hw?fmt(hw.memory_percent,0):'—'}%</div><div className="sub">{hw?fmt(hw.memory_used_mb,0)+' MB':''}</div></div>
        {hw?.gpu_name && <><div className="stat"><div className="label">GPU Util</div><div className="value">{fmt(hw.gpu_utilization_pct,0)}%</div><div className="sub">{hw.gpu_name}</div></div>
        <div className="stat"><div className="label">VRAM</div><div className="value">{fmt(hw.vram_used_mb,0)} MB</div><div className="sub">/ {fmtInt(hw.vram_total_mb)} MB</div></div></>}
      </div>
      <div className="card">
        <div className="card-title">Completed Jobs Throughput</div>
        <table>
          <thead><tr><th>Job</th><th>Mode</th><th>Device</th><th>Tokens</th><th>Tokens/s</th><th>Duration</th><th>Final Loss</th></tr></thead>
          <tbody>{jobs.map(j=>(<tr key={j.id}><td className="mono text-sm">{j.id.slice(0,8)}</td><td>{j.mode}</td><td>{j.device}</td><td>{fmtInt(j.tokens_processed)}</td><td>{fmt(j.tokens_per_sec,0)}</td><td>{fmt(j.elapsed_seconds,0)}s</td><td className="mono">{fmt(j.final_loss)}</td></tr>))}</tbody>
        </table>
      </div>
    </div>
  );
}
