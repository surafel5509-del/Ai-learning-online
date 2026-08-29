import { useEffect, useState } from 'react';
import { api } from '../api';
import { StatusBadge, fmtDate } from '../ui';
import { useInterval } from '../ui';

export function Workers() {
  const [ws, setWs] = useState<any[]>([]);
  async function refresh() { setWs(await api.get<any[]>('/dashboard/workers')); }
  useEffect(() => { refresh(); }, []);
  useInterval(refresh, 3000);
  return (
    <div>
      <div className="page-header"><h1>Workers</h1><p>Training worker processes. Real heartbeats and device info.</p></div>
      <div className="card">
        <table>
          <thead><tr><th>ID</th><th>Host</th><th>Device</th><th>Device Name</th><th>Status</th><th>Current Job</th><th>Last Heartbeat</th></tr></thead>
          <tbody>{ws.map(w => (
            <tr key={w.id}><td className="mono text-sm">{w.id.slice(0,16)}</td><td>{w.hostname}</td><td>{w.device}</td><td>{w.device_name}</td>
              <td><StatusBadge status={w.status} /></td><td className="mono text-sm">{w.current_job_id?.slice(0,8)||'—'}</td>
              <td>{fmtDate(w.last_heartbeat)}</td></tr>
          ))}</tbody>
        </table>
        {ws.length === 0 && <div className="muted mt-8">No workers registered. Start a worker: <code>python -m apps.api.worker</code></div>}
      </div>
    </div>
  );
}
