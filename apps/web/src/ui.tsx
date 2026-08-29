// Shared small UI helpers used across pages.

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { color: string; label: string }> = {
    learning: { color: 'green', label: '🟢 LEARNING' },
    running: { color: 'green', label: '🟢 RUNNING' },
    online: { color: 'green', label: '🟢 ONLINE' },
    ready: { color: 'green', label: '🟢 READY' },
    completed: { color: 'green', label: 'COMPLETED' },
    production: { color: 'green', label: 'PRODUCTION' },
    validated: { color: 'blue', label: 'VALIDATED' },
    evaluating: { color: 'blue', label: '🔵 EVALUATING' },
    queued: { color: 'yellow', label: '🟡 QUEUED' },
    paused: { color: 'yellow', label: 'PAUSED' },
    candidate: { color: 'yellow', label: 'CANDIDATE' },
    training: { color: 'blue', label: 'TRAINING' },
    idle: { color: 'grey', label: '⚪ IDLE' },
    archived: { color: 'grey', label: 'ARCHIVED' },
    failed: { color: 'red', label: '🔴 FAILED' },
    cancelled: { color: 'grey', label: 'CANCELLED' },
    error: { color: 'red', label: '🔴 ERROR' },
    dead: { color: 'red', label: 'DEAD' },
    busy: { color: 'green', label: 'BUSY' },
  };
  const m = map[status] || { color: 'grey', label: status.toUpperCase() };
  return <span className={`badge ${m.color}`}>{m.label}</span>;
}

export function fmt(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined || n !== n) return '—';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(digits) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(digits) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(digits) + 'k';
  return n.toFixed(digits);
}

export function fmtInt(n: number | null | undefined): string {
  if (n === null || n === undefined) return '—';
  return Number(n).toLocaleString();
}

export function fmtTime(s: number | null | undefined): string {
  if (!s || s !== s) return '—';
  if (s < 60) return s.toFixed(1) + 's';
  const m = Math.floor(s / 60); const sec = Math.floor(s % 60);
  if (m < 60) return `${m}m ${sec}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function fmtDate(d: string | null | undefined): string {
  if (!d) return '—';
  return new Date(d).toLocaleString();
}

export function useInterval(cb: () => void, ms: number | null) {
  const ref = React.useRef(cb);
  React.useEffect(() => { ref.current = cb; }, [cb]);
  React.useEffect(() => {
    if (ms === null) return;
    const id = setInterval(() => ref.current(), ms);
    return () => clearInterval(id);
  }, [ms]);
}

import React from 'react';
