import { useState } from 'react';
import { useAuth } from '../auth';

export function Login() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [err, setErr] = useState('');
  const [loading, setLoading] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(''); setLoading(true);
    try {
      if (mode === 'login') await login(username, password);
      else await register(username, password);
    } catch (e: any) {
      setErr(e.message);
    } finally { setLoading(false); }
  }

  return (
    <div style={{ display: 'flex', height: '100vh', alignItems: 'center', justifyContent: 'center', background: 'var(--bg)' }}>
      <div className="card" style={{ width: 380 }}>
        <div style={{ textAlign: 'center', marginBottom: 24 }}>
          <div style={{ fontSize: 40 }}>🧠</div>
          <h2 style={{ marginTop: 8 }}>AI Continual-Learning Platform</h2>
          <p className="muted text-sm" style={{ marginTop: 4 }}>Real trainable Transformer LM</p>
        </div>
        <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
          <button className={mode === 'login' ? '' : 'secondary'} style={{ flex: 1 }} onClick={() => setMode('login')}>Login</button>
          <button className={mode === 'register' ? '' : 'secondary'} style={{ flex: 1 }} onClick={() => setMode('register')}>Register</button>
        </div>
        <form onSubmit={submit}>
          <div className="form-row">
            <label>Username</label>
            <input value={username} onChange={e => setUsername(e.target.value)} required minLength={3} />
          </div>
          <div className="form-row">
            <label>Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} required minLength={6} />
          </div>
          {err && <div className="error mb-8">{err}</div>}
          <button type="submit" disabled={loading} className="w-full" style={{ marginTop: 8 }}>
            {loading ? '…' : mode === 'login' ? 'Login' : 'Create account'}
          </button>
        </form>
      </div>
    </div>
  );
}
