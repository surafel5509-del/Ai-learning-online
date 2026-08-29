import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { api, setToken, clearToken } from './api';

interface AuthState {
  user: { user_id: string; username: string; is_admin: boolean } | null;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const Ctx = createContext<AuthState>(null!);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthState['user']>(null);

  useEffect(() => {
    const saved = localStorage.getItem('ai_user');
    if (saved) {
      try { setUser(JSON.parse(saved)); } catch { /* ignore */ }
    }
  }, []);

  async function register(username: string, password: string) {
    const r = await api.post<any>('/auth/register', { username, password });
    setToken(r.access_token);
    const u = { user_id: r.user_id, username: r.username, is_admin: r.is_admin };
    setUser(u);
    localStorage.setItem('ai_user', JSON.stringify(u));
  }

  async function login(username: string, password: string) {
    const r = await api.post<any>('/auth/login', { username, password });
    setToken(r.access_token);
    const u = { user_id: r.user_id, username: r.username, is_admin: r.is_admin };
    setUser(u);
    localStorage.setItem('ai_user', JSON.stringify(u));
  }

  function logout() {
    clearToken();
    localStorage.removeItem('ai_user');
    setUser(null);
  }

  return <Ctx.Provider value={{ user, login, register, logout }}>{children}</Ctx.Provider>;
}

export function useAuth() {
  return useContext(Ctx);
}
