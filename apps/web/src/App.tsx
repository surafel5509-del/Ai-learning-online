import { Routes, Route, Navigate, NavLink, useLocation } from 'react-router-dom';
import { useAuth } from './auth';
import { Login } from './pages/Login';
import { Dashboard } from './pages/Dashboard';
import { AIGrowth } from './pages/AIGrowth';
import { Training } from './pages/Training';
import { Datasets } from './pages/Datasets';
import { TrainingQueue } from './pages/TrainingQueue';
import { Models } from './pages/Models';
import { TestLab } from './pages/TestLab';
import { Knowledge } from './pages/Knowledge';
import { Vocabulary } from './pages/Vocabulary';
import { Memory } from './pages/Memory';
import { Evaluations } from './pages/Evaluations';
import { Checkpoints } from './pages/Checkpoints';
import { Workers } from './pages/Workers';
import { Performance } from './pages/Performance';
import { Settings } from './pages/Settings';
import { Chat } from './pages/Chat';

const navGroups = [
  { group: 'Overview', items: [
    { to: '/', label: 'Dashboard', icon: '📊' },
    { to: '/growth', label: 'AI Growth', icon: '🌱' },
  ]},
  { group: 'Learning', items: [
    { to: '/chat', label: 'Chat', icon: '💬' },
    { to: '/test-lab', label: 'Model Test Lab', icon: '🧪' },
    { to: '/training', label: 'Training', icon: '🎯' },
    { to: '/queue', label: 'Training Queue', icon: '📋' },
    { to: '/datasets', label: 'Datasets', icon: '📚' },
  ]},
  { group: 'Knowledge', items: [
    { to: '/knowledge', label: 'Knowledge', icon: '🧠' },
    { to: '/vocabulary', label: 'Vocabulary', icon: '🔤' },
    { to: '/memory', label: 'Memory', icon: '💾' },
    { to: '/evaluations', label: 'Evaluations', icon: '✅' },
  ]},
  { group: 'System', items: [
    { to: '/models', label: 'Models', icon: '📦' },
    { to: '/checkpoints', label: 'Checkpoints', icon: '🔖' },
    { to: '/workers', label: 'Workers', icon: '⚙️' },
    { to: '/performance', label: 'Performance', icon: '⚡' },
    { to: '/settings', label: 'Settings', icon: '🔧' },
  ]},
];

function Sidebar() {
  const { user, logout } = useAuth();
  return (
    <aside className="sidebar">
      <div className="logo">🧠 AI Platform</div>
      <nav className="nav">
        {navGroups.map(g => (
          <div key={g.group}>
            <div className="group">{g.group}</div>
            {g.items.map(it => (
              <NavLink key={it.to} to={it.to} end={it.to === '/'}
                className={({isActive}) => isActive ? 'active' : ''}>
                <span>{it.icon}</span> {it.label}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>
      <div className="user-box">
        <div>{user?.username}</div>
        <button className="ghost" style={{marginTop:6, padding:'4px 8px', fontSize:11}} onClick={logout}>Logout</button>
      </div>
    </aside>
  );
}

export default function App() {
  const { user } = useAuth();
  const loc = useLocation();
  if (!user) return <Login />;
  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/growth" element={<AIGrowth />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/test-lab" element={<TestLab />} />
          <Route path="/training" element={<Training />} />
          <Route path="/queue" element={<TrainingQueue />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/knowledge" element={<Knowledge />} />
          <Route path="/vocabulary" element={<Vocabulary />} />
          <Route path="/memory" element={<Memory />} />
          <Route path="/evaluations" element={<Evaluations />} />
          <Route path="/models" element={<Models />} />
          <Route path="/checkpoints" element={<Checkpoints />} />
          <Route path="/workers" element={<Workers />} />
          <Route path="/performance" element={<Performance />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </main>
    </div>
  );
}
