import { useState } from 'react'
import { tokens } from '../App'

function NavItem({ label, isActive, onClick, emoji, badge, dark }) {
  const [hovered, setHovered] = useState(false)
  const t = dark ? tokens.dark : tokens.light

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={label}
      style={{
        width: '100%', border: 'none', borderRadius: '14px',
        padding: '11px 0', marginBottom: '4px',
        background: isActive ? (dark ? '#2D2B4E' : '#EEF2FF') : hovered ? (dark ? '#1e1c35' : '#F5F7FF') : 'transparent',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        cursor: 'pointer', transition: 'all 0.2s', position: 'relative',
      }}
    >
      <span style={{ fontSize: '20px', lineHeight: 1 }}>{emoji}</span>
      {badge > 0 && (
        <div style={{
          position: 'absolute', top: '6px', right: '8px',
          width: '16px', height: '16px', borderRadius: '50%',
          background: '#4F46E5', display: 'flex', alignItems: 'center',
          justifyContent: 'center',
        }}>
          <span style={{ fontSize: '9px', fontWeight: 700, color: 'white' }}>{badge > 9 ? '9+' : badge}</span>
        </div>
      )}
      {isActive && (
        <div style={{
          position: 'absolute', right: '-11px', width: '3px', height: '22px',
          background: '#4F46E5', borderRadius: '99px 0 0 99px',
        }} />
      )}
    </button>
  )
}

function Sidebar({ activePage, onNavigate, dark, reportCount }) {
  const t = dark ? tokens.dark : tokens.light

  const navItems = [
    { id: 'dashboard', emoji: '⊞', label: 'Dashboard' },
    { id: 'analysis', emoji: '🔬', label: 'Analysis' },
    { id: 'models', emoji: '📊', label: 'Model Info' },
    { id: 'reports', emoji: '📄', label: 'Reports', badge: reportCount },
    { id: 'about', emoji: 'ℹ️', label: 'About' },
  ]
  const bottomItems = [
    { id: 'alerts', emoji: '🔔', label: 'Alerts' },
    { id: 'settings', emoji: '⚙️', label: 'Settings' },
  ]

  return (
    <aside style={{
      position: 'fixed', left: 0, top: 0, bottom: 0, width: '72px',
      background: t.sidebar, borderRight: `1px solid ${t.sidebarBorder}`,
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      paddingTop: '20px', paddingBottom: '20px', zIndex: 100,
      boxShadow: dark ? '2px 0 16px rgba(0,0,0,0.4)' : '2px 0 16px rgba(79,70,229,0.08)',
      transition: 'background 0.3s, border-color 0.3s',
    }}>
      <div style={{
        width: '42px', height: '42px',
        background: 'linear-gradient(135deg, #4F46E5, #7C3AED)',
        borderRadius: '13px', display: 'flex', alignItems: 'center',
        justifyContent: 'center', marginBottom: '28px', fontSize: '21px',
        boxShadow: '0 4px 14px rgba(79,70,229,0.4)',
        cursor: 'pointer',
      }} onClick={() => onNavigate('dashboard')}>🧠</div>

      <nav style={{ flex: 1, width: '100%', padding: '0 10px' }}>
        {navItems.map(item => (
          <NavItem
            key={item.id}
            emoji={item.emoji}
            label={item.label}
            isActive={activePage === item.id}
            onClick={() => onNavigate(item.id)}
            badge={item.badge}
            dark={dark}
          />
        ))}
      </nav>

      <div style={{ width: '100%', padding: '0 10px' }}>
        {bottomItems.map(item => (
          <NavItem
            key={item.id}
            emoji={item.emoji}
            label={item.label}
            isActive={activePage === item.id}
            onClick={() => onNavigate(item.id)}
            dark={dark}
          />
        ))}
      </div>
    </aside>
  )
}

export default Sidebar  