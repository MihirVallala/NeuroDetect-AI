import { tokens } from '../App'

function StatCard({ label, value, sub, emoji, color, bg, bar, dark }) {
  const t = dark ? tokens.dark : tokens.light
  return (
    <div style={{
      background: t.card, borderRadius: '18px', border: `1px solid ${t.border}`,
      padding: '22px 24px', boxShadow: '0 2px 8px rgba(79,70,229,0.06)',
      transition: 'background 0.3s, border-color 0.3s',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '18px' }}>
        <div>
          <p style={{ fontSize: '13px', color: t.muted, fontWeight: 500, marginBottom: '8px' }}>{label}</p>
          <p style={{ fontSize: '30px', fontWeight: 700, color: t.text, letterSpacing: '-0.5px', lineHeight: 1, fontFamily: 'Sora, sans-serif' }}>
            {value}
          </p>
        </div>
        <div style={{ width: '42px', height: '42px', borderRadius: '12px', background: dark ? `${color}20` : bg, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '19px' }}>
          {emoji}
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <p style={{ fontSize: '12px', color: t.muted, fontWeight: 500 }}>{sub}</p>
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '2px', height: '24px' }}>
          {bar.map((h, i) => (
            <div key={i} style={{ width: '4px', height: `${h * 2.5}px`, background: color, borderRadius: '2px', opacity: 0.4 + (i / bar.length) * 0.6 }} />
          ))}
        </div>
      </div>
    </div>
  )
}

function StatsBar({ dark }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: '20px', marginBottom: '28px' }}>
      <StatCard label="Total Subjects" value="459" sub="Across all 3 datasets" emoji="👥" color="#4F46E5" bg="#EEF2FF" bar={[4, 6, 5, 7, 6, 8, 7, 9, 8, 10]} dark={dark} />
      <StatCard label="Speech Dataset" value="130" sub="HC / RBD / PD · 3-class" emoji="🎤" color="#10B981" bg="#ECFDF5" bar={[3, 5, 4, 6, 5, 7, 6, 7, 6, 8]} dark={dark} />
      <StatCard label="Gait Dataset" value="165" sub="HC / PD · Binary" emoji="🚶" color="#F59E0B" bg="#FFFBEB" bar={[5, 6, 5, 7, 6, 8, 7, 8, 7, 9]} dark={dark} />
      <StatCard label="Handwriting" value="204" sub="HC / PD · Images" emoji="✍️" color="#8B5CF6" bg="#F5F3FF" bar={[4, 5, 6, 7, 6, 8, 7, 9, 8, 10]} dark={dark} />
    </div>
  )
}

export default StatsBar  