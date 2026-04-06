import { useEffect, useState } from 'react'
import { tokens } from '../App'

function getConfig(score) {
  if (score === null) return { color: '#9CA3AF', light: '#F9FAFB', border: '#E5E7EB', label: 'Awaiting Analysis', emoji: '⏳' }
  if (score < 35) return { color: '#10B981', light: '#ECFDF5', border: '#A7F3D0', label: 'Low Risk', emoji: '✅' }
  if (score < 60) return { color: '#F59E0B', light: '#FFFBEB', border: '#FDE68A', label: 'Moderate Risk', emoji: '⚠️' }
  return { color: '#EF4444', light: '#FEF2F2', border: '#FECACA', label: 'High Risk', emoji: '🔴' }
}

function RiskGauge({ score, label, dark }) {
  const [animated, setAnimated] = useState(0)
  const t = dark ? tokens.dark : tokens.light

  useEffect(() => {
    if (score === null) { setAnimated(0); return }
    const timer = setTimeout(() => setAnimated(score), 200)
    return () => clearTimeout(timer)
  }, [score])

  const cfg = getConfig(score === null ? null : animated)
  const size = 200, cx = 100, cy = 115, r = 78
  const startDeg = -210, endDeg = 30, totalArc = endDeg - startDeg
  const fillDeg = startDeg + (animated / 100) * totalArc
  const toRad = d => (d * Math.PI) / 180

  const arcPath = (s, e) => {
    const x1 = cx + r * Math.cos(toRad(s)), y1 = cy + r * Math.sin(toRad(s))
    const x2 = cx + r * Math.cos(toRad(e)), y2 = cy + r * Math.sin(toRad(e))
    return `M ${x1} ${y1} A ${r} ${r} 0 ${(e - s > 180) ? 1 : 0} 1 ${x2} ${y2}`
  }

  return (
    <div style={{
      background: t.card, borderRadius: '20px', border: `1px solid ${t.border}`,
      padding: '28px 24px', boxShadow: '0 2px 8px rgba(79,70,229,0.06)', textAlign: 'center',
      transition: 'background 0.3s, border-color 0.3s',
    }}>
      <h3 style={{ fontSize: '16px', fontWeight: 700, color: t.text, marginBottom: '4px', fontFamily: 'Sora, sans-serif' }}>
        Fusion Risk Score
      </h3>
      <p style={{ fontSize: '12px', color: t.muted, marginBottom: '20px' }}>Weighted multimodal assessment</p>

      <div style={{ position: 'relative', display: 'inline-block' }}>
        <svg width={size} height={size * 0.72} viewBox={`0 0 ${size} ${size}`}>
          <path d={arcPath(startDeg, endDeg)} fill="none" stroke={dark ? '#2D2B4E' : '#EEF2FF'} strokeWidth="16" strokeLinecap="round" />
          <path d={arcPath(startDeg, startDeg + totalArc * 0.35)} fill="none" stroke={dark ? '#064e3b60' : '#D1FAE5'} strokeWidth="16" strokeLinecap="round" />
          <path d={arcPath(startDeg + totalArc * 0.35, startDeg + totalArc * 0.6)} fill="none" stroke={dark ? '#78350f60' : '#FEF3C7'} strokeWidth="16" strokeLinecap="round" />
          <path d={arcPath(startDeg + totalArc * 0.6, endDeg)} fill="none" stroke={dark ? '#7f1d1d60' : '#FEE2E2'} strokeWidth="16" strokeLinecap="round" />
          {animated > 0 && (
            <path d={arcPath(startDeg, fillDeg)} fill="none" stroke={cfg.color}
              strokeWidth="16" strokeLinecap="round"
              style={{ transition: 'all 1.2s cubic-bezier(0.4,0,0.2,1)', filter: `drop-shadow(0 0 6px ${cfg.color}60)` }}
            />
          )}
        </svg>

        <div style={{ position: 'absolute', bottom: '0px', left: '50%', transform: 'translateX(-50%)', textAlign: 'center', width: '120px' }}>
          <div style={{ fontSize: '46px', fontWeight: 800, color: score === null ? (dark ? '#374151' : '#D1D5DB') : cfg.color, lineHeight: 1, transition: 'color 0.6s', fontFamily: 'Sora, sans-serif' }}>
            {score === null ? '--' : Math.round(animated)}
          </div>
          <div style={{ fontSize: '12px', color: t.muted, marginTop: '3px' }}>out of 100</div>
        </div>
      </div>

      <div style={{
        display: 'inline-flex', alignItems: 'center', gap: '7px',
        padding: '9px 20px', borderRadius: '99px',
        background: dark ? `${cfg.color}20` : cfg.light,
        border: `1.5px solid ${dark ? `${cfg.color}40` : cfg.border}`,
        marginTop: '18px', transition: 'all 0.5s',
      }}>
        <span style={{ fontSize: '15px' }}>{cfg.emoji}</span>
        <span style={{ fontSize: '13px', fontWeight: 600, color: cfg.color }}>{cfg.label}</span>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '16px', padding: '0 4px' }}>
        {[['Healthy', '#10B981'], ['At Risk', '#F59E0B'], ['High Risk', '#EF4444']].map(([l, c]) => (
          <span key={l} style={{ fontSize: '10px', fontWeight: 700, color: c, textTransform: 'uppercase', letterSpacing: '0.4px' }}>{l}</span>
        ))}
      </div>

      {label && (
        <div style={{ marginTop: '16px', padding: '11px 16px', borderRadius: '12px', background: dark ? `${cfg.color}20` : cfg.light, border: `1px solid ${dark ? `${cfg.color}40` : cfg.border}` }}>
          <p style={{ fontSize: '13px', fontWeight: 600, color: cfg.color }}>{label}</p>
        </div>
      )}
    </div>
  )
}

export default RiskGauge  