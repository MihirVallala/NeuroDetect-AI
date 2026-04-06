import { tokens } from '../App'

function Recommendation({ results, dark }) {
  if (!results) return null
  const t = dark ? tokens.dark : tokens.light
  const { risk_score, risk_level, final_label, recommendation, fusion_weights, processing_time_ms } = results

  const cfg =
    risk_score < 35
      ? { color: '#10B981', bg: dark ? '#064e3b40' : '#ECFDF5', border: dark ? '#10B98150' : '#A7F3D0', emoji: '✅', title: 'No Significant Risk Detected' }
      : risk_score < 60
        ? { color: '#F59E0B', bg: dark ? '#78350f40' : '#FFFBEB', border: dark ? '#F59E0B50' : '#FDE68A', emoji: '⚠️', title: 'At-Risk / Early Stage Indicators' }
        : { color: '#EF4444', bg: dark ? '#7f1d1d40' : '#FEF2F2', border: dark ? '#EF444450' : '#FECACA', emoji: '🔴', title: 'Strong PD Indicators Detected' }

  const weightLabels = { speech: '🎤 Speech', gait: '🚶 Gait', handwriting: '✍️ Handwriting' }
  const weightColors = { speech: '#4F46E5', gait: '#10B981', handwriting: '#7C3AED' }

  return (
    <div style={{ marginBottom: '28px', animation: 'slideUp 0.4s ease' }}>
      <h2 style={{ fontSize: '17px', fontWeight: 700, color: t.text, fontFamily: 'Sora, sans-serif', marginBottom: '18px' }}>
        Clinical Recommendation
      </h2>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>

        {/* Recommendation */}
        <div style={{ background: cfg.bg, borderRadius: '20px', border: `1.5px solid ${cfg.border}`, padding: '26px', boxShadow: `0 4px 16px ${cfg.color}15` }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px' }}>
            <div style={{ width: '48px', height: '48px', borderRadius: '14px', background: dark ? `${cfg.color}20` : '#FFFFFF', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '24px', boxShadow: '0 2px 8px rgba(0,0,0,0.06)', flexShrink: 0 }}>
              {cfg.emoji}
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px', flexWrap: 'wrap' }}>
                <h3 style={{ fontSize: '15px', fontWeight: 700, color: cfg.color, fontFamily: 'Sora, sans-serif' }}>{cfg.title}</h3>
                <span style={{ padding: '3px 10px', borderRadius: '99px', background: dark ? `${cfg.color}20` : '#FFFFFF', border: `1.5px solid ${cfg.border}`, fontSize: '11px', fontWeight: 700, color: cfg.color }}>
                  {risk_level}
                </span>
              </div>
              <p style={{ fontSize: '13px', color: t.subtext, lineHeight: 1.7 }}>{recommendation}</p>
            </div>
          </div>
          <div style={{ marginTop: '18px', padding: '12px 16px', borderRadius: '12px', background: dark ? 'rgba(255,255,255,0.05)' : '#FFFFFF', border: `1px solid ${cfg.border}`, display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
            <span style={{ fontSize: '15px', flexShrink: 0, marginTop: '1px' }}>ℹ️</span>
            <p style={{ fontSize: '11px', color: t.muted, lineHeight: 1.6 }}>
              This is an AI-powered screening tool and does not constitute a clinical diagnosis.
              Always consult a qualified neurologist or movement disorder specialist for medical evaluation.
            </p>
          </div>
        </div>

        {/* Fusion Details */}
        <div style={{ background: t.card, borderRadius: '20px', border: `1px solid ${t.border}`, padding: '26px', boxShadow: '0 2px 8px rgba(79,70,229,0.06)' }}>
          <h3 style={{ fontSize: '15px', fontWeight: 700, color: t.text, marginBottom: '18px', fontFamily: 'Sora, sans-serif' }}>Fusion Details</h3>

          <p style={{ fontSize: '12px', fontWeight: 600, color: t.muted, marginBottom: '12px', letterSpacing: '0.2px' }}>⚖️ MODALITY WEIGHTS</p>
          {fusion_weights && Object.entries(fusion_weights).map(([key, val]) => (
            <div key={key} style={{ marginBottom: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ fontSize: '12px', fontWeight: 500, color: t.muted }}>{weightLabels[key]}</span>
                <span style={{ fontSize: '12px', fontWeight: 700, color: weightColors[key] }}>{(val * 100).toFixed(0)}%</span>
              </div>
              <div style={{ height: '7px', background: dark ? '#2D2B4E' : '#F3F4F6', borderRadius: '99px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${val * 100}%`, background: weightColors[key], borderRadius: '99px', transition: 'width 1s ease', boxShadow: `0 1px 4px ${weightColors[key]}40` }} />
              </div>
            </div>
          ))}

          <div style={{ marginTop: '18px', padding: '16px', borderRadius: '14px', background: dark ? 'rgba(255,255,255,0.04)' : '#F8FAFF', border: `1px solid ${t.border}`, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
            <div>
              <p style={{ fontSize: '11px', color: t.muted, marginBottom: '4px', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.3px' }}>Risk Score</p>
              <p style={{ fontSize: '24px', fontWeight: 800, color: cfg.color, fontFamily: 'Sora, sans-serif' }}>{risk_score?.toFixed(1)}</p>
            </div>
            <div>
              <p style={{ fontSize: '11px', color: t.muted, marginBottom: '4px', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.3px' }}>Diagnosis</p>
              <p style={{ fontSize: '12px', fontWeight: 700, color: t.text, lineHeight: 1.4 }}>{final_label}</p>
            </div>
            <div>
              <p style={{ fontSize: '11px', color: t.muted, marginBottom: '4px', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.3px' }}>Processing</p>
              <p style={{ fontSize: '13px', fontWeight: 600, color: t.subtext }}>⏱ {processing_time_ms?.toFixed(0)}ms</p>
            </div>
            <div>
              <p style={{ fontSize: '11px', color: t.muted, marginBottom: '4px', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.3px' }}>Risk Level</p>
              <p style={{ fontSize: '13px', fontWeight: 700, color: cfg.color }}>{risk_level}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Recommendation  