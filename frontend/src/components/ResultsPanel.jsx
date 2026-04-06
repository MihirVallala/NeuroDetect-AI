import { tokens } from '../App'

function ProbBar({ label, value, color, dark }) {
  const t = dark ? tokens.dark : tokens.light
  return (
    <div style={{ marginBottom: '10px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
        <span style={{ fontSize: '12px', fontWeight: 500, color: t.muted }}>{label}</span>
        <span style={{ fontSize: '12px', fontWeight: 700, color }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: '7px', background: dark ? '#2D2B4E' : '#F3F4F6', borderRadius: '99px', overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${value * 100}%`, background: color,
          borderRadius: '99px', transition: 'width 0.9s cubic-bezier(0.4,0,0.2,1)',
        }} />
      </div>
    </div>
  )
}

function ModalityCard({ title, emoji, result, color, bg, border, dark }) {
  const t = dark ? tokens.dark : tokens.light

  if (!result || result.label === 'Not provided') {
    return (
      <div style={{ background: t.card, borderRadius: '18px', border: `1px solid ${t.border}`, padding: '22px', boxShadow: '0 2px 8px rgba(79,70,229,0.04)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
          <div style={{ width: '38px', height: '38px', borderRadius: '11px', background: dark ? '#2D2B4E' : '#F3F4F6', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '17px' }}>{emoji}</div>
          <div>
            <p style={{ fontSize: '13px', fontWeight: 600, color: t.muted }}>{title}</p>
            <p style={{ fontSize: '11px', color: dark ? '#374151' : '#D1D5DB' }}>No file uploaded</p>
          </div>
        </div>
        <div style={{ height: '76px', borderRadius: '12px', background: dark ? '#131226' : '#F9FAFB', border: `1.5px dashed ${t.border}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <p style={{ fontSize: '12px', color: dark ? '#374151' : '#D1D5DB' }}>Upload file to see results</p>
        </div>
      </div>
    )
  }

  const pred = result.label || ''
  const conf = result.confidence || 0
  const probs = result.probabilities || {}
  const ms = result.processing_time_ms
  const probColors = { HC: '#10B981', RBD: '#F59E0B', PD: '#EF4444' }
  const predColor =
    pred.includes('HC') || pred === 'HC' ? '#10B981' :
    pred.includes('RBD') || pred === 'RBD' ? '#F59E0B' : '#EF4444'

  return (
    <div style={{
      borderRadius: '18px', border: `1.5px solid ${border}`,
      padding: '22px', background: dark ? t.card : bg,
      boxShadow: `0 4px 16px ${color}15`,
      animation: 'slideUp 0.4s ease',
      transition: 'background 0.3s',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: '38px', height: '38px', borderRadius: '11px', background: dark ? `${color}20` : '#FFFFFF', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '17px', boxShadow: '0 2px 6px rgba(0,0,0,0.06)' }}>{emoji}</div>
          <div>
            <p style={{ fontSize: '13px', fontWeight: 600, color: t.text }}>{title}</p>
            {ms && <p style={{ fontSize: '11px', color: t.muted }}>⏱ {ms.toFixed(0)}ms</p>}
          </div>
        </div>
        <div style={{ padding: '5px 12px', borderRadius: '99px', background: dark ? `${predColor}20` : '#FFFFFF', border: `1.5px solid ${predColor}`, boxShadow: `0 2px 8px ${predColor}20` }}>
          <span style={{ fontSize: '12px', fontWeight: 700, color: predColor }}>{pred}</span>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px', padding: '12px 14px', borderRadius: '12px', background: dark ? `${color}15` : '#FFFFFF', border: `1px solid ${dark ? `${color}30` : '#E0E7FF'}` }}>
        <span style={{ fontSize: '12px', color: t.muted, fontWeight: 500 }}>Confidence</span>
        <span style={{ fontSize: '22px', fontWeight: 800, color: predColor, fontFamily: 'Sora, sans-serif' }}>
          {(conf * 100).toFixed(1)}%
        </span>
      </div>

      {Object.entries(probs).map(([key, val]) => (
        <ProbBar key={key} label={key} value={val} color={probColors[key] || color} dark={dark} />
      ))}
    </div>
  )
}

function ResultsPanel({ results, dark }) {
  const modality = results?.modality_results || {}
  const t = dark ? tokens.dark : tokens.light

  return (
    <div style={{ marginBottom: '0' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
        <div>
          <h2 style={{ fontSize: '17px', fontWeight: 700, color: t.text, fontFamily: 'Sora, sans-serif' }}>
            Modality Results
          </h2>
          <p style={{ fontSize: '13px', color: t.muted, marginTop: '3px' }}>
            {results ? 'Individual model predictions per modality' : 'Upload files and run analysis to see results'}
          </p>
        </div>
        {results && (
          <span style={{ padding: '4px 14px', borderRadius: '99px', fontSize: '12px', fontWeight: 600, background: '#ECFDF5', color: '#10B981', border: '1px solid #A7F3D0' }}>
            ✅ Analysis Complete
          </span>
        )}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '18px' }}>
        <ModalityCard title="Speech Analysis" emoji="🎤" result={modality.speech} color="#4F46E5" bg="#EEF2FF" border={dark ? '#4F46E550' : '#C7D2FE'} dark={dark} />
        <ModalityCard title="Gait Analysis" emoji="🚶" result={modality.gait} color="#10B981" bg="#ECFDF5" border={dark ? '#10B98150' : '#A7F3D0'} dark={dark} />
        <ModalityCard title="Handwriting" emoji="✍️" result={modality.handwriting} color="#7C3AED" bg="#F5F3FF" border={dark ? '#7C3AED50' : '#DDD6FE'} dark={dark} />
      </div>
    </div>
  )
}

export default ResultsPanel  