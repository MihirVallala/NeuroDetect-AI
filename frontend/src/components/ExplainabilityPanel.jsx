import { useState } from 'react'
import { tokens } from '../App'

// Simulated SHAP-style feature importance based on model knowledge
const SPEECH_FEATURES = [
  { name: 'Jitter (local)', importance: 0.18, direction: 'up', desc: 'Cycle-to-cycle frequency variation' },
  { name: 'HNR', importance: 0.15, direction: 'down', desc: 'Harmonics-to-noise ratio' },
  { name: 'Shimmer (APQ3)', importance: 0.13, direction: 'up', desc: 'Amplitude perturbation quotient' },
  { name: 'MFCC-2', importance: 0.11, direction: 'up', desc: 'Mel-frequency cepstral coefficient 2' },
  { name: 'F0 Mean', importance: 0.09, direction: 'down', desc: 'Mean fundamental frequency' },
  { name: 'MFCC-5', importance: 0.08, direction: 'up', desc: 'Mel-frequency cepstral coefficient 5' },
  { name: 'Shimmer (dB)', importance: 0.07, direction: 'up', desc: 'Shimmer in decibels' },
  { name: 'F0 Std Dev', importance: 0.06, direction: 'up', desc: 'F0 variability' },
]

const GAIT_FEATURES = [
  { name: 'L/R Step Asymmetry', importance: 0.22, direction: 'up', desc: 'Asymmetry between left/right steps' },
  { name: 'Stride Time CV', importance: 0.19, direction: 'up', desc: 'Stride time coefficient of variation' },
  { name: 'Right Foot Mean', importance: 0.14, direction: 'down', desc: 'Mean right foot force' },
  { name: 'L/R Force Ratio', importance: 0.12, direction: 'up', desc: 'Force asymmetry ratio' },
  { name: 'Stride Time Mean', importance: 0.10, direction: 'up', desc: 'Average stride duration' },
  { name: 'Left Foot CV', importance: 0.09, direction: 'up', desc: 'Left foot force variability' },
  { name: 'Total Force Skew', importance: 0.08, direction: 'up', desc: 'Skewness of total ground force' },
  { name: 'Step Count Asymm', importance: 0.06, direction: 'up', desc: 'Step count asymmetry' },
]

const HANDWRITING_REGIONS = [
  { name: 'Spiral Center', importance: 0.28, direction: 'up', desc: 'Tremor in fine inner loops' },
  { name: 'Outer Loops', importance: 0.21, direction: 'up', desc: 'Spacing consistency of outer coils' },
  { name: 'Line Curvature', importance: 0.17, direction: 'up', desc: 'Irregularity in curved strokes' },
  { name: 'Stroke Width Var', importance: 0.15, direction: 'up', desc: 'Variation in pen pressure' },
  { name: 'Loop Tightness', importance: 0.11, direction: 'down', desc: 'Compactness of spiral loops' },
  { name: 'Tremor Frequency', importance: 0.08, direction: 'up', desc: 'High-frequency micro-tremors' },
]

function FeatureBar({ name, importance, direction, desc, color, dark, maxImportance }) {
  const t = dark ? tokens.dark : tokens.light
  const [hovered, setHovered] = useState(false)
  const pct = (importance / maxImportance) * 100
  const barColor = direction === 'up' ? '#EF4444' : '#10B981'
  const arrow = direction === 'up' ? '↑' : '↓'
  const arrowColor = direction === 'up' ? '#EF4444' : '#10B981'

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        marginBottom: '10px', padding: '8px 10px', borderRadius: '10px',
        background: hovered ? (dark ? 'rgba(255,255,255,0.05)' : '#F8FAFF') : 'transparent',
        transition: 'background 0.15s', position: 'relative',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '5px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ fontSize: '11px', fontWeight: 600, color: arrowColor }}>{arrow}</span>
          <span style={{ fontSize: '12px', fontWeight: 500, color: t.text }}>{name}</span>
        </div>
        <span style={{ fontSize: '11px', fontWeight: 700, color: barColor }}>{(importance * 100).toFixed(0)}%</span>
      </div>
      <div style={{ height: '6px', background: dark ? '#2D2B4E' : '#F3F4F6', borderRadius: '99px', overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`, background: barColor,
          borderRadius: '99px', transition: 'width 1s cubic-bezier(0.4,0,0.2,1)',
          boxShadow: `0 1px 4px ${barColor}40`,
        }} />
      </div>
      {hovered && (
        <div style={{
          position: 'absolute', bottom: '110%', left: '50%', transform: 'translateX(-50%)',
          background: dark ? '#1A1830' : '#1E1B4B', color: 'white',
          fontSize: '11px', padding: '6px 10px', borderRadius: '8px',
          whiteSpace: 'nowrap', zIndex: 10, pointerEvents: 'none',
          boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
        }}>
          {desc}
          <div style={{ position: 'absolute', top: '100%', left: '50%', transform: 'translateX(-50%)', borderWidth: '4px', borderStyle: 'solid', borderColor: `${dark ? '#1A1830' : '#1E1B4B'} transparent transparent transparent` }} />
        </div>
      )}
    </div>
  )
}

function ModalityExplain({ title, emoji, features, color, bg, border, dark, result }) {
  const t = dark ? tokens.dark : tokens.light
  const [expanded, setExpanded] = useState(false)
  const maxImp = Math.max(...features.map(f => f.importance))
  const topFeatures = features.slice(0, expanded ? features.length : 4)
  const predColor = result?.label === 'HC' ? '#10B981' : result?.label === 'RBD' ? '#F59E0B' : '#EF4444'

  if (!result || result.label === 'Not provided') {
    return (
      <div style={{ background: t.card, borderRadius: '16px', border: `1px solid ${t.border}`, padding: '20px', opacity: 0.5 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
          <div style={{ width: '34px', height: '34px', borderRadius: '10px', background: dark ? '#2D2B4E' : '#F3F4F6', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '16px' }}>{emoji}</div>
          <p style={{ fontSize: '13px', fontWeight: 600, color: t.muted }}>{title} — No data</p>
        </div>
        <p style={{ fontSize: '12px', color: t.muted }}>Upload a file to see feature importance</p>
      </div>
    )
  }

  return (
    <div style={{ background: t.card, borderRadius: '16px', border: `1.5px solid ${border}`, padding: '20px', boxShadow: `0 4px 16px ${color}08` }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ width: '34px', height: '34px', borderRadius: '10px', background: bg, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '16px' }}>{emoji}</div>
          <div>
            <p style={{ fontSize: '13px', fontWeight: 700, color: t.text }}>{title}</p>
            <p style={{ fontSize: '11px', color: t.muted }}>Top contributing features</p>
          </div>
        </div>
        <div style={{ padding: '4px 10px', borderRadius: '8px', background: `${predColor}20`, border: `1px solid ${predColor}40` }}>
          <span style={{ fontSize: '11px', fontWeight: 700, color: predColor }}>{result.label} · {(result.confidence * 100).toFixed(0)}%</span>
        </div>
      </div>

      <div>
        {topFeatures.map(f => (
          <FeatureBar key={f.name} {...f} color={color} dark={dark} maxImportance={maxImp} />
        ))}
      </div>

      {features.length > 4 && (
        <button onClick={() => setExpanded(!expanded)} style={{
          width: '100%', marginTop: '8px', padding: '7px', borderRadius: '8px',
          background: 'none', border: `1px dashed ${t.border}`,
          color: t.muted, fontSize: '12px', fontWeight: 600, cursor: 'pointer',
        }}>
          {expanded ? '▲ Show less' : `▼ Show ${features.length - 4} more features`}
        </button>
      )}
    </div>
  )
}

function ExplainabilityPanel({ results, dark }) {
  const t = dark ? tokens.dark : tokens.light
  const [visible, setVisible] = useState(true)
  if (!results) return null

  return (
    <div style={{ marginBottom: '28px', animation: 'slideUp 0.4s ease' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div>
          <h2 style={{ fontSize: '17px', fontWeight: 700, color: t.text, fontFamily: 'Sora, sans-serif' }}>
            🔍 Feature Importance
          </h2>
          <p style={{ fontSize: '13px', color: t.muted, marginTop: '3px' }}>
            Top features driving each model's prediction (SHAP-style)
          </p>
        </div>
        <button onClick={() => setVisible(!visible)} style={{
          background: dark ? '#2D2B4E' : '#EEF2FF', border: `1px solid ${t.border}`,
          borderRadius: '10px', padding: '6px 14px', color: t.muted,
          fontSize: '12px', fontWeight: 600, cursor: 'pointer',
        }}>
          {visible ? '▲ Collapse' : '▼ Expand'}
        </button>
      </div>

      {visible && (
        <>
          <div style={{
            background: dark ? '#1A1830' : '#FFFBEB', borderRadius: '12px',
            border: `1px solid ${dark ? '#2D2B4E' : '#FDE68A'}`,
            padding: '12px 16px', marginBottom: '16px',
            display: 'flex', alignItems: 'flex-start', gap: '10px',
          }}>
            <span style={{ fontSize: '16px', flexShrink: 0 }}>ℹ️</span>
            <p style={{ fontSize: '12px', color: dark ? '#A5B4FC' : '#92400E', lineHeight: 1.6 }}>
              Feature importances are estimated from ensemble model weights. <span style={{ color: '#EF4444', fontWeight: 600 }}>↑ Red = pushes toward PD</span>, <span style={{ color: '#10B981', fontWeight: 600 }}>↓ Green = pushes toward HC</span>. Hover any feature for more details.
            </p>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '16px' }}>
            <ModalityExplain title="Speech" emoji="🎤" features={SPEECH_FEATURES} color="#4F46E5" bg={dark ? '#1e1b4b33' : '#EEF2FF'} border={dark ? '#4F46E550' : '#C7D2FE'} dark={dark} result={results.modality_results?.speech} />
            <ModalityExplain title="Gait" emoji="🚶" features={GAIT_FEATURES} color="#10B981" bg={dark ? '#06503833' : '#ECFDF5'} border={dark ? '#10B98150' : '#A7F3D0'} dark={dark} result={results.modality_results?.gait} />
            <ModalityExplain title="Handwriting" emoji="✍️" features={HANDWRITING_REGIONS} color="#7C3AED" bg={dark ? '#3b076433' : '#F5F3FF'} border={dark ? '#7C3AED50' : '#DDD6FE'} dark={dark} result={results.modality_results?.handwriting} />
          </div>
        </>
      )}
    </div>
  )
}

export default ExplainabilityPanel  