import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './App.css'
import Sidebar from './components/Sidebar'
import StatsBar from './components/StatsBar'
import UploadSection from './components/UploadSection'
import RiskGauge from './components/RiskGauge'
import ResultsPanel from './components/ResultsPanel'
import Recommendation from './components/Recommendation'
import ExplainabilityPanel from './components/ExplainabilityPanel'
import generatePDF from './components/PDFReport'

const API_URL = 'http://localhost:8000'

// ── Theme Context ────────────────────────────────────────
export function useTheme() {
  const [dark, setDark] = useState(() => {
    try { return window.__neuroTheme === 'dark' } catch { return false }
  })
  const toggle = () => {
    const next = !dark
    window.__neuroTheme = next ? 'dark' : 'light'
    setDark(next)
  }
  return { dark, toggle }
}

// ── Report Store (in-memory) ─────────────────────────────
const reportStore = { reports: [] }

function saveReport(results, files) {
  const report = {
    id: Date.now(),
    timestamp: new Date().toISOString(),
    results,
    filesUploaded: Object.entries(files)
      .filter(([, v]) => v)
      .map(([k]) => k),
  }
  reportStore.reports.unshift(report)
  if (reportStore.reports.length > 50) reportStore.reports.pop()
  return report
}

// ── Dashboard Footer ─────────────────────────────────────
function DashboardFooter({ dark }) {
  const t = dark ? tokens.dark : tokens.light
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '20px 0 8px', borderTop: `1px solid ${t.border}`, marginTop: '16px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <div style={{
          width: '28px', height: '28px',
          background: 'linear-gradient(135deg, #4F46E5, #7C3AED)',
          borderRadius: '8px', display: 'flex', alignItems: 'center',
          justifyContent: 'center', fontSize: '14px',
        }}>🧠</div>
        <span style={{ fontSize: '13px', fontWeight: 600, color: t.text }}>NeuroDetect AI</span>
        <span style={{ fontSize: '12px', color: t.muted }}>v2.0 · Final Year Project 2026</span>
      </div>
      <p style={{ fontSize: '12px', color: t.muted }}>
        For research purposes only · Not a clinical diagnostic tool
      </p>
    </div>
  )
}

// ── Download Button ──────────────────────────────────────
function DownloadButton({ results }) {
  const [downloading, setDownloading] = useState(false)
  const handleDownload = () => {
    setDownloading(true)
    setTimeout(() => { generatePDF(results); setDownloading(false) }, 400)
  }
  return (
    <button onClick={handleDownload} disabled={downloading} style={{
      display: 'flex', alignItems: 'center', gap: '10px',
      padding: '14px 28px', borderRadius: '14px', border: 'none',
      background: downloading ? '#E0E7FF' : 'linear-gradient(135deg, #10B981, #059669)',
      color: downloading ? '#6B7280' : 'white',
      fontSize: '14px', fontWeight: 600,
      cursor: downloading ? 'not-allowed' : 'pointer',
      boxShadow: downloading ? 'none' : '0 6px 20px rgba(16,185,129,0.4)',
      transition: 'all 0.3s', fontFamily: 'DM Sans, sans-serif',
      width: '100%', justifyContent: 'center',
    }}>
      {downloading ? (
        <><span style={{ display: 'inline-block', width: '16px', height: '16px', border: '2px solid #9CA3AF', borderTopColor: '#4F46E5', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />Generating PDF...</>
      ) : (<>📄 Download Report (PDF)</>)}
    </button>
  )
}

// ── Demo Banner ──────────────────────────────────────────
function DemoBanner({ onDismiss }) {
  return (
    <div style={{
      background: 'linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)',
      borderRadius: '16px', padding: '16px 22px', marginBottom: '24px',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      boxShadow: '0 8px 24px rgba(79,70,229,0.3)',
      animation: 'slideUp 0.4s ease',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <span style={{ fontSize: '24px' }}>🎯</span>
        <div>
          <p style={{ fontSize: '14px', fontWeight: 700, color: 'white', marginBottom: '2px' }}>Demo Mode Active</p>
          <p style={{ fontSize: '12px', color: 'rgba(255,255,255,0.75)' }}>Showing sample PD patient results — no real files needed</p>
        </div>
      </div>
      <button onClick={onDismiss} style={{
        background: 'rgba(255,255,255,0.2)', border: '1px solid rgba(255,255,255,0.3)',
        borderRadius: '10px', padding: '8px 16px', color: 'white',
        fontSize: '13px', fontWeight: 600, cursor: 'pointer',
      }}>✕ Dismiss</button>
    </div>
  )
}

// ── Dashboard Page ───────────────────────────────────────
function DashboardPage({ files, onFileChange, onAnalyze, loading, results, error, dark, onDemo }) {
  const t = dark ? tokens.dark : tokens.light
  const [demoActive, setDemoActive] = useState(false)

  const handleDemo = () => { setDemoActive(true); onDemo() }

  return (
    <div className="page-enter">
      {demoActive && results && <DemoBanner onDismiss={() => setDemoActive(false)} />}
      <StatsBar dark={dark} />
      <UploadSection files={files} onFileChange={onFileChange} onAnalyze={onAnalyze} loading={loading} dark={dark} />

      {/* Demo button */}
      {!results && !loading && (
        <div style={{ textAlign: 'center', margin: '-10px 0 24px' }}>
          <button onClick={handleDemo} style={{
            background: 'none', border: `1.5px dashed ${t.border}`,
            borderRadius: '12px', padding: '10px 24px',
            color: t.muted, fontSize: '13px', fontWeight: 600,
            cursor: 'pointer', transition: 'all 0.2s',
            fontFamily: 'DM Sans, sans-serif',
          }}>
            🎯 Try with Demo Data
          </button>
        </div>
      )}

      {error && (
        <div style={{
          padding: '14px 18px', borderRadius: '12px', marginBottom: '24px',
          background: '#FEF2F2', border: '1.5px solid #FECACA',
          fontSize: '13px', color: '#EF4444', fontWeight: 500,
        }}>⚠️ {error}</div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px', alignItems: 'start', marginBottom: '24px' }}>
        <ResultsPanel results={results} dark={dark} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <RiskGauge score={results?.risk_score ?? null} label={results?.final_label ?? null} dark={dark} />
          {results && <DownloadButton results={results} />}
        </div>
      </div>

      {results && <ExplainabilityPanel results={results} dark={dark} />}
      <Recommendation results={results} dark={dark} />
      <DashboardFooter dark={dark} />
    </div>
  )
}

// ── Analysis Page ────────────────────────────────────────
function AnalysisPage({ files, onFileChange, onAnalyze, loading, results, error, dark }) {
  return (
    <div className="page-enter">
      <UploadSection files={files} onFileChange={onFileChange} onAnalyze={onAnalyze} loading={loading} dark={dark} />
      {error && (
        <div style={{ padding: '14px 18px', borderRadius: '12px', marginBottom: '24px', background: '#FEF2F2', border: '1.5px solid #FECACA', fontSize: '13px', color: '#EF4444', fontWeight: 500 }}>
          ⚠️ {error}
        </div>
      )}
      {results && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px', alignItems: 'start', marginBottom: '24px' }}>
            <ResultsPanel results={results} dark={dark} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <RiskGauge score={results.risk_score} label={results.final_label} dark={dark} />
              <DownloadButton results={results} />
            </div>
          </div>
          <ExplainabilityPanel results={results} dark={dark} />
          <Recommendation results={results} dark={dark} />
        </>
      )}
    </div>
  )
}

// ── Models Page ──────────────────────────────────────────
function ModelsPage({ dark }) {
  const t = dark ? tokens.dark : tokens.light
  const models = [
    {
      name: 'Speech Model', emoji: '🎤', architecture: 'Soft Voting Ensemble',
      components: 'Random Forest · XGBoost · SVM · Logistic Regression',
      dataset: 'Czech UDPR', subjects: 130, features: 24,
      task: '3-class: HC / RBD / PD',
      metrics: { 'CV ROC-AUC': '0.9802', 'CV Accuracy': '55.4%', 'Training Accuracy': '86.2%' },
      color: '#4F46E5', bg: dark ? '#1e1b4b33' : '#EEF2FF', border: dark ? '#4F46E550' : '#C7D2FE',
      note: 'High ROC-AUC indicates excellent discriminative ability. Lower accuracy reflects inherent difficulty of 3-class speech task.',
    },
    {
      name: 'Gait Model', emoji: '🚶', architecture: 'Soft Voting Ensemble',
      components: 'Random Forest · XGBoost · SVM · Logistic Regression',
      dataset: 'PhysioNet', subjects: 165, features: 24,
      task: 'Binary: HC vs PD',
      metrics: { 'CV ROC-AUC': '0.7940', 'CV Accuracy': '73.3%', 'Training Accuracy': '89.7%' },
      color: '#10B981', bg: dark ? '#06503833' : '#ECFDF5', border: dark ? '#10B98150' : '#A7F3D0',
      note: 'XGBoost performed best individually (AUC 0.80). Ensemble balances precision across HC and PD classes.',
    },
    {
      name: 'Handwriting Model', emoji: '✍️', architecture: 'EfficientNetB0 (Transfer Learning)',
      components: 'Frozen backbone (Phase 1) → Full fine-tuning (Phase 2)',
      dataset: 'Kaggle Spiral/Wave', subjects: 204, features: '224×224 images',
      task: 'Binary: HC vs PD',
      metrics: { 'Test ROC-AUC': '0.8844', 'Test Accuracy': '81.7%', 'Test F1': '83.1%' },
      color: '#7C3AED', bg: dark ? '#3b0764aa' : '#F5F3FF', border: dark ? '#7C3AED50' : '#DDD6FE',
      note: 'Two-phase training improved AUC by 8%. High PD recall (90%) is clinically desirable.',
    },
  ]

  return (
    <div className="page-enter">
      <div style={{ background: 'linear-gradient(135deg, #4F46E5, #7C3AED)', borderRadius: '20px', padding: '24px 28px', marginBottom: '24px', color: 'white' }}>
        <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '8px', fontFamily: 'Sora, sans-serif' }}>🔗 Multimodal Fusion Strategy</h3>
        <p style={{ fontSize: '13px', opacity: 0.9, marginBottom: '16px', lineHeight: 1.6 }}>
          Weighted Late Fusion combines predictions from all three modalities. Each modality is weighted by its individual ROC-AUC performance.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '16px' }}>
          {[['🎤 Speech', '45%', 'Highest AUC 0.98'], ['✍️ Handwriting', '30%', 'Middle AUC 0.88'], ['🚶 Gait', '25%', 'AUC 0.79']].map(([l, w, a]) => (
            <div key={l} style={{ background: 'rgba(255,255,255,0.15)', borderRadius: '12px', padding: '14px', textAlign: 'center' }}>
              <p style={{ fontSize: '15px', marginBottom: '4px' }}>{l}</p>
              <p style={{ fontSize: '22px', fontWeight: 800, fontFamily: 'Sora, sans-serif' }}>{w}</p>
              <p style={{ fontSize: '11px', opacity: 0.8 }}>{a}</p>
            </div>
          ))}
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '20px' }}>
        {models.map(m => (
          <div key={m.name} style={{ background: t.card, borderRadius: '20px', border: `1.5px solid ${m.border}`, padding: '24px', boxShadow: `0 4px 16px ${m.color}10` }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '18px' }}>
              <div style={{ width: '44px', height: '44px', borderRadius: '12px', background: m.bg, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '22px' }}>{m.emoji}</div>
              <div>
                <p style={{ fontSize: '14px', fontWeight: 700, color: t.text }}>{m.name}</p>
                <p style={{ fontSize: '11px', color: t.muted }}>{m.task}</p>
              </div>
            </div>
            <div style={{ marginBottom: '14px' }}>
              <p style={{ fontSize: '11px', fontWeight: 600, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.3px', marginBottom: '6px' }}>Architecture</p>
              <p style={{ fontSize: '13px', fontWeight: 600, color: t.text }}>{m.architecture}</p>
              <p style={{ fontSize: '12px', color: t.subtext, marginTop: '3px' }}>{m.components}</p>
            </div>
            <div style={{ marginBottom: '14px' }}>
              <p style={{ fontSize: '11px', fontWeight: 600, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.3px', marginBottom: '6px' }}>Dataset</p>
              <p style={{ fontSize: '13px', color: t.subtext }}>{m.dataset} · {m.subjects} subjects · {m.features} features</p>
            </div>
            <div style={{ marginBottom: '14px' }}>
              <p style={{ fontSize: '11px', fontWeight: 600, color: t.muted, textTransform: 'uppercase', letterSpacing: '0.3px', marginBottom: '8px' }}>Performance</p>
              {Object.entries(m.metrics).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span style={{ fontSize: '12px', color: t.muted }}>{k}</span>
                  <span style={{ fontSize: '12px', fontWeight: 700, color: m.color }}>{v}</span>
                </div>
              ))}
            </div>
            <div style={{ padding: '10px 12px', borderRadius: '10px', background: m.bg, border: `1px solid ${m.border}` }}>
              <p style={{ fontSize: '11px', color: m.color, lineHeight: 1.5 }}>{m.note}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Reports Page ─────────────────────────────────────────
function ReportsPage({ dark, reports, onLoadReport, onDeleteReport }) {
  const t = dark ? tokens.dark : tokens.light
  const [selected, setSelected] = useState(null)

  const riskColor = score =>
    score < 35 ? '#10B981' : score < 60 ? '#F59E0B' : '#EF4444'

  const riskLabel = score =>
    score < 35 ? 'Low Risk' : score < 60 ? 'Moderate' : 'High Risk'

  if (reports.length === 0) {
    return (
      <div className="page-enter">
        <div style={{ background: t.card, borderRadius: '20px', border: `1px solid ${t.border}`, padding: '80px 40px', textAlign: 'center', boxShadow: '0 2px 8px rgba(79,70,229,0.06)' }}>
          <div style={{ fontSize: '52px', marginBottom: '16px' }}>📄</div>
          <h3 style={{ fontSize: '18px', fontWeight: 700, color: t.text, marginBottom: '8px', fontFamily: 'Sora, sans-serif' }}>No Reports Yet</h3>
          <p style={{ fontSize: '14px', color: t.muted }}>Run an analysis from the Dashboard to generate reports.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="page-enter">
      <div style={{ display: 'grid', gridTemplateColumns: selected ? '1fr 1.4fr' : '1fr', gap: '20px' }}>
        {/* Report List */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <p style={{ fontSize: '13px', color: t.muted, fontWeight: 500 }}>{reports.length} report{reports.length !== 1 ? 's' : ''} saved</p>
            <button onClick={() => { reportStore.reports = []; onDeleteReport() }} style={{
              background: 'none', border: `1px solid #FECACA`, borderRadius: '8px',
              padding: '5px 12px', color: '#EF4444', fontSize: '12px',
              fontWeight: 600, cursor: 'pointer',
            }}>🗑 Clear All</button>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {reports.map(r => {
              const score = r.results?.risk_score ?? 0
              const color = riskColor(score)
              const isSelected = selected?.id === r.id
              return (
                <div key={r.id} onClick={() => setSelected(isSelected ? null : r)}
                  style={{
                    background: t.card, borderRadius: '16px',
                    border: `1.5px solid ${isSelected ? '#4F46E5' : t.border}`,
                    padding: '18px 20px', cursor: 'pointer',
                    boxShadow: isSelected ? '0 4px 16px rgba(79,70,229,0.15)' : '0 2px 8px rgba(0,0,0,0.04)',
                    transition: 'all 0.2s',
                    display: 'flex', alignItems: 'center', gap: '16px',
                  }}>
                  <div style={{
                    width: '48px', height: '48px', borderRadius: '12px',
                    background: `${color}20`, display: 'flex', alignItems: 'center',
                    justifyContent: 'center', flexShrink: 0,
                  }}>
                    <span style={{ fontSize: '20px', fontWeight: 800, color, fontFamily: 'Sora, sans-serif' }}>
                      {Math.round(score)}
                    </span>
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                      <p style={{ fontSize: '13px', fontWeight: 700, color: t.text }}>
                        {r.results?.final_label || 'Unknown'}
                      </p>
                      <span style={{ padding: '2px 8px', borderRadius: '99px', fontSize: '10px', fontWeight: 700, background: `${color}20`, color }}>{riskLabel(score)}</span>
                    </div>
                    <p style={{ fontSize: '11px', color: t.muted }}>
                      {new Date(r.timestamp).toLocaleString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
                    </p>
                    <div style={{ display: 'flex', gap: '6px', marginTop: '6px' }}>
                      {r.filesUploaded.map(f => (
                        <span key={f} style={{ fontSize: '10px', padding: '2px 7px', borderRadius: '99px', background: t.bg, color: t.muted, border: `1px solid ${t.border}` }}>
                          {f === 'speech' ? '🎤' : f === 'gait' ? '🚶' : '✍️'} {f}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button onClick={e => { e.stopPropagation(); onLoadReport(r.results) }} style={{
                      background: '#EEF2FF', border: 'none', borderRadius: '8px',
                      padding: '6px 12px', fontSize: '11px', fontWeight: 600,
                      color: '#4F46E5', cursor: 'pointer',
                    }}>Load</button>
                    <button onClick={e => {
                      e.stopPropagation()
                      reportStore.reports = reportStore.reports.filter(x => x.id !== r.id)
                      if (selected?.id === r.id) setSelected(null)
                      onDeleteReport()
                    }} style={{
                      background: '#FEF2F2', border: 'none', borderRadius: '8px',
                      padding: '6px 10px', fontSize: '11px', color: '#EF4444', cursor: 'pointer',
                    }}>✕</button>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Detail Panel */}
        {selected && (
          <div style={{ animation: 'slideUp 0.3s ease' }}>
            <div style={{ background: t.card, borderRadius: '20px', border: `1px solid ${t.border}`, padding: '24px', marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h3 style={{ fontSize: '15px', fontWeight: 700, color: t.text, fontFamily: 'Sora, sans-serif' }}>Report Details</h3>
                <button onClick={() => generatePDF(selected.results)} style={{
                  background: 'linear-gradient(135deg, #10B981, #059669)', border: 'none',
                  borderRadius: '10px', padding: '8px 16px', color: 'white',
                  fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                }}>📄 Export PDF</button>
              </div>
              <RiskGauge score={selected.results?.risk_score ?? null} label={selected.results?.final_label ?? null} dark={dark} />
            </div>
            <ResultsPanel results={selected.results} dark={dark} />
          </div>
        )}
      </div>
    </div>
  )
}

// ── About Page ───────────────────────────────────────────
function AboutPage({ dark }) {
  const t = dark ? tokens.dark : tokens.light
  return (
    <div className="page-enter">
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px' }}>
        <div style={{ background: t.card, borderRadius: '20px', border: `1px solid ${t.border}`, padding: '32px', boxShadow: '0 2px 8px rgba(79,70,229,0.06)' }}>
          <h3 style={{ fontSize: '18px', fontWeight: 700, color: t.text, marginBottom: '16px', fontFamily: 'Sora, sans-serif' }}>🧠 NeuroDetect AI</h3>
          <p style={{ fontSize: '14px', color: t.subtext, lineHeight: 1.8, marginBottom: '16px' }}>
            NeuroDetect AI is a multimodal machine learning system for early Parkinson's Disease detection.
            It combines three independent biomarker modalities — speech patterns, gait dynamics, and handwriting
            characteristics — to provide a comprehensive risk assessment.
          </p>
          <p style={{ fontSize: '14px', color: t.subtext, lineHeight: 1.8, marginBottom: '24px' }}>
            The system is particularly novel in its inclusion of RBD (REM Sleep Behavior Disorder) detection,
            an early precursor to Parkinson's Disease, enabling earlier clinical intervention.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '16px' }}>
            {[['459', 'Total Subjects', 'Across all datasets'], ['3', 'Modalities', 'Speech · Gait · Writing'], ['0.98', 'Best AUC', 'Speech model']].map(([v, l, s]) => (
              <div key={l} style={{ background: t.bg, borderRadius: '14px', padding: '16px', textAlign: 'center', border: `1px solid ${t.border}` }}>
                <p style={{ fontSize: '26px', fontWeight: 800, color: '#4F46E5', fontFamily: 'Sora, sans-serif' }}>{v}</p>
                <p style={{ fontSize: '12px', fontWeight: 600, color: t.text, marginTop: '4px' }}>{l}</p>
                <p style={{ fontSize: '11px', color: t.muted, marginTop: '2px' }}>{s}</p>
              </div>
            ))}
          </div>
        </div>
        <div style={{ background: 'linear-gradient(135deg, #4F46E5, #7C3AED)', borderRadius: '20px', padding: '32px', color: 'white' }}>
          <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '20px', fontFamily: 'Sora, sans-serif' }}>Tech Stack</h3>
          {[['🐍 Python', 'scikit-learn · PyTorch · FastAPI'], ['⚛️ React', 'Vite · Axios · jsPDF'], ['🧠 Models', 'EfficientNetB0 · XGBoost · SVM'], ['📊 Data', 'Czech UDPR · PhysioNet · Kaggle'], ['🔗 Fusion', 'Weighted Late Fusion']].map(([t, d]) => (
            <div key={t} style={{ marginBottom: '16px' }}>
              <p style={{ fontSize: '13px', fontWeight: 600, marginBottom: '3px' }}>{t}</p>
              <p style={{ fontSize: '12px', opacity: 0.8 }}>{d}</p>
            </div>
          ))}
          <div style={{ marginTop: '24px', paddingTop: '20px', borderTop: '1px solid rgba(255,255,255,0.2)' }}>
            <p style={{ fontSize: '12px', opacity: 0.7 }}>Final Year Project · 2026</p>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Alerts Page (API Health Monitor) ────────────────────
function AlertsPage({ dark }) {
  const t = dark ? tokens.dark : tokens.light
  const [health, setHealth] = useState(null)
  const [checking, setChecking] = useState(false)
  const [history, setHistory] = useState([])
  const [latencies, setLatencies] = useState([])
  const [reqCount, setReqCount] = useState(0)
  const intervalRef = useRef(null)

  const check = async () => {
    setChecking(true)
    const start = Date.now()
    try {
      const res = await axios.get(`${API_URL}/health`, { timeout: 3000 })
      const lat = Date.now() - start
      setHealth({ ok: true, data: res.data, latency: lat })
      setLatencies(prev => [...prev.slice(-19), lat])
      setHistory(prev => [{ time: new Date(), ok: true, latency: lat }, ...prev.slice(0, 19)])
      setReqCount(c => c + 1)
    } catch {
      const lat = Date.now() - start
      setHealth({ ok: false, latency: lat })
      setLatencies(prev => [...prev.slice(-19), lat])
      setHistory(prev => [{ time: new Date(), ok: false, latency: lat }, ...prev.slice(0, 19)])
      setReqCount(c => c + 1)
    } finally {
      setChecking(false)
    }
  }

  useEffect(() => {
    check()
    intervalRef.current = setInterval(check, 30000)
    return () => clearInterval(intervalRef.current)
  }, [])

  const avgLat = latencies.length ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length) : 0
  const uptime = history.length ? Math.round((history.filter(h => h.ok).length / history.length) * 100) : 100
  const maxLat = latencies.length ? Math.max(...latencies) : 0

  return (
    <div className="page-enter">
      {/* Status Card */}
      <div style={{
        background: health?.ok ? (dark ? '#064e3b' : '#ECFDF5') : (dark ? '#450a0a' : '#FEF2F2'),
        border: `1.5px solid ${health?.ok ? '#10B981' : '#EF4444'}`,
        borderRadius: '20px', padding: '24px 28px', marginBottom: '20px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        boxShadow: `0 4px 16px ${health?.ok ? '#10B98120' : '#EF444420'}`,
        animation: 'slideUp 0.4s ease',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '18px' }}>
          <div style={{
            width: '56px', height: '56px', borderRadius: '16px',
            background: health?.ok ? '#10B981' : '#EF4444',
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '26px',
            boxShadow: `0 4px 14px ${health?.ok ? '#10B98140' : '#EF444440'}`,
          }}>
            {checking ? '⏳' : health?.ok ? '🟢' : '🔴'}
          </div>
          <div>
            <h3 style={{ fontSize: '18px', fontWeight: 700, color: health?.ok ? '#10B981' : '#EF4444', fontFamily: 'Sora, sans-serif' }}>
              {checking ? 'Checking...' : health?.ok ? 'Backend Online' : 'Backend Offline'}
            </h3>
            <p style={{ fontSize: '13px', color: t.muted, marginTop: '3px' }}>
              {health?.ok ? `FastAPI running · Last checked just now` : 'Cannot reach localhost:8000 — is the backend running?'}
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          {health?.latency && (
            <div style={{ textAlign: 'right' }}>
              <p style={{ fontSize: '22px', fontWeight: 800, color: health.ok ? '#10B981' : '#EF4444', fontFamily: 'Sora, sans-serif' }}>{health.latency}ms</p>
              <p style={{ fontSize: '11px', color: t.muted }}>latency</p>
            </div>
          )}
          <button onClick={check} disabled={checking} style={{
            background: 'linear-gradient(135deg, #4F46E5, #7C3AED)',
            border: 'none', borderRadius: '12px', padding: '10px 20px',
            color: 'white', fontSize: '13px', fontWeight: 600, cursor: 'pointer',
          }}>
            {checking ? '⏳' : '🔄 Refresh'}
          </button>
        </div>
      </div>

      {/* Metric Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: '16px', marginBottom: '20px' }}>
        {[
          { label: 'Avg Latency', value: `${avgLat}ms`, color: '#4F46E5', emoji: '⚡' },
          { label: 'Max Latency', value: `${maxLat}ms`, color: '#F59E0B', emoji: '📈' },
          { label: 'Uptime (session)', value: `${uptime}%`, color: '#10B981', emoji: '✅' },
          { label: 'Requests Made', value: reqCount, color: '#7C3AED', emoji: '📡' },
        ].map(m => (
          <div key={m.label} style={{ background: t.card, borderRadius: '16px', border: `1px solid ${t.border}`, padding: '18px 20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
              <span style={{ fontSize: '18px' }}>{m.emoji}</span>
              <p style={{ fontSize: '12px', color: t.muted, fontWeight: 500 }}>{m.label}</p>
            </div>
            <p style={{ fontSize: '26px', fontWeight: 800, color: m.color, fontFamily: 'Sora, sans-serif' }}>{m.value}</p>
          </div>
        ))}
      </div>

      {/* Latency Sparkline */}
      {latencies.length > 1 && (
        <div style={{ background: t.card, borderRadius: '16px', border: `1px solid ${t.border}`, padding: '20px', marginBottom: '20px' }}>
          <p style={{ fontSize: '13px', fontWeight: 600, color: t.text, marginBottom: '14px' }}>⚡ Latency History</p>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: '4px', height: '48px' }}>
            {latencies.map((l, i) => {
              const h = Math.max(4, (l / Math.max(...latencies)) * 48)
              const color = l < 100 ? '#10B981' : l < 300 ? '#F59E0B' : '#EF4444'
              return <div key={i} style={{ flex: 1, height: `${h}px`, background: color, borderRadius: '3px', opacity: 0.7 + (i / latencies.length) * 0.3 }} />
            })}
          </div>
        </div>
      )}

      {/* Request Log */}
      <div style={{ background: t.card, borderRadius: '16px', border: `1px solid ${t.border}`, padding: '20px' }}>
        <p style={{ fontSize: '13px', fontWeight: 600, color: t.text, marginBottom: '14px' }}>📋 Health Check Log</p>
        {history.length === 0 ? (
          <p style={{ fontSize: '13px', color: t.muted }}>No checks yet...</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '280px', overflowY: 'auto' }}>
            {history.map((h, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '10px 14px', borderRadius: '10px',
                background: dark ? 'rgba(255,255,255,0.04)' : '#F8FAFF',
                border: `1px solid ${t.border}`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span>{h.ok ? '🟢' : '🔴'}</span>
                  <span style={{ fontSize: '12px', color: t.muted }}>
                    {h.time.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </span>
                </div>
                <span style={{ fontSize: '12px', fontWeight: 600, color: h.ok ? '#10B981' : '#EF4444' }}>{h.latency}ms</span>
              </div>
            ))}
          </div>
        )}
        <p style={{ fontSize: '11px', color: t.muted, marginTop: '12px' }}>Auto-refreshes every 30 seconds</p>
      </div>
    </div>
  )
}

// ── Settings Page ────────────────────────────────────────
function SettingsPage({ dark, onToggleDark }) {
  const t = dark ? tokens.dark : tokens.light
  const [apiUrl, setApiUrl] = useState('http://localhost:8000')
  const [notifications, setNotif] = useState(true)
  const [autoAnalyze, setAuto] = useState(false)
  const [showTime, setShowTime] = useState(true)
  const [showProbs, setShowProbs] = useState(true)
  const [clinicalMode, setClinical] = useState(false)
  const [riskLow, setRiskLow] = useState(35)
  const [riskHigh, setRiskHigh] = useState(60)
  const [confidence, setConf] = useState(50)
  const [saved, setSaved] = useState(false)

  const handleSave = () => { setSaved(true); setTimeout(() => setSaved(false), 2500) }
  const handleReset = () => { setApiUrl('http://localhost:8000'); setNotif(true); setAuto(false); setShowTime(true); setShowProbs(true); setClinical(false); setRiskLow(35); setRiskHigh(60); setConf(50) }

  const Toggle = ({ value, onChange }) => (
    <div onClick={() => onChange(!value)} style={{ width: '44px', height: '24px', borderRadius: '99px', cursor: 'pointer', background: value ? '#4F46E5' : (dark ? '#374151' : '#E5E7EB'), position: 'relative', transition: 'background 0.25s', flexShrink: 0 }}>
      <div style={{ position: 'absolute', top: '3px', left: value ? '23px' : '3px', width: '18px', height: '18px', borderRadius: '50%', background: 'white', transition: 'left 0.25s', boxShadow: '0 1px 4px rgba(0,0,0,0.15)' }} />
    </div>
  )

  const Section = ({ title, children }) => (
    <div style={{ background: t.card, borderRadius: '20px', border: `1px solid ${t.border}`, padding: '26px', marginBottom: '20px', boxShadow: '0 2px 8px rgba(79,70,229,0.06)' }}>
      <h3 style={{ fontSize: '14px', fontWeight: 700, color: t.text, marginBottom: '18px', fontFamily: 'Sora, sans-serif', paddingBottom: '12px', borderBottom: `1px solid ${t.border}` }}>{title}</h3>
      {children}
    </div>
  )

  const Row = ({ label, sub, children }) => (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
      <div style={{ flex: 1, paddingRight: '16px' }}>
        <p style={{ fontSize: '13px', fontWeight: 500, color: t.text }}>{label}</p>
        {sub && <p style={{ fontSize: '11px', color: t.muted, marginTop: '2px' }}>{sub}</p>}
      </div>
      {children}
    </div>
  )

  const Slider = ({ value, onChange, min, max, color, label }) => (
    <div style={{ marginBottom: '22px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
        <span style={{ fontSize: '13px', fontWeight: 500, color: t.text }}>{label}</span>
        <span style={{ fontSize: '14px', fontWeight: 700, color }}>{value}</span>
      </div>
      <input type="range" min={min} max={max} value={value} onChange={e => onChange(Number(e.target.value))}
        style={{ width: '100%', accentColor: color, cursor: 'pointer', height: '4px' }} />
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px' }}>
        <span style={{ fontSize: '11px', color: t.muted }}>{min}</span>
        <span style={{ fontSize: '11px', color: t.muted }}>{max}</span>
      </div>
    </div>
  )

  return (
    <div className="page-enter">
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <div>
          <Section title="⚙️ API Configuration">
            <Row label="Backend URL" sub="FastAPI server endpoint">
              <input value={apiUrl} onChange={e => setApiUrl(e.target.value)} style={{ width: '200px', padding: '8px 12px', borderRadius: '10px', border: `1.5px solid ${t.border}`, fontSize: '13px', color: t.text, fontFamily: 'DM Sans, sans-serif', outline: 'none', background: t.bg }} />
            </Row>
          </Section>
          <Section title="🎨 Appearance">
            <Row label="Dark Mode" sub="Switch to dark theme"><Toggle value={dark} onChange={onToggleDark} /></Row>
            <Row label="Show Processing Time" sub="Display inference time on result cards"><Toggle value={showTime} onChange={setShowTime} /></Row>
            <Row label="Show Raw Probabilities" sub="Display probability bars per class"><Toggle value={showProbs} onChange={setShowProbs} /></Row>
          </Section>
          <Section title="🔔 Notifications">
            <Row label="Analysis Complete Alert" sub="Notify when prediction finishes"><Toggle value={notifications} onChange={setNotif} /></Row>
            <Row label="Auto-Analyze" sub="Run automatically when all 3 files uploaded"><Toggle value={autoAnalyze} onChange={setAuto} /></Row>
            <Row label="Clinical Mode" sub="Add extra disclaimers for clinical presentation"><Toggle value={clinicalMode} onChange={setClinical} /></Row>
          </Section>
        </div>
        <div>
          <Section title="📊 Risk Score Thresholds">
            <p style={{ fontSize: '12px', color: t.muted, marginBottom: '18px' }}>Drag to customize the risk level boundaries</p>
            <Slider value={riskLow} onChange={setRiskLow} min={20} max={50} color="#10B981" label="Low → Moderate threshold" />
            <Slider value={riskHigh} onChange={setRiskHigh} min={50} max={80} color="#EF4444" label="Moderate → High threshold" />
            <div style={{ borderRadius: '12px', overflow: 'hidden', height: '10px', display: 'flex', marginTop: '4px' }}>
              <div style={{ width: `${riskLow}%`, background: '#10B981', transition: 'width 0.2s' }} />
              <div style={{ width: `${riskHigh - riskLow}%`, background: '#F59E0B', transition: 'width 0.2s' }} />
              <div style={{ flex: 1, background: '#EF4444' }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px' }}>
              <span style={{ fontSize: '11px', fontWeight: 600, color: '#10B981' }}>Healthy (0–{riskLow})</span>
              <span style={{ fontSize: '11px', fontWeight: 600, color: '#F59E0B' }}>At Risk ({riskLow}–{riskHigh})</span>
              <span style={{ fontSize: '11px', fontWeight: 600, color: '#EF4444' }}>PD ({riskHigh}–100)</span>
            </div>
          </Section>
          <Section title="🔬 Analysis Settings">
            <Slider value={confidence} onChange={setConf} min={0} max={90} color="#4F46E5" label="Minimum confidence threshold (%)" />
          </Section>
        </div>
      </div>
      <div style={{ display: 'flex', gap: '12px', marginTop: '4px' }}>
        <button onClick={handleSave} style={{ padding: '13px 32px', borderRadius: '14px', border: 'none', cursor: 'pointer', background: saved ? '#10B981' : 'linear-gradient(135deg, #4F46E5, #7C3AED)', color: 'white', fontSize: '14px', fontWeight: 600, boxShadow: saved ? '0 4px 14px rgba(16,185,129,0.4)' : '0 4px 14px rgba(79,70,229,0.4)', transition: 'all 0.3s', fontFamily: 'DM Sans, sans-serif' }}>
          {saved ? '✅ Saved!' : '💾 Save Settings'}
        </button>
        <button onClick={handleReset} style={{ padding: '13px 32px', borderRadius: '14px', cursor: 'pointer', background: t.card, color: t.muted, fontSize: '14px', fontWeight: 600, border: `1.5px solid ${t.border}`, transition: 'all 0.25s', fontFamily: 'DM Sans, sans-serif' }}>
          🔄 Reset
        </button>
      </div>
    </div>
  )
}

// ── Demo Data ────────────────────────────────────────────
const DEMO_RESULTS = {
  risk_score: 73.4,
  risk_level: 'High Risk',
  final_label: 'Parkinson\'s Disease (PD)',
  recommendation: 'Strong indicators of Parkinson\'s Disease detected across multiple modalities. Immediate referral to a movement disorder specialist is strongly recommended. Further clinical evaluation including neurological examination and dopamine transporter (DaT) scan is advised.',
  processing_time_ms: 142,
  fusion_weights: { speech: 0.45, handwriting: 0.30, gait: 0.25 },
  modality_results: {
    speech: { label: 'PD', confidence: 0.81, probabilities: { HC: 0.09, RBD: 0.10, PD: 0.81 }, processing_time_ms: 38 },
    gait: { label: 'PD', confidence: 0.76, probabilities: { HC: 0.24, PD: 0.76 }, processing_time_ms: 22 },
    handwriting: { label: 'PD', confidence: 0.88, probabilities: { HC: 0.12, PD: 0.88 }, processing_time_ms: 82 },
  },
}

// ── Design Tokens ────────────────────────────────────────
export const tokens = {
  light: {
    bg: '#EEF2FF',
    card: '#FFFFFF',
    border: '#E0E7FF',
    text: '#1E1B4B',
    subtext: '#4B5563',
    muted: '#6B7280',
    sidebar: '#FFFFFF',
    sidebarBorder: '#E0E7FF',
  },
  dark: {
    bg: '#0F0E1A',
    card: '#1A1830',
    border: '#2D2B4E',
    text: '#E8E6FF',
    subtext: '#A5B4FC',
    muted: '#6B7280',
    sidebar: '#13122A',
    sidebarBorder: '#2D2B4E',
  },
}

// ── Main App ─────────────────────────────────────────────
function App() {
  const { dark, toggle: toggleDark } = useTheme()
  const t = dark ? tokens.dark : tokens.light
  const [activePage, setActivePage] = useState('dashboard')
  const [files, setFiles] = useState({ speech: null, gait: null, handwriting: null })
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [savedReports, setSavedReports] = useState([])

  const handleFileChange = (key, file) => {
    setFiles(prev => ({ ...prev, [key]: file }))
    setResults(null); setError(null)
  }

  const handleAnalyze = async () => {
    setLoading(true); setError(null); setResults(null)
    try {
      const formData = new FormData()
      if (files.speech) formData.append('speech_file', files.speech)
      if (files.gait) formData.append('gait_file', files.gait)
      if (files.handwriting) formData.append('handwriting_file', files.handwriting)
      const res = await axios.post(`${API_URL}/predict/fusion`, formData, { headers: { 'Content-Type': 'multipart/form-data' } })
      setResults(res.data)
      saveReport(res.data, files)
      setSavedReports([...reportStore.reports])
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Analysis failed. Is the backend running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  const handleDemo = () => {
    setResults(DEMO_RESULTS)
    saveReport(DEMO_RESULTS, { speech: 'demo', gait: 'demo', handwriting: 'demo' })
    setSavedReports([...reportStore.reports])
  }

  const handleLoadReport = (r) => { setResults(r); setActivePage('dashboard') }
  const handleDeleteReport = () => setSavedReports([...reportStore.reports])

  const pageProps = { files, onFileChange: handleFileChange, onAnalyze: handleAnalyze, loading, results, error, dark }

  const pageTitle = {
    dashboard: { title: 'Dashboard', sub: "Multimodal Parkinson's Disease Detection System" },
    analysis: { title: 'Analysis', sub: 'Upload patient data and run detection pipeline' },
    models: { title: 'Model Info', sub: 'Architecture and performance metrics' },
    reports: { title: 'Reports', sub: `${savedReports.length} analysis report${savedReports.length !== 1 ? 's' : ''} saved` },
    about: { title: 'About', sub: 'Project information and tech stack' },
    alerts: { title: 'System Alerts', sub: 'Backend health monitor & API status' },
    settings: { title: 'Settings', sub: 'Configure your NeuroDetect AI preferences' },
  }

  const current = pageTitle[activePage] || pageTitle.dashboard

  // Apply dark mode to body
  useEffect(() => {
    document.body.style.background = t.bg
    document.body.style.color = t.text
  }, [dark])

  return (
    <div className="app-layout" style={{ background: t.bg }}>
      <Sidebar activePage={activePage} onNavigate={setActivePage} dark={dark} reportCount={savedReports.length} />
      <main className="main-content" style={{ background: t.bg }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '28px' }}>
          <div>
            <h1 style={{ fontSize: '26px', fontWeight: 800, color: t.text, letterSpacing: '-0.5px', fontFamily: 'Sora, sans-serif' }}>
              {current.title}
            </h1>
            <p style={{ fontSize: '14px', color: t.muted, marginTop: '3px' }}>{current.sub}</p>
          </div>
          <button onClick={toggleDark} style={{
            background: dark ? '#2D2B4E' : '#EEF2FF',
            border: `1px solid ${t.border}`, borderRadius: '12px',
            padding: '10px 16px', cursor: 'pointer',
            fontSize: '18px', lineHeight: 1, transition: 'all 0.2s',
          }} title={dark ? 'Switch to light mode' : 'Switch to dark mode'}>
            {dark ? '☀️' : '🌙'}
          </button>
        </div>

        {activePage === 'dashboard' && <DashboardPage {...pageProps} onDemo={handleDemo} />}
        {activePage === 'analysis' && <AnalysisPage {...pageProps} />}
        {activePage === 'models' && <ModelsPage dark={dark} />}
        {activePage === 'reports' && <ReportsPage dark={dark} reports={savedReports} onLoadReport={handleLoadReport} onDeleteReport={handleDeleteReport} />}
        {activePage === 'about' && <AboutPage dark={dark} />}
        {activePage === 'alerts' && <AlertsPage dark={dark} />}
        {activePage === 'settings' && <SettingsPage dark={dark} onToggleDark={toggleDark} />}
      </main>
    </div>
  )
}

export default App 