import { useState, useRef, useEffect } from 'react'
import { tokens } from '../App'

// ── Mic Recorder ─────────────────────────────────────────
function MicRecorder({ onRecorded, dark }) {
  const t = dark ? tokens.dark : tokens.light
  const [state, setState] = useState('idle') // idle | recording | done
  const [seconds, setSeconds] = useState(0)
  const [waveform, setWaveform] = useState(Array(24).fill(2))
  const mediaRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const analyserRef = useRef(null)
  const animRef = useRef(null)

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const ctx = new AudioContext()
      const src = ctx.createMediaStreamSource(stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 64
      src.connect(analyser)
      analyserRef.current = analyser

      const mr = new MediaRecorder(stream)
      mediaRef.current = mr
      chunksRef.current = []
      mr.ondataavailable = e => chunksRef.current.push(e.data)
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/wav' })
        const file = new File([blob], `recording_${Date.now()}.wav`, { type: 'audio/wav' })
        onRecorded(file)
        stream.getTracks().forEach(t => t.stop())
        cancelAnimationFrame(animRef.current)
      }
      mr.start()
      setState('recording')
      setSeconds(0)

      timerRef.current = setInterval(() => setSeconds(s => s + 1), 1000)

      const drawWave = () => {
        const data = new Uint8Array(analyser.frequencyBinCount)
        analyser.getByteFrequencyData(data)
        setWaveform(Array.from(data.slice(0, 24)).map(v => Math.max(2, (v / 255) * 40)))
        animRef.current = requestAnimationFrame(drawWave)
      }
      drawWave()
    } catch (e) {
      alert('Microphone access denied. Please allow microphone access and try again.')
    }
  }

  const stopRecording = () => {
    mediaRef.current?.stop()
    clearInterval(timerRef.current)
    setState('done')
    setWaveform(Array(24).fill(2))
  }

  const reset = () => { setState('idle'); setSeconds(0); setWaveform(Array(24).fill(2)); onRecorded(null) }

  const fmt = s => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`

  if (state === 'idle') {
    return (
      <div style={{ textAlign: 'center', padding: '10px 0' }}>
        <button onClick={startRecording} style={{
          background: 'linear-gradient(135deg, #EF4444, #DC2626)',
          border: 'none', borderRadius: '50%', width: '52px', height: '52px',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'pointer', margin: '0 auto 8px',
          boxShadow: '0 4px 14px rgba(239,68,68,0.4)',
          fontSize: '22px',
        }}>🎙️</button>
        <p style={{ fontSize: '11px', color: t.muted }}>Click to record live audio</p>
      </div>
    )
  }

  if (state === 'recording') {
    return (
      <div style={{ padding: '10px 0' }}>
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'center', gap: '2px', height: '44px', marginBottom: '10px' }}>
          {waveform.map((h, i) => (
            <div key={i} style={{
              width: '4px', height: `${h}px`, background: '#EF4444',
              borderRadius: '2px', transition: 'height 0.08s ease',
              opacity: 0.6 + (i % 3) * 0.13,
            }} />
          ))}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#EF4444', animation: 'pulse 1s infinite' }} />
          <span style={{ fontSize: '13px', fontWeight: 600, color: '#EF4444' }}>Recording {fmt(seconds)}</span>
          <button onClick={stopRecording} style={{
            background: '#1E1B4B', border: 'none', borderRadius: '8px',
            padding: '5px 12px', color: 'white', fontSize: '12px', fontWeight: 600, cursor: 'pointer',
          }}>⏹ Stop</button>
        </div>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 0' }}>
      <span style={{ fontSize: '20px' }}>✅</span>
      <div style={{ flex: 1 }}>
        <p style={{ fontSize: '12px', fontWeight: 600, color: t.text }}>Recording saved ({fmt(seconds)})</p>
        <p style={{ fontSize: '11px', color: t.muted }}>Ready to analyze</p>
      </div>
      <button onClick={reset} style={{ background: '#FEF2F2', border: 'none', borderRadius: '8px', padding: '5px 10px', color: '#EF4444', fontSize: '12px', cursor: 'pointer' }}>✕</button>
    </div>
  )
}

// ── Drawing Pad ───────────────────────────────────────────
function DrawingPad({ onDrawn, dark }) {
  const t = dark ? tokens.dark : tokens.light
  const canvasRef = useRef(null)
  const [drawing, setDrawing] = useState(false)
  const [hasDrawing, setHasDrawing] = useState(false)
  const [mode, setMode] = useState('spiral') // spiral | wave

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = '#FFFFFF'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    // Draw guide
    drawGuide(ctx, mode)
  }, [mode])

  const drawGuide = (ctx, type) => {
    ctx.save()
    ctx.strokeStyle = '#E0E7FF'
    ctx.lineWidth = 1
    ctx.setLineDash([4, 4])
    if (type === 'spiral') {
      for (let i = 0; i < 4; i++) {
        ctx.beginPath()
        for (let a = 0; a < Math.PI * 2 * (i + 1); a += 0.05) {
          const r = (i + 1) * 18 + a * 3
          const x = 112 + r * Math.cos(a - Math.PI / 2)
          const y = 100 + r * Math.sin(a - Math.PI / 2)
          if (a === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.stroke()
      }
    } else {
      for (let i = 0; i < 4; i++) {
        ctx.beginPath()
        for (let x = 0; x < 224; x++) {
          const y = 100 + Math.sin((x / 224) * Math.PI * 4) * (20 + i * 10)
          if (x === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.stroke()
      }
    }
    ctx.restore()
  }

  const getPos = (e, canvas) => {
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    if (e.touches) {
      return { x: (e.touches[0].clientX - rect.left) * scaleX, y: (e.touches[0].clientY - rect.top) * scaleY }
    }
    return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY }
  }

  const startDraw = (e) => {
    e.preventDefault()
    setDrawing(true)
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const pos = getPos(e, canvas)
    ctx.beginPath()
    ctx.moveTo(pos.x, pos.y)
    ctx.strokeStyle = '#1E1B4B'
    ctx.lineWidth = 2.5
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
  }

  const draw = (e) => {
    e.preventDefault()
    if (!drawing) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const pos = getPos(e, canvas)
    ctx.lineTo(pos.x, pos.y)
    ctx.stroke()
    setHasDrawing(true)
  }

  const endDraw = () => {
    setDrawing(false)
    if (hasDrawing) {
      canvasRef.current.toBlob(blob => {
        const file = new File([blob], `drawing_${mode}_${Date.now()}.png`, { type: 'image/png' })
        onDrawn(file)
      }, 'image/png')
    }
  }

  const clear = () => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    ctx.fillStyle = '#FFFFFF'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    drawGuide(ctx, mode)
    setHasDrawing(false)
    onDrawn(null)
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: '6px', marginBottom: '8px', justifyContent: 'center' }}>
        {['spiral', 'wave'].map(m => (
          <button key={m} onClick={() => { setMode(m); clear() }} style={{
            padding: '4px 12px', borderRadius: '8px', fontSize: '11px', fontWeight: 600,
            background: mode === m ? '#4F46E5' : t.bg, color: mode === m ? 'white' : t.muted,
            border: `1px solid ${mode === m ? '#4F46E5' : t.border}`, cursor: 'pointer',
          }}>{m === 'spiral' ? '🌀 Spiral' : '〰️ Wave'}</button>
        ))}
        {hasDrawing && (
          <button onClick={clear} style={{ padding: '4px 10px', borderRadius: '8px', fontSize: '11px', background: '#FEF2F2', color: '#EF4444', border: 'none', cursor: 'pointer' }}>🗑 Clear</button>
        )}
      </div>
      <canvas
        ref={canvasRef}
        width={224} height={200}
        onMouseDown={startDraw} onMouseMove={draw} onMouseUp={endDraw} onMouseLeave={endDraw}
        onTouchStart={startDraw} onTouchMove={draw} onTouchEnd={endDraw}
        style={{
          width: '100%', height: '130px', borderRadius: '10px', cursor: 'crosshair',
          border: `1.5px solid ${drawing ? '#4F46E5' : t.border}`, display: 'block',
          touchAction: 'none',
        }}
      />
      <p style={{ fontSize: '10px', color: t.muted, textAlign: 'center', marginTop: '5px' }}>
        {hasDrawing ? '✅ Drawing captured — ready to analyze' : 'Draw over the guide lines'}
      </p>
    </div>
  )
}

// ── Upload Card ───────────────────────────────────────────
function UploadCard({ title, subtitle, accept, fileKey, files, onFileChange, color, bg, border, hint, emoji, dark }) {
  const [dragging, setDragging] = useState(false)
  const [tab, setTab] = useState('upload') // upload | record | draw
  const t = dark ? tokens.dark : tokens.light
  const inputRef = useRef()
  const file = files[fileKey]
  const isSpeech = fileKey === 'speech'
  const isHandwriting = fileKey === 'handwriting'

  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) onFileChange(fileKey, f)
  }
  const handleChange = (e) => { if (e.target.files[0]) onFileChange(fileKey, e.target.files[0]) }
  const removeFile = (e) => { e.stopPropagation(); onFileChange(fileKey, null); if (inputRef.current) inputRef.current.value = '' }

  const tabs = isSpeech
    ? [{ id: 'upload', label: '📁 File' }, { id: 'record', label: '🎙️ Record' }]
    : isHandwriting
      ? [{ id: 'upload', label: '📁 File' }, { id: 'draw', label: '✏️ Draw' }]
      : null

  return (
    <div style={{
      background: t.card, borderRadius: '18px',
      border: dragging ? `2px dashed ${color}` : file ? `2px solid ${color}` : `1.5px solid ${t.border}`,
      padding: '22px', transition: 'all 0.25s',
      transform: dragging ? 'scale(1.02)' : 'none',
      boxShadow: file ? `0 4px 16px ${color}20` : `0 2px 8px rgba(79,70,229,0.06)`,
    }}>
      <input ref={inputRef} type="file" accept={accept} onChange={handleChange} style={{ display: 'none' }} />

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: '40px', height: '40px', borderRadius: '11px', background: bg, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '19px' }}>{emoji}</div>
          <div>
            <p style={{ fontSize: '14px', fontWeight: 600, color: t.text }}>{title}</p>
            <p style={{ fontSize: '12px', color: t.muted, marginTop: '1px' }}>{subtitle}</p>
          </div>
        </div>
        {file && (
          <button onClick={removeFile} style={{ background: '#FEF2F2', border: 'none', borderRadius: '8px', width: '28px', height: '28px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', color: '#EF4444', fontSize: '18px', lineHeight: 1 }}>×</button>
        )}
      </div>

      {/* Tab switcher */}
      {tabs && !file && (
        <div style={{ display: 'flex', gap: '6px', marginBottom: '12px' }}>
          {tabs.map(tb => (
            <button key={tb.id} onClick={() => setTab(tb.id)} style={{
              flex: 1, padding: '6px 0', borderRadius: '8px', fontSize: '11px', fontWeight: 600,
              background: tab === tb.id ? color : t.bg,
              color: tab === tb.id ? 'white' : t.muted,
              border: `1px solid ${tab === tb.id ? color : t.border}`,
              cursor: 'pointer', transition: 'all 0.15s',
            }}>{tb.label}</button>
          ))}
        </div>
      )}

      {/* Content */}
      {file ? (
        <>
          <div style={{ borderRadius: '13px', padding: '13px 16px', background: t.bg, display: 'flex', alignItems: 'center', gap: '12px', border: `1px solid ${t.border}` }}>
            <div style={{ width: '36px', height: '36px', borderRadius: '9px', background: bg, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '16px', flexShrink: 0 }}>📎</div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <p style={{ fontSize: '13px', fontWeight: 600, color: t.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{file.name}</p>
              <p style={{ fontSize: '11px', color: t.muted, marginTop: '2px' }}>{(file.size / 1024).toFixed(1)} KB · Ready</p>
            </div>
            <span style={{ fontSize: '18px' }}>✅</span>
          </div>
          {fileKey === 'handwriting' && (
            <div style={{ marginTop: '12px', borderRadius: '11px', overflow: 'hidden', height: '90px', border: `1px solid ${t.border}` }}>
              <img src={URL.createObjectURL(file)} alt="Preview" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            </div>
          )}
        </>
      ) : tab === 'record' && isSpeech ? (
        <MicRecorder onRecorded={f => f && onFileChange(fileKey, f)} dark={dark} />
      ) : tab === 'draw' && isHandwriting ? (
        <DrawingPad onDrawn={f => f && onFileChange(fileKey, f)} dark={dark} />
      ) : (
        <div
          onClick={() => inputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          style={{
            borderRadius: '13px', padding: '22px 16px', textAlign: 'center',
            background: dragging ? bg : t.bg, border: `1.5px dashed ${t.border}`,
            transition: 'all 0.2s', cursor: 'pointer',
          }}
        >
          <div style={{ fontSize: '26px', marginBottom: '8px' }}>📂</div>
          <p style={{ fontSize: '13px', fontWeight: 500, color: t.muted, marginBottom: '3px' }}>
            Drop file here or <span style={{ color, fontWeight: 600 }}>browse</span>
          </p>
          <p style={{ fontSize: '11px', color: t.muted }}>{hint}</p>
        </div>
      )}
    </div>
  )
}

// ── Upload Section ────────────────────────────────────────
function UploadSection({ files, onFileChange, onAnalyze, loading, dark }) {
  const t = dark ? tokens.dark : tokens.light
  const hasFile = Object.values(files).some(f => f !== null)
  const count = Object.values(files).filter(Boolean).length

  return (
    <div style={{ marginBottom: '28px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
        <div>
          <h2 style={{ fontSize: '17px', fontWeight: 700, color: t.text, fontFamily: 'Sora, sans-serif' }}>
            Upload Patient Data
          </h2>
          <p style={{ fontSize: '13px', color: t.muted, marginTop: '3px' }}>
            Upload files, record live audio, or draw spirals directly
          </p>
        </div>
        <span style={{
          padding: '4px 14px', borderRadius: '99px', fontSize: '12px', fontWeight: 600,
          background: count > 0 ? (dark ? '#1e1b4b' : '#EEF2FF') : t.bg,
          color: count > 0 ? '#4F46E5' : t.muted,
          border: `1px solid ${count > 0 ? '#C7D2FE' : t.border}`,
        }}>{count}/3 uploaded</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '20px', marginBottom: '20px' }}>
        <UploadCard title="Speech Analysis" subtitle="Czech UDPR Features" accept=".csv,.wav" fileKey="speech" files={files} onFileChange={onFileChange} color="#4F46E5" bg="#EEF2FF" border="#C7D2FE" hint="WAV audio or CSV features" emoji="🎤" dark={dark} />
        <UploadCard title="Gait Analysis" subtitle="PhysioNet Format" accept=".txt" fileKey="gait" files={files} onFileChange={onFileChange} color="#10B981" bg="#ECFDF5" border="#A7F3D0" hint="19-column tab-separated .txt" emoji="🚶" dark={dark} />
        <UploadCard title="Handwriting" subtitle="Spiral / Wave Drawing" accept=".png,.jpg,.jpeg" fileKey="handwriting" files={files} onFileChange={onFileChange} color="#7C3AED" bg="#F5F3FF" border="#DDD6FE" hint="PNG or JPG spiral/wave image" emoji="✍️" dark={dark} />
      </div>

      <button
        onClick={onAnalyze}
        disabled={!hasFile || loading}
        style={{
          width: '100%', padding: '15px 32px', borderRadius: '16px',
          background: (!hasFile || loading) ? (dark ? '#2D2B4E' : '#C7D2FE') : 'linear-gradient(135deg, #4F46E5, #7C3AED)',
          color: (!hasFile || loading) ? (dark ? '#6B7280' : '#6B7280') : 'white',
          border: 'none', fontSize: '15px', fontWeight: 600,
          cursor: (!hasFile || loading) ? 'not-allowed' : 'pointer',
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px',
          boxShadow: (!hasFile || loading) ? 'none' : '0 6px 20px rgba(79,70,229,0.4)',
          transition: 'all 0.25s', letterSpacing: '0.1px',
          fontFamily: 'DM Sans, sans-serif',
        }}
      >
        {loading ? (
          <>
            <span style={{ display: 'inline-block', width: '18px', height: '18px', border: '2.5px solid rgba(255,255,255,0.3)', borderTopColor: 'white', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />
            Analyzing all modalities...
          </>
        ) : (<>🔬 Run Multimodal Analysis</>)}
      </button>
    </div>
  )
}

export default UploadSection  