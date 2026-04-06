import jsPDF from 'jspdf'

function generatePDF(results) {
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' })
  const W = 210, margin = 20
  let y = 0

  const colors = {
    primary:   [79,  70,  229],
    purple:    [124, 58,  237],
    green:     [16,  185, 129],
    amber:     [245, 158, 11],
    red:       [239, 68,  68],
    dark:      [30,  27,  75],
    gray:      [107, 114, 128],
    lightgray: [243, 244, 246],
    white:     [255, 255, 255],
    border:    [224, 231, 255],
  }

  const riskColor = results.risk_score < 35 ? colors.green :
                    results.risk_score < 60 ? colors.amber : colors.red

  const predColor = (label) =>
    label?.includes('HC')  || label === 'HC'  ? colors.green :
    label?.includes('RBD') || label === 'RBD' ? colors.amber : colors.red

  // ── HEADER BANNER ──────────────────────────────────────
  doc.setFillColor(...colors.primary)
  doc.rect(0, 0, 150, 42, 'F')
  doc.setFillColor(...colors.purple)
  doc.rect(150, 0, 60, 42, 'F')

  // Logo circle
  doc.setFillColor(255, 255, 255)
  doc.circle(margin + 8, 21, 8, 'F')
  doc.setFontSize(9)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...colors.primary)
  doc.text('AI', margin + 4.8, 23.5)

  // Title
  doc.setTextColor(...colors.white)
  doc.setFontSize(20)
  doc.setFont('helvetica', 'bold')
  doc.text('NeuroDetect AI', margin + 22, 17)

  doc.setFontSize(10)
  doc.setFont('helvetica', 'normal')
  doc.text("Multimodal Parkinson's Disease Detection Report", margin + 22, 25)

  const now = new Date()
  const dateStr = now.toLocaleDateString('en-GB', { day:'2-digit', month:'long', year:'numeric' })
  const timeStr = now.toLocaleTimeString('en-GB', { hour:'2-digit', minute:'2-digit' })
  doc.setFontSize(9)
  doc.text(`Generated: ${dateStr} at ${timeStr}`, margin + 22, 33)

  y = 52

  // ── RISK SCORE CARD ────────────────────────────────────
  doc.setFillColor(...colors.lightgray)
  doc.roundedRect(margin, y, W - margin*2, 36, 4, 4, 'F')
  doc.setDrawColor(...colors.border)
  doc.roundedRect(margin, y, W - margin*2, 36, 4, 4, 'S')

  doc.setFontSize(32)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...riskColor)
  doc.text(results.risk_score?.toFixed(1) || '--', margin + 10, y + 24)

  doc.setFontSize(9)
  doc.setFont('helvetica', 'normal')
  doc.setTextColor(...colors.gray)
  doc.text('RISK SCORE', margin + 10, y + 32)

  doc.setDrawColor(...colors.border)
  doc.line(margin + 45, y + 6, margin + 45, y + 30)

  doc.setFillColor(...riskColor)
  doc.roundedRect(margin + 52, y + 8, 50, 10, 3, 3, 'F')
  doc.setFontSize(10)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...colors.white)
  doc.text(results.risk_level || 'Unknown', margin + 77, y + 15.5, { align:'center' })

  doc.setFontSize(11)
  doc.setFont('helvetica', 'normal')
  doc.setTextColor(...colors.dark)
  doc.text(`Final Diagnosis: ${results.final_label || 'N/A'}`, margin + 52, y + 28)

  const barX = margin + 115, barY = y + 12, barW = 55, barH = 8
  doc.setFillColor(229, 231, 235)
  doc.roundedRect(barX, barY, barW, barH, 2, 2, 'F')
  const fillW = (results.risk_score / 100) * barW
  doc.setFillColor(...riskColor)
  doc.roundedRect(barX, barY, fillW, barH, 2, 2, 'F')
  doc.setFontSize(8)
  doc.setTextColor(...colors.gray)
  doc.text('0', barX, barY + 16)
  doc.text('100', barX + barW - 6, barY + 16)

  y += 46

  // ── MODALITY RESULTS ───────────────────────────────────
  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...colors.dark)
  doc.text('Modality Results', margin, y)
  y += 8

  const modalities = [
    { key:'speech',      label:'Speech Analysis', tag:'[SPEECH]', color:colors.primary  },
    { key:'gait',        label:'Gait Analysis',   tag:'[GAIT]',   color:colors.green    },
    { key:'handwriting', label:'Handwriting',     tag:'[WRITE]',  color:[124,58,237]    },
  ]

  const cardW = (W - margin*2 - 10) / 3

  modalities.forEach((mod, i) => {
    const cx = margin + i * (cardW + 5)
    const result = results.modality_results?.[mod.key]
    const notProvided = !result || result.label === 'Not provided'

    doc.setFillColor(notProvided ? 249 : 255, notProvided ? 250 : 255, notProvided ? 251 : 255)
    doc.roundedRect(cx, y, cardW, 52, 4, 4, 'F')
    doc.setDrawColor(...(notProvided ? [229,231,235] : mod.color))
    doc.setLineWidth(notProvided ? 0.3 : 0.8)
    doc.roundedRect(cx, y, cardW, 52, 4, 4, 'S')
    doc.setLineWidth(0.2)

    doc.setFontSize(10)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(...(notProvided ? colors.gray : colors.dark))
    doc.text(mod.label, cx + 5, y + 10)

    if (notProvided) {
      doc.setFontSize(9)
      doc.setFont('helvetica', 'normal')
      doc.setTextColor(...colors.gray)
      doc.text('Not provided', cx + 5, y + 24)
      doc.text('No file uploaded', cx + 5, y + 33)
    } else {
      const pc = predColor(result.label)
      doc.setFillColor(...pc)
      doc.roundedRect(cx + cardW - 22, y + 4, 17, 8, 2, 2, 'F')
      doc.setFontSize(8)
      doc.setFont('helvetica', 'bold')
      doc.setTextColor(...colors.white)
      doc.text(result.label || '', cx + cardW - 13.5, y + 9.5, { align:'center' })

      doc.setFontSize(18)
      doc.setFont('helvetica', 'bold')
      doc.setTextColor(...pc)
      doc.text(`${((result.confidence||0)*100).toFixed(1)}%`, cx + 5, y + 28)

      doc.setFontSize(8)
      doc.setFont('helvetica', 'normal')
      doc.setTextColor(...colors.gray)
      doc.text('Confidence', cx + 5, y + 34)

      const probs = result.probabilities || {}
      const probColors = { HC: colors.green, RBD: colors.amber, PD: colors.red }
      let py = y + 40
      Object.entries(probs).forEach(([cls, val]) => {
        doc.setFontSize(7)
        doc.setTextColor(...colors.gray)
        doc.text(cls, cx + 5, py + 3)
        doc.setFillColor(229, 231, 235)
        doc.roundedRect(cx + 14, py, cardW - 26, 4, 1, 1, 'F')
        doc.setFillColor(...(probColors[cls] || mod.color))
        doc.roundedRect(cx + 14, py, (val * (cardW - 26)), 4, 1, 1, 'F')
        doc.setFontSize(7)
        doc.setTextColor(...colors.dark)
        doc.text(`${(val*100).toFixed(0)}%`, cx + cardW - 7, py + 3.5)
        py += 7
      })
    }
  })

  y += 62

  // ── FUSION WEIGHTS ─────────────────────────────────────
  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...colors.dark)
  doc.text('Fusion Details', margin, y)
  y += 8

  doc.setFillColor(...colors.lightgray)
  doc.roundedRect(margin, y, W - margin*2, 28, 4, 4, 'F')

  const weights = results.fusion_weights || {}
  const wLabels = { speech:'Speech', gait:'Gait', handwriting:'Handwriting' }
  const wColors = { speech:colors.primary, gait:colors.green, handwriting:[124,58,237] }
  let wx = margin + 8
  Object.entries(weights).forEach(([key, val]) => {
    doc.setFontSize(9)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(...(wColors[key] || colors.gray))
    doc.text(wLabels[key] || key, wx, y + 10)
    doc.setFontSize(14)
    doc.text(`${(val*100).toFixed(0)}%`, wx, y + 22)
    wx += 58
  })

  doc.setFontSize(9)
  doc.setFont('helvetica', 'normal')
  doc.setTextColor(...colors.gray)
  doc.text(`Processing time: ${results.processing_time_ms?.toFixed(0) || '--'}ms`, W - margin - 50, y + 16)

  y += 38

  // ── RECOMMENDATION ─────────────────────────────────────
  doc.setFontSize(13)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...colors.dark)
  doc.text('Clinical Recommendation', margin, y)
  y += 8

  const recBg = riskColor.map(c => Math.min(255, c + 190))
  doc.setFillColor(...recBg)
  doc.roundedRect(margin, y, W - margin*2, 30, 4, 4, 'F')
  doc.setDrawColor(...riskColor)
  doc.setLineWidth(0.5)
  doc.roundedRect(margin, y, W - margin*2, 30, 4, 4, 'S')
  doc.setLineWidth(0.2)

  doc.setFontSize(10)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...riskColor)
  doc.text(results.risk_level || '', margin + 6, y + 10)

  doc.setFont('helvetica', 'normal')
  doc.setTextColor(...colors.dark)
  doc.setFontSize(9)
  const recLines = doc.splitTextToSize(results.recommendation || '', W - margin*2 - 12)
  doc.text(recLines, margin + 6, y + 18)

  y += 40

  // ── DISCLAIMER ─────────────────────────────────────────
  doc.setFillColor(254, 242, 242)
  doc.roundedRect(margin, y, W - margin*2, 18, 4, 4, 'F')
  doc.setFontSize(8)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...colors.red)
  doc.text('DISCLAIMER', margin + 6, y + 7)
  doc.setFont('helvetica', 'normal')
  doc.setTextColor(...colors.gray)
  doc.text(
    'This report is generated by an AI screening tool for research purposes only and does not constitute a clinical diagnosis.',
    margin + 6, y + 13
  )

  y += 26

  // ── FOOTER ─────────────────────────────────────────────
  doc.setDrawColor(...colors.border)
  doc.line(margin, y, W - margin, y)
  y += 6
  doc.setFontSize(8)
  doc.setTextColor(...colors.gray)
  doc.text('NeuroDetect AI v1.0  |  For research purposes only  |  Not a clinical diagnostic tool', W/2, y, { align:'center' })

  // ── SAVE ───────────────────────────────────────────────
  const filename = `NeuroDetect_Report_${now.toISOString().slice(0,10)}_${now.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit'}).replace(':','')}.pdf`
  doc.save(filename)
}

export default generatePDF  