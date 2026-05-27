// ═══════════════════════════════════════════════════════════
//  main.js — TrafficNet Dashboard Interactive Logic
// ═══════════════════════════════════════════════════════════

'use strict';

// ── State ─────────────────────────────────────────────────
let currentFile       = null;
let confidenceChart   = null;
let accChart          = null;
let lossChart         = null;
let allClasses        = [];

// ── Init ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkModelStatus();
  loadHistory();
  loadClasses();
  loadModelStats();
});

// ═══════════════════════════════════════════════════════════
//  TAB NAVIGATION
// ═══════════════════════════════════════════════════════════

const TAB_META = {
  detect:    { title: 'Traffic Sign Detection',      sub: 'Upload an image to identify road signs using AI' },
  history:   { title: 'Prediction History',          sub: 'All past detections with timestamps' },
  analytics: { title: 'Model Analytics',             sub: 'Training curves and performance metrics' },
  classes:   { title: 'All 43 Traffic Sign Classes', sub: 'Browse the GTSRB dataset classes' },
  about:     { title: 'About TrafficNet',            sub: 'Architecture, tech stack, and usage guide' },
};

function switchTab(tab) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  document.getElementById(`tab-${tab}`).classList.add('active');
  document.getElementById(`nav-${tab}`).classList.add('active');
  const meta = TAB_META[tab] || {};
  document.getElementById('page-title').textContent    = meta.title || '';
  document.getElementById('page-subtitle').textContent = meta.sub   || '';
  if (tab === 'history')   loadHistory();
  if (tab === 'analytics') loadModelStats();
  if (tab === 'classes')   renderClasses(allClasses);
}

// ═══════════════════════════════════════════════════════════
//  MODEL STATUS
// ═══════════════════════════════════════════════════════════

async function checkModelStatus() {
  try {
    const res  = await fetch('/model-status');
    const data = await res.json();
    const dot  = document.getElementById('status-dot');
    const lbl  = document.getElementById('status-label');
    if (data.model_loaded) {
      dot.className = 'model-status-dot online';
      lbl.textContent = 'Model Online';
    } else if (data.model_exists) {
      dot.className = 'model-status-dot';
      dot.style.background = '#FFA502';
      lbl.textContent = 'Model Found';
    } else {
      dot.className = 'model-status-dot offline';
      lbl.textContent = 'Train Required';
    }
  } catch {
    document.getElementById('status-label').textContent = 'Server Offline';
  }
}

// ═══════════════════════════════════════════════════════════
//  FILE UPLOAD & DRAG-DROP
// ═══════════════════════════════════════════════════════════

function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.add('drag-over');
}
function handleDragLeave(e) {
  document.getElementById('drop-zone').classList.remove('drag-over');
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) loadPreview(file);
}
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) loadPreview(file);
}

function loadPreview(file) {
  if (!file.type.startsWith('image/')) {
    showToast('⚠️ Please upload an image file.');
    return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = document.getElementById('preview-img');
    img.src = e.target.result;
    img.classList.remove('hidden');
    document.getElementById('drop-zone-content').classList.add('hidden');
    document.getElementById('analyze-btn').disabled = false;
  };
  reader.readAsDataURL(file);
  resetResults();
}

function clearUpload() {
  currentFile = null;
  document.getElementById('preview-img').classList.add('hidden');
  document.getElementById('preview-img').src = '';
  document.getElementById('drop-zone-content').classList.remove('hidden');
  document.getElementById('analyze-btn').disabled = true;
  document.getElementById('file-input').value = '';
  resetResults();
}

// ═══════════════════════════════════════════════════════════
//  ANALYZE IMAGE
// ═══════════════════════════════════════════════════════════

async function analyzeImage() {
  if (!currentFile) return;

  setLoadingState(true);

  const formData = new FormData();
  formData.append('image', currentFile);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok || !data.success) {
      showToast(`❌ ${data.error || 'Prediction failed'}`);
      setLoadingState(false);
      return;
    }

    renderResult(data);
    renderGradCAM(data);
    renderConfidenceChart(data.top_k);
    loadHistory();
    checkModelStatus();
    showToast(`✅ Detected: ${data.class_name} (${data.confidence.toFixed(1)}%)`);

  } catch (err) {
    showToast('❌ Network error. Is the server running?');
    console.error(err);
  } finally {
    setLoadingState(false);
  }
}

// ── Loading State ─────────────────────────────────────────
function setLoadingState(on) {
  const loading = document.getElementById('loading-state');
  const empty   = document.getElementById('empty-state');
  const result  = document.getElementById('result-content');
  if (on) {
    loading.classList.remove('hidden');
    empty.classList.add('hidden');
    result.classList.add('hidden');
  } else {
    loading.classList.add('hidden');
  }
}

function resetResults() {
  document.getElementById('empty-state').classList.remove('hidden');
  document.getElementById('result-content').classList.add('hidden');
  document.getElementById('loading-state').classList.add('hidden');
  document.getElementById('gradcam-empty').classList.remove('hidden');
  document.getElementById('gradcam-images').classList.add('hidden');
  document.getElementById('chart-empty').classList.remove('hidden');
  if (confidenceChart) { confidenceChart.destroy(); confidenceChart = null; }
}

// ═══════════════════════════════════════════════════════════
//  RENDER PREDICTION RESULT
// ═══════════════════════════════════════════════════════════

function renderResult(data) {
  document.getElementById('empty-state').classList.add('hidden');
  document.getElementById('result-content').classList.remove('hidden');

  // Badge
  const badge = document.getElementById('prediction-badge');
  badge.style.borderColor = data.category_color;
  badge.style.boxShadow   = `0 0 16px ${data.category_color}22`;

  document.getElementById('pred-category').textContent  = data.category.toUpperCase();
  document.getElementById('pred-category').style.color  = data.category_color;
  document.getElementById('pred-name').textContent       = data.class_name;
  document.getElementById('pred-conf-value').textContent = `${data.confidence.toFixed(1)}%`;
  document.getElementById('pred-conf-value').style.color = data.category_color;

  const bar = document.getElementById('pred-conf-bar');
  bar.style.width      = '0%';
  bar.style.background = `linear-gradient(90deg, ${data.category_color}99, ${data.category_color})`;
  setTimeout(() => { bar.style.width = `${data.confidence}%`; }, 50);

  // Top-K list
  const list = document.getElementById('top-k-list');
  list.innerHTML = '';
  data.top_k.forEach((item, i) => {
    const div = document.createElement('div');
    div.className = 'top-k-item';
    div.innerHTML = `
      <div class="top-k-rank">${item.rank}</div>
      <div class="top-k-name">${item.class_name}</div>
      <div class="top-k-bar-wrap">
        <div class="top-k-bar" style="width:${item.confidence}%;background:${item.color}"></div>
      </div>
      <div class="top-k-conf" style="color:${item.color}">${item.confidence.toFixed(1)}%</div>
    `;
    list.appendChild(div);
  });
}

// ═══════════════════════════════════════════════════════════
//  GRAD-CAM
// ═══════════════════════════════════════════════════════════

function renderGradCAM(data) {
  document.getElementById('gradcam-empty').classList.add('hidden');
  document.getElementById('gradcam-images').classList.remove('hidden');

  if (data.image_b64) {
    document.getElementById('gradcam-original').src = `data:image/png;base64,${data.image_b64}`;
  }
  if (data.gradcam_b64) {
    document.getElementById('gradcam-overlay').src = `data:image/png;base64,${data.gradcam_b64}`;
  } else {
    document.getElementById('gradcam-overlay').src = document.getElementById('gradcam-original').src;
  }
}

// ═══════════════════════════════════════════════════════════
//  CONFIDENCE CHART (Chart.js)
// ═══════════════════════════════════════════════════════════

function renderConfidenceChart(topK) {
  document.getElementById('chart-empty').classList.add('hidden');

  if (confidenceChart) confidenceChart.destroy();

  const ctx    = document.getElementById('confidence-chart').getContext('2d');
  const labels = topK.map(i => i.class_name.length > 22 ? i.class_name.slice(0, 22) + '…' : i.class_name);
  const values = topK.map(i => i.confidence);
  const colors = topK.map(i => i.color);

  confidenceChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Confidence (%)',
        data: values,
        backgroundColor: colors.map(c => c + '99'),
        borderColor:     colors,
        borderWidth: 2,
        borderRadius: 8,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 800, easing: 'easeOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: ctx => ` ${ctx.parsed.x.toFixed(2)}%` }
        }
      },
      scales: {
        x: {
          max: 100,
          grid:  { color: '#30363D' },
          ticks: { color: '#8B949E', callback: v => v + '%' }
        },
        y: { grid: { display: false }, ticks: { color: '#C9D1D9', font: { size: 11 } } }
      }
    }
  });
}

// ═══════════════════════════════════════════════════════════
//  HISTORY
// ═══════════════════════════════════════════════════════════

async function loadHistory() {
  try {
    const res  = await fetch('/history?n=50');
    const data = await res.json();
    document.getElementById('history-count').textContent = data.total || 0;
    renderHistoryList(data.history || []);
  } catch {}
}

function renderHistoryList(items) {
  const list = document.getElementById('history-list');
  if (!items.length) {
    list.innerHTML = `<div class="empty-state"><div class="empty-icon">📋</div><p>No predictions yet.</p></div>`;
    return;
  }
  list.innerHTML = items.map(item => `
    <div class="history-item">
      <div class="history-cat-dot" style="background:${getCategoryColor(item.category)}"></div>
      <div class="history-name">${item.class_name}</div>
      <div class="history-conf" style="color:${getCategoryColor(item.category)}">${item.confidence.toFixed(1)}%</div>
      <div class="history-time">${formatTime(item.timestamp)}</div>
    </div>
  `).join('');
}

async function clearHistory() {
  try {
    await fetch('/clear-history', { method: 'POST' });
    document.getElementById('history-count').textContent = '0';
    renderHistoryList([]);
    showToast('🗑️ History cleared.');
  } catch { showToast('❌ Failed to clear history.'); }
}

// ═══════════════════════════════════════════════════════════
//  MODEL ANALYTICS
// ═══════════════════════════════════════════════════════════

async function loadModelStats() {
  try {
    const res  = await fetch('/model-stats');
    const data = await res.json();

    if (!data.available) {
      document.getElementById('analytics-acc').textContent = 'N/A';
      document.getElementById('analytics-loss').textContent = 'N/A';
      document.getElementById('analytics-epochs').textContent = 'N/A';
      document.getElementById('analytics-params').textContent = 'N/A';
      return;
    }

    document.getElementById('analytics-acc').textContent    = `${data.best_val_accuracy.toFixed(2)}%`;
    document.getElementById('analytics-loss').textContent   = data.best_val_loss.toFixed(4);
    document.getElementById('analytics-epochs').textContent = data.epochs_trained;
    document.getElementById('analytics-params').textContent =
      data.model_info.parameters ? formatNumber(data.model_info.parameters) : 'N/A';

    document.getElementById('pill-acc').textContent = `Val Acc: ${data.best_val_accuracy.toFixed(1)}%`;

    const h = data.history;
    renderLineChart('acc-chart',  'acc-chart-empty',  h.accuracy, h.val_accuracy, 'Accuracy', '%', accChart,  c => accChart  = c);
    renderLineChart('loss-chart', 'loss-chart-empty', h.loss,     h.val_loss,     'Loss',     '',  lossChart, c => lossChart = c);
  } catch {}
}

function renderLineChart(canvasId, emptyId, trainData, valData, label, unit, chartRef, setter) {
  document.getElementById(emptyId).classList.add('hidden');
  if (chartRef) chartRef.destroy();
  const ctx    = document.getElementById(canvasId).getContext('2d');
  const epochs = trainData.map((_, i) => `Epoch ${i + 1}`);
  const scale  = unit === '%' ? { callback: v => (v * 100).toFixed(1) + '%' } : { callback: v => v.toFixed(3) };
  setter(new Chart(ctx, {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        { label: `Train ${label}`, data: trainData, borderColor: '#58A6FF', backgroundColor: 'rgba(88,166,255,0.08)', borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4 },
        { label: `Val ${label}`,   data: valData,   borderColor: '#3FB950', backgroundColor: 'rgba(63,185,80,0.08)',  borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4, borderDash: [5,3] }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 600 },
      plugins: { legend: { labels: { color: '#8B949E', font: { size: 11 } } } },
      scales: {
        x: { grid: { color: '#30363D' }, ticks: { color: '#484F58', maxTicksLimit: 10 } },
        y: { grid: { color: '#30363D' }, ticks: { color: '#8B949E', ...scale } }
      }
    }
  }));
}

// ═══════════════════════════════════════════════════════════
//  ALL CLASSES
// ═══════════════════════════════════════════════════════════

async function loadClasses() {
  try {
    const res  = await fetch('/class-info');
    const data = await res.json();
    allClasses = data.classes || [];
    renderClasses(allClasses);
  } catch {}
}

function renderClasses(classes) {
  const grid = document.getElementById('classes-grid');
  if (!classes.length) {
    grid.innerHTML = `<div class="empty-state"><div class="spinner"></div><p>Loading...</p></div>`;
    return;
  }
  grid.innerHTML = classes.map(cls => `
    <div class="class-item" data-category="${cls.category}" style="border-color:${cls.color}22">
      <div class="class-id">Class #${String(cls.id).padStart(2,'0')}</div>
      <div class="class-name">${cls.name}</div>
      <span class="class-cat-badge" style="background:${cls.color}22;color:${cls.color}">${cls.category}</span>
    </div>
  `).join('');
}

function filterClasses(cat, btn) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const filtered = cat === 'all' ? allClasses : allClasses.filter(c => c.category === cat);
  renderClasses(filtered);
}

// ═══════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════

function getCategoryColor(cat) {
  const map = { prohibitory: '#FF4757', mandatory: '#2ED573', warning: '#FFA502', priority: '#1E90FF' };
  return map[cat] || '#A4B0BE';
}

function formatTime(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch { return iso; }
}

function formatNumber(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

// ── Toast Notifications ───────────────────────────────────
let toastTimer = null;
function showToast(msg, duration = 3500) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), duration);
}
