"""Interfaz Flask para el modelo SVM de detección de reseñas."""

import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from nlp_pipeline import limpiar_texto

logger = logging.getLogger(__name__)
MODEL_PATH = PROJECT_DIR / "modelo_svm.pkl"
VECTORIZER_PATH = PROJECT_DIR / "vectorizador.pkl"

app = Flask(__name__)
app.secret_key = 'clave-secreta-prototipo'

login_manager = LoginManager(app)
login_manager.login_view = 'login'

USUARIOS = {
    'admin': generate_password_hash('1234'),
    'demo':  generate_password_hash('demo'),
}
HISTORIAL = { u: [] for u in USUARIOS }


class Usuario(UserMixin):
    def __init__(self, username): self.id = username


@login_manager.user_loader
def load_user(username):
    return Usuario(username) if username in USUARIOS else None

modelo = None
vectorizador = None
ERROR_CARGA_MODELO = None
metricas = {
    "exactitud": "N/D",
    "precision": "N/D",
    "sensibilidad": "N/D",
    "f1": "N/D",
}
try:
    if not MODEL_PATH.is_file() or not VECTORIZER_PATH.is_file():
        raise FileNotFoundError(
            "No se encontraron modelo_svm.pkl y/o vectorizador.pkl en "
            f"{PROJECT_DIR}."
        )
    modelo = joblib.load(MODEL_PATH)
    vectorizador = joblib.load(VECTORIZER_PATH)
    if set(modelo.classes_) != {0, 1}:
        raise ValueError("El modelo no contiene las clases esperadas 0 y 1.")
    metricas = {
        "exactitud": "98.39",
        "precision": "98.39",
        "sensibilidad": "98.39",
        "f1": "98.39",
    }
except Exception as error:
    ERROR_CARGA_MODELO = error
    logger.exception("No se pudo cargar el modelo SVM y su vectorizador.")

# ── TEMPLATES ────────────────────────────────
ESTILOS_BASE = """
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  :root { --bg:#0d0f14;--surface:#161921;--border:#252a35;--accent:#00e5a0;--danger:#ff4d6d;--warn:#f59e0b;--text:#e8eaf0;--muted:#6b7280; }
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:40px 20px;}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:28px;width:100%;max-width:660px;margin-bottom:20px;}
  label{display:block;font-size:.8rem;font-family:'Space Mono',monospace;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;}
  input,textarea{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:'DM Sans',sans-serif;font-size:.95rem;padding:12px 14px;outline:none;transition:border-color .2s;margin-bottom:14px;}
  input:focus,textarea:focus{border-color:var(--accent);}
  textarea{resize:vertical;min-height:120px;margin-bottom:0;}
  .btn{width:100%;padding:13px;background:var(--accent);color:#0d0f14;font-family:'Space Mono',monospace;font-weight:700;font-size:.9rem;border:none;border-radius:8px;cursor:pointer;transition:opacity .2s;margin-top:14px;}
  .btn:hover{opacity:.85;}
  .btn-ghost{background:transparent;border:1px solid var(--border);color:var(--muted);margin-top:0;}
  .btn-ghost:hover{border-color:var(--accent);color:var(--accent);opacity:1;}
  h1{font-family:'Space Mono',monospace;font-size:1.6rem;color:var(--accent);}
  .sub{color:var(--muted);font-size:.9rem;margin-top:6px;}
  .section-title{font-family:'Space Mono',monospace;font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;}
  .error{color:var(--danger);font-size:.85rem;margin-top:8px;font-family:'Space Mono',monospace;}
  footer{color:var(--muted);font-size:.75rem;margin-top:10px;font-family:'Space Mono',monospace;}
  .topbar{width:100%;max-width:660px;display:flex;justify-content:space-between;align-items:center;margin-bottom:30px;}
  .topbar span{font-family:'Space Mono',monospace;font-size:.8rem;color:var(--muted);}
  .topbar a{font-family:'Space Mono',monospace;font-size:.8rem;color:var(--danger);text-decoration:none;}
  .topbar a:hover{text-decoration:underline;}
  .veredicto{font-family:'Space Mono',monospace;font-size:1.3rem;font-weight:700;margin-bottom:6px;}
  .veredicto.falsa{color:var(--danger);} .veredicto.autentica{color:var(--accent);}
  .barra-bg{background:var(--border);border-radius:99px;height:8px;overflow:hidden;}
  .barra-fill{height:100%;border-radius:99px;transition:width .6s ease;}
  .barra-fill.falsa{background:var(--danger);} .barra-fill.autentica{background:var(--accent);}
  .pct{font-family:'Space Mono',monospace;font-size:.85rem;color:var(--muted);margin-top:5px;text-align:right;}
  .metricas-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}
  .metrica{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;text-align:center;}
  .metrica .val{font-family:'Space Mono',monospace;font-size:1.2rem;color:var(--accent);font-weight:700;}
  .metrica .nom{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-top:4px;}
  .hist-item{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:10px;}
  .hist-item:last-child{margin-bottom:0;}
  .hist-texto{font-size:.9rem;color:var(--text);margin-bottom:8px;line-height:1.5;}
  .hist-meta{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;}
  .tag{font-family:'Space Mono',monospace;font-size:.72rem;padding:3px 10px;border-radius:99px;font-weight:700;}
  .tag.falsa{background:rgba(255,77,109,.15);color:var(--danger);}
  .tag.autentica{background:rgba(0,229,160,.12);color:var(--accent);}
  .hist-fecha,.hist-conf{font-family:'Space Mono',monospace;font-size:.72rem;color:var(--muted);}
  .empty{text-align:center;color:var(--muted);font-size:.85rem;padding:20px 0;}
  .tabs{display:flex;gap:8px;margin-bottom:20px;flex-wrap:wrap;}
  .tab{font-family:'Space Mono',monospace;font-size:.75rem;padding:7px 14px;border-radius:8px;cursor:pointer;border:1px solid var(--border);color:var(--muted);background:transparent;transition:all .2s;}
  .tab.active{border-color:var(--accent);color:var(--accent);}
  .panel{display:none;} .panel.active{display:block;}
  /* Imagen mockup */
  .upload-zone{border:2px dashed var(--border);border-radius:10px;padding:28px;text-align:center;cursor:pointer;transition:border-color .2s;margin-top:16px;position:relative;}
  .upload-zone:hover{border-color:var(--accent);}
  .upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;}
  .upload-icon{font-size:2rem;margin-bottom:8px;}
  .upload-label{font-family:'Space Mono',monospace;font-size:.8rem;color:var(--muted);}
  .upload-sub{font-size:.75rem;color:var(--muted);margin-top:4px;opacity:.6;}
  .mockup-badge{display:inline-block;background:rgba(245,158,11,.12);color:var(--warn);font-family:'Space Mono',monospace;font-size:.7rem;padding:3px 10px;border-radius:99px;border:1px solid rgba(245,158,11,.25);margin-left:8px;}
  .img-preview{display:none;margin-top:14px;border-radius:8px;overflow:hidden;border:1px solid var(--border);}
  .img-preview img{width:100%;max-height:200px;object-fit:cover;}
  .img-info{padding:10px 14px;background:var(--bg);display:flex;justify-content:space-between;align-items:center;}
  .img-name{font-family:'Space Mono',monospace;font-size:.75rem;color:var(--accent);}
  .img-remove{font-family:'Space Mono',monospace;font-size:.7rem;color:var(--danger);cursor:pointer;}
  /* Estadísticas */
  .stat-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px;}
  .stat-card{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:16px;text-align:center;}
  .stat-card .num{font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;}
  .stat-card .lbl{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-top:4px;}
  .num.total{color:var(--text);}
  .num.falsas{color:var(--danger);}
  .num.autenticas{color:var(--accent);}
  canvas{width:100%!important;}
  .chart-wrap{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:14px;}
  .chart-title{font-family:'Space Mono',monospace;font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:16px;}
  .no-stats{text-align:center;color:var(--muted);font-size:.85rem;padding:30px 0;}
</style>
"""

LOGIN_HTML = """<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<title>Login · Detector de Reseñas</title>""" + ESTILOS_BASE + """</head><body>
<div style="margin-bottom:30px;text-align:center;">
  <h1>// DETECTOR DE RESEÑAS</h1>
  <p class="sub">Inicia sesión para continuar</p>
</div>
<div class="card">
  <div class="section-title">Acceso al sistema</div>
  <label>Usuario</label><input type="text" id="usr" placeholder="usuario">
  <label>Contraseña</label><input type="password" id="pwd" placeholder="••••••••">
  <div id="err" class="error" style="display:none">Usuario o contraseña incorrectos.</div>
  <button class="btn" onclick="login()">INICIAR SESIÓN →</button>
  <p style="margin-top:14px;font-size:.8rem;color:var(--muted);font-family:'Space Mono',monospace;">
    Usuarios: <b style="color:var(--accent)">admin / 1234</b> · <b style="color:var(--accent)">demo / demo</b>
  </p>
</div>
<script>
  async function login(){
    const r=await fetch('/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({usuario:document.getElementById('usr').value,password:document.getElementById('pwd').value})});
    const d=await r.json();
    if(d.ok) window.location.href='/';
    else document.getElementById('err').style.display='block';
  }
  document.addEventListener('keydown',e=>{if(e.key==='Enter')login();});
</script></body></html>"""

APP_HTML = """<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<title>Detector de Reseñas Falsas</title>""" + ESTILOS_BASE + """
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head><body>

<div class="topbar">
  <h1 style="font-size:1.1rem;">// DETECTOR DE RESEÑAS</h1>
  <span>usuario: <b style="color:var(--text)">{{ usuario }}</b> &nbsp;·&nbsp; <a href="/auth/logout">cerrar sesión</a></span>
</div>

<div class="tabs" style="width:100%;max-width:660px;">
  <button class="tab active"   onclick="showTab('analizador',this)">ANALIZADOR</button>
  <button class="tab"          onclick="showTab('historial',this)">HISTORIAL</button>
  <button class="tab"          onclick="showTab('estadisticas',this)">ESTADÍSTICAS</button>
  <button class="tab"          onclick="showTab('modelo',this)">MODELO</button>
</div>

<!-- ═══════════════ TAB: ANALIZADOR ═══════════════ -->
<div id="tab-analizador" class="panel active" style="width:100%;max-width:660px;">
  <div class="card">
    <label>Reseña en texto</label>
    <textarea id="resena" placeholder="Escribe o pega aquí el texto de la reseña..."></textarea>

    <!-- Mockup subida de imagen -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-top:20px;
                border-top:1px solid var(--border);padding-top:16px;">
      <div style="display:flex;align-items:center;gap:8px;">
        <label style="margin:0;font-size:.8rem;">Analizar desde imagen</label>
        <span class="mockup-badge">PRÓXIMAMENTE</span>
      </div>
      <button onclick="toggleImagen()" id="btn-toggle-img"
              style="background:transparent;border:1px solid var(--border);color:var(--muted);
                     font-family:'Space Mono',monospace;font-size:.7rem;padding:4px 10px;
                     border-radius:6px;cursor:pointer;transition:all .2s;margin-top:0;width:auto;">
        MOSTRAR ▾
      </button>
    </div>

    <div id="panel-imagen" style="display:none;margin-top:12px;">
      <p style="font-size:.78rem;color:var(--muted);margin-bottom:10px;">
        Captura de pantalla de una reseña en plataformas como Amazon, Google o MercadoLibre.
        El sistema extraerá el texto automáticamente mediante OCR.
      </p>
      <div class="upload-zone" onclick="document.getElementById('img-input').click()">
        <input type="file" id="img-input" accept="image/*" onchange="previsualizarImagen(event)">
        <div class="upload-icon">🖼️</div>
        <div class="upload-label">Arrastra una imagen o haz clic para seleccionar</div>
        <div class="upload-sub">PNG, JPG, WEBP · máx. 5 MB</div>
      </div>
      <div class="img-preview" id="img-preview">
        <img id="img-thumb" src="" alt="preview">
        <div class="img-info">
          <span class="img-name" id="img-name"></span>
          <span class="img-remove" onclick="quitarImagen()">✕ quitar</span>
        </div>
      </div>
      <div id="img-mockup-msg" style="display:none;margin-top:12px;padding:12px 14px;
           background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.2);border-radius:8px;
           font-size:.82rem;color:var(--warn);font-family:'Space Mono',monospace;">
        ⚠ Funcionalidad OCR no disponible en el prototipo actual. Ingresa el texto manualmente.
      </div>
    </div>

    <button class="btn" onclick="analizar()">ANALIZAR RESEÑA →</button>
  </div>

  <div class="card" id="resultado" style="display:none">
    <div class="section-title">Resultado del análisis</div>
    <div class="veredicto" id="veredicto-texto"></div>
    <div id="descripcion" style="color:var(--muted);font-size:.9rem;margin-bottom:16px;"></div>
    <div style="font-size:.8rem;color:var(--muted);font-family:'Space Mono',monospace;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Clasificación SVM · el margen no es una probabilidad</div>
    <div class="barra-bg"><div class="barra-fill" id="barra" style="width:0%"></div></div>
    <div class="pct" id="pct-texto"></div>
  </div>
</div>

<!-- ═══════════════ TAB: HISTORIAL ═══════════════ -->
<div id="tab-historial" class="panel" style="width:100%;max-width:660px;">
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
      <div class="section-title" style="margin:0;">Reseñas analizadas</div>
      <button class="btn btn-ghost" style="width:auto;padding:6px 14px;font-size:.75rem;" onclick="limpiarHistorial()">LIMPIAR</button>
    </div>
    <div id="lista-historial"><p class="empty">Aún no has analizado ninguna reseña.</p></div>
  </div>
</div>

<!-- ═══════════════ TAB: ESTADÍSTICAS ═══════════════ -->
<div id="tab-estadisticas" class="panel" style="width:100%;max-width:660px;">

  <!-- Tarjetas resumen -->
  <div class="stat-cards">
    <div class="stat-card"><div class="num total" id="st-total">0</div><div class="lbl">Total analizadas</div></div>
    <div class="stat-card"><div class="num falsas" id="st-falsas">0</div><div class="lbl">Falsas detectadas</div></div>
    <div class="stat-card"><div class="num autenticas" id="st-autenticas">0</div><div class="lbl">Auténticas</div></div>
  </div>

  <div id="stats-empty" class="card" style="display:none">
    <p class="no-stats">📊 Analiza al menos una reseña para ver las estadísticas.</p>
  </div>

  <div id="stats-content" style="display:none">
    <!-- Donut: distribución -->
    <div class="chart-wrap">
      <div class="chart-title">Distribución de clasificaciones</div>
      <canvas id="chart-donut" height="220"></canvas>
    </div>

    <!-- Barras: margen SVM medio -->
    <div class="chart-wrap">
      <div class="chart-title">Margen absoluto promedio por clase (no es probabilidad)</div>
      <canvas id="chart-barras" height="160"></canvas>
    </div>

    <!-- Línea: evolución temporal -->
    <div class="chart-wrap">
      <div class="chart-title">Evolución de análisis (últimas 10)</div>
      <canvas id="chart-linea" height="180"></canvas>
    </div>
  </div>
</div>

<!-- ═══════════════ TAB: MODELO ═══════════════ -->
<div id="tab-modelo" class="panel" style="width:100%;max-width:660px;">
  <div class="card">
    <div class="section-title">Métricas del modelo (conjunto de prueba)</div>
    <div class="metricas-grid">
      <div class="metrica"><div class="val">{{ metricas.exactitud }}{% if metricas.exactitud != 'N/D' %}%{% endif %}</div><div class="nom">Exactitud</div></div>
      <div class="metrica"><div class="val">{{ metricas.precision }}{% if metricas.precision != 'N/D' %}%{% endif %}</div><div class="nom">Precisión macro</div></div>
      <div class="metrica"><div class="val">{{ metricas.sensibilidad }}{% if metricas.sensibilidad != 'N/D' %}%{% endif %}</div><div class="nom">Recall macro</div></div>
      <div class="metrica"><div class="val">{{ metricas.f1 }}{% if metricas.f1 != 'N/D' %}%{% endif %}</div><div class="nom">F1 macro</div></div>
    </div>
  </div>
  <div class="card">
    <div class="section-title">Pipeline del modelo</div>
    <p style="font-family:'Space Mono',monospace;font-size:.8rem;color:var(--muted);line-height:2.2;">
      Texto original<br>↓ limpiar_texto (SpaCy: lemas y stopwords)<br>
      ↓ Vectorizador TF-IDF entrenado (unigramas y bigramas)
      <br>↓ LinearSVC entrenado<br>
      <span style="color:var(--accent)">→ Clasificación: real (0) / falsa (1)</span>
    </p>
  </div>
</div>

<footer>prototipo académico · TF-IDF + LinearSVC · español</footer>

<!-- ═══════════════ SCRIPTS ═══════════════ -->
<script>
  let chartDonut=null, chartBarras=null, chartLinea=null;

  function showTab(name, el){
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
    document.getElementById('tab-'+name).classList.add('active');
    el.classList.add('active');
    if(name==='historial')   cargarHistorial();
    if(name==='estadisticas') cargarEstadisticas();
  }

  /* ── Imagen mockup ── */
  function toggleImagen(){
    const panel=document.getElementById('panel-imagen');
    const btn=document.getElementById('btn-toggle-img');
    const visible=panel.style.display==='block';
    panel.style.display=visible?'none':'block';
    btn.textContent=visible?'MOSTRAR ▾':'OCULTAR ▴';
    btn.style.borderColor=visible?'var(--border)':'var(--accent)';
    btn.style.color=visible?'var(--muted)':'var(--accent)';
  }
  function previsualizarImagen(e){
    const file=e.target.files[0]; if(!file) return;
    const reader=new FileReader();
    reader.onload=ev=>{
      document.getElementById('img-thumb').src=ev.target.result;
      document.getElementById('img-name').textContent=file.name;
      document.getElementById('img-preview').style.display='block';
      document.getElementById('img-mockup-msg').style.display='block';
    };
    reader.readAsDataURL(file);
  }
  function quitarImagen(){
    document.getElementById('img-input').value='';
    document.getElementById('img-preview').style.display='none';
    document.getElementById('img-mockup-msg').style.display='none';
  }

  /* ── Analizador ── */
  async function analizar(){
    const texto=document.getElementById('resena').value.trim();
    if(!texto){
      document.getElementById('resultado').style.display='block';
      document.getElementById('veredicto-texto').textContent='';
      document.getElementById('descripcion').textContent='Escribe una reseña antes de analizarla.';
      document.getElementById('pct-texto').textContent='';
      return;
    }
    try{
      const res=await fetch('/predecir',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({resena:texto})});
      const data=await res.json();
      if(!res.ok) throw new Error(data.error||'No se pudo analizar la reseña.');
      const cls=data.clase===1?'falsa':'autentica';
      document.getElementById('resultado').style.display='block';
      const verd=document.getElementById('veredicto-texto');
      verd.className='veredicto '+cls;
      verd.textContent=data.clase===1?'⚠ RESEÑA CLASIFICADA COMO FALSA':'✓ RESEÑA CLASIFICADA COMO REAL';
      document.getElementById('descripcion').textContent=data.clase===1
        ?'El modelo asignó la etiqueta 1 (falsa). Esta es una predicción, no una garantía.'
        :'El modelo asignó la etiqueta 0 (real). Esta es una predicción, no una garantía.';
      document.getElementById('barra').className='barra-fill '+cls;
      document.getElementById('barra').style.width='100%';
      document.getElementById('pct-texto').textContent=
        'Margen SVM: '+Number(data.margen_decision).toFixed(4)+' (no es una probabilidad)';
    }catch(error){
      document.getElementById('resultado').style.display='block';
      document.getElementById('veredicto-texto').textContent='No se pudo completar el análisis';
      document.getElementById('descripcion').textContent=error.message;
      document.getElementById('pct-texto').textContent='';
    }
  }

  /* ── Historial ── */
  async function cargarHistorial(){
    const res=await fetch('/historial');
    const data=await res.json();
    const lista=document.getElementById('lista-historial');
    if(!data.items.length){lista.innerHTML='<p class="empty">Aún no has analizado ninguna reseña.</p>';return;}
    lista.innerHTML=data.items.slice().reverse().map(item=>{
      const cls=item.clase===1?'falsa':'autentica';
      const label=item.clase===1?'⚠ FALSA':'✓ AUTÉNTICA';
      return `<div class="hist-item">
        <div class="hist-texto">${escaparHTML(item.texto)}</div>
        <div class="hist-meta">
          <span class="tag ${cls}">${label}</span>
          <span class="hist-conf">Margen SVM: ${Number(item.margen_decision).toFixed(4)} (no es probabilidad)</span>
          <span class="hist-fecha">${item.fecha}</span>
        </div></div>`;
    }).join('');
  }
  function escaparHTML(valor){
    return String(valor).replace(/[&<>"']/g, caracter=>({
      '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
    })[caracter]);
  }
  async function limpiarHistorial(){await fetch('/historial/limpiar',{method:'POST'});cargarHistorial();}

  /* ── Estadísticas ── */
  async function cargarEstadisticas(){
    const res=await fetch('/historial');
    const {items}=await res.json();

    const total=items.length;
    const falsas=items.filter(i=>i.clase===1).length;
    const autenticas=total-falsas;

    document.getElementById('st-total').textContent=total;
    document.getElementById('st-falsas').textContent=falsas;
    document.getElementById('st-autenticas').textContent=autenticas;

    if(!total){
      document.getElementById('stats-empty').style.display='block';
      document.getElementById('stats-content').style.display='none';
      return;
    }
    document.getElementById('stats-empty').style.display='none';
    document.getElementById('stats-content').style.display='block';

    const margenFalsas = falsas ? items.filter(i=>i.clase===1).reduce((a,b)=>a+Math.abs(b.margen_decision),0)/falsas : 0;
    const margenReales = autenticas ? items.filter(i=>i.clase===0).reduce((a,b)=>a+Math.abs(b.margen_decision),0)/autenticas : 0;

    const ultimas10 = items.slice(-10);
    const labelsLinea = ultimas10.map((_,i)=>'#'+(items.length-ultimas10.length+i+1));
    const dataLinea   = ultimas10.map(i=>Math.abs(i.margen_decision));
    const colsLinea   = ultimas10.map(i=>i.clase===1?'rgba(255,77,109,0.8)':'rgba(0,229,160,0.8)');

    const cfg = {
      donut:{
        type:'doughnut',
        data:{labels:['Falsas','Auténticas'],
          datasets:[{data:[falsas,autenticas],
            backgroundColor:['rgba(255,77,109,.75)','rgba(0,229,160,.75)'],
            borderColor:['#ff4d6d','#00e5a0'],borderWidth:2,hoverOffset:6}]},
        options:{plugins:{legend:{labels:{color:'#e8eaf0',font:{family:'DM Sans'}}}},cutout:'62%'}
      },
      barras:{
        type:'bar',
        data:{labels:['Falsas','Auténticas'],
          datasets:[{label:'Margen absoluto medio (no es probabilidad)',
            data:[margenFalsas, margenReales],
            backgroundColor:['rgba(255,77,109,.6)','rgba(0,229,160,.6)'],
            borderColor:['#ff4d6d','#00e5a0'],borderWidth:2,borderRadius:6}]},
        options:{plugins:{legend:{display:false}},scales:{
          y:{min:0,ticks:{color:'#6b7280'},grid:{color:'#252a35'}},
          x:{ticks:{color:'#6b7280'},grid:{display:false}}}}
      },
      linea:{
        type:'line',
        data:{labels:labelsLinea,
          datasets:[{label:'Margen absoluto SVM (no es probabilidad)',data:dataLinea,
            borderColor:'#00e5a0',backgroundColor:'rgba(0,229,160,.08)',
            pointBackgroundColor:colsLinea,pointBorderColor:colsLinea,
            pointRadius:5,tension:.35,fill:true}]},
        options:{plugins:{legend:{display:false}},scales:{
          y:{min:0,ticks:{color:'#6b7280'},grid:{color:'#252a35'}},
          x:{ticks:{color:'#6b7280'},grid:{display:false}}}}
      }
    };

    ['donut','barras','linea'].forEach(k=>{
      const canvas=document.getElementById('chart-'+k);
      const prev=k==='donut'?chartDonut:k==='barras'?chartBarras:chartLinea;
      if(prev) prev.destroy();
      const c=new Chart(canvas,cfg[k]);
      if(k==='donut')   chartDonut=c;
      if(k==='barras')  chartBarras=c;
      if(k==='linea')   chartLinea=c;
    });
  }
</script>
</body></html>"""

# ── RUTAS ────────────────────────────────────
@app.route('/')
@login_required
def index():
    return render_template_string(APP_HTML, metricas=metricas, usuario=current_user.id)

@app.route('/login')
def login():
    return render_template_string(LOGIN_HTML)

@app.route('/auth/login', methods=['POST'])
def auth_login():
    data = request.get_json()
    usr, pwd = data.get('usuario',''), data.get('password','')
    if usr in USUARIOS and check_password_hash(USUARIOS[usr], pwd):
        login_user(Usuario(usr))
        return jsonify({'ok': True})
    return jsonify({'ok': False})

@app.route('/auth/logout')
@login_required
def auth_logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/predecir', methods=['POST'])
@login_required
def predecir():
    if modelo is None or vectorizador is None:
        return jsonify({
            'error': 'El modelo SVM o el vectorizador no están disponibles.'
        }), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'La solicitud debe incluir una reseña.'}), 400

    texto = data.get('resena')
    if not isinstance(texto, str) or not texto.strip():
        return jsonify({'error': 'Escribe una reseña antes de analizarla.'}), 400

    texto = texto.strip()
    try:
        texto_limpio = limpiar_texto(texto)
        matriz_tfidf = vectorizador.transform([texto_limpio])
        clase = int(modelo.predict(matriz_tfidf)[0])
        margen_decision = float(modelo.decision_function(matriz_tfidf)[0])
    except Exception:
        logger.exception("Falló el procesamiento o la predicción de una reseña.")
        return jsonify({
            'error': 'No se pudo procesar la reseña. Revisa el modelo de idioma '
                     'de SpaCy y los archivos del modelo.'
        }), 500

    if clase not in {0, 1}:
        logger.error("El clasificador devolvió una etiqueta inesperada: %s", clase)
        return jsonify({'error': 'El modelo devolvió una clase no reconocida.'}), 500

    HISTORIAL[current_user.id].append({
        'texto': texto[:200],
        'clase': clase,
        'margen_decision': margen_decision,
        'fecha': datetime.now().strftime('%d/%m/%Y %H:%M')
    })
    return jsonify({
        'clase': clase,
        'margen_decision': margen_decision,
        'caracteristicas_tfidf': matriz_tfidf.shape[1],
    })

@app.route('/historial')
@login_required
def historial():
    return jsonify({'items': HISTORIAL[current_user.id]})

@app.route('/historial/limpiar', methods=['POST'])
@login_required
def limpiar_historial():
    HISTORIAL[current_user.id] = []
    return jsonify({'ok': True})

if __name__ == '__main__':
    if modelo is None:
        print(f"\n✗ No se pudo cargar el modelo: {ERROR_CARGA_MODELO}")
    else:
        print("\n✓ Modelo SVM y vectorizador cargados")
        print(f"  Exactitud de prueba: {metricas['exactitud']}%  F1: {metricas['f1']}%")
    print("\n→ http://localhost:5000   (admin/1234 · demo/demo)\n")
    app.run(debug=False)
