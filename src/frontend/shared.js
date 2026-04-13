/* shared.js — SkyGate shared utilities */

const API = window.location.protocol === 'file:' || window.location.port !== '5000'
  ? 'http://localhost:5000'
  : window.location.origin;

/* ── Active nav link ─────────────────────────────────────────── */
(function() {
  const page = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-links a').forEach(a => {
    const href = a.getAttribute('href').split('/').pop();
    if (href === page) a.classList.add('active');
  });
})();

/* ── Nav scroll tint ─────────────────────────────────────────── */
window.addEventListener('scroll', () => {
  document.getElementById('main-nav').classList.toggle('scrolled', window.scrollY > 40);
}, { passive: true });

/* ── Theme toggle ─────────────────────────────────────────────── */
function toggleTheme() {
  const html = document.documentElement;
  const btn = document.getElementById('theme-btn');
  const ripple = document.getElementById('theme-ripple');
  const isDark = html.getAttribute('data-theme') !== 'light';
  const rect = btn.getBoundingClientRect();
  ripple.style.setProperty('--rx', (rect.left + rect.width / 2) + 'px');
  ripple.style.setProperty('--ry', (rect.top + rect.height / 2) + 'px');
  ripple.style.background = isDark ? '#F2EEE8' : '#04040A';
  ripple.className = '';
  void ripple.offsetWidth;
  ripple.className = 'expanding';
  html.classList.add('theme-transitioning');
  setTimeout(() => { html.setAttribute('data-theme', isDark ? 'light' : 'dark'); }, 320);
  setTimeout(() => {
    ripple.classList.add('fading');
    html.classList.remove('theme-transitioning');
    setTimeout(() => { ripple.className = ''; }, 320);
  }, 700);
}

/* ── Stats ───────────────────────────────────────────────────── */
function animateNumber(el, target) {
  if (!el) return;
  const start = parseFloat(el.textContent.replace(/,/g, '')) || 0;
  const dur = 700, t0 = performance.now();
  const isFloat = !Number.isInteger(target);
  requestAnimationFrame(function tick(t) {
    const p = Math.min((t - t0) / dur, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    const val = start + (target - start) * ease;
    el.textContent = isFloat ? val.toFixed(1) : Math.round(val).toLocaleString();
    if (p < 1) requestAnimationFrame(tick);
  });
}

function updateStats(d) {
  const ids = { 'stat-flights': d.flights_today, 'stat-anomalies': d.anomalies_flagged, 'stat-acc': d.accuracy, 'stat-lat': d.avg_latency_ms };
  for (const [id, val] of Object.entries(ids)) {
    if (val === undefined) continue;
    const el = document.getElementById(id);
    if (el) animateNumber(el, typeof val === 'string' ? parseFloat(val) : val);
  }
}

async function loadStats() {
  try {
    const r = await fetch(`${API}/api/stats`);
    if (!r.ok) return;
    updateStats(await r.json());
  } catch (e) {}
}

/* ── Leaderboard ─────────────────────────────────────────────── */
const AIRLINES = {
  'IGO':'IndiGo','AIC':'Air India','SEJ':'SpiceJet','VTI':'Vistara','AXB':'Air India Express',
  'LLR':'Alliance Air','AKJ':'Akasa Air','UAE':'Emirates','QTR':'Qatar Airways','THA':'Thai Airways',
  'BAW':'British Airways','SIA':'Singapore Airlines','EIN':'Aer Lingus','AAL':'American Airlines',
  'UAL':'United Airlines','DAL':'Delta Air Lines'
};
const lbTally = { airlines:{}, flights:{}, aircrafts:{}, total:0 };
const seenAnomalies = new Set();

function getMockAircraft(icao) {
  if (!icao || icao === '??????') return 'Unknown Type';
  const types = ["Airbus A320","Airbus A321","Boeing 737-800","Boeing 737 MAX 8","Airbus A350","Boeing 777-300ER","ATR 72-600","Bombardier Q400"];
  let hash = 0;
  for (let i = 0; i < icao.length; i++) hash += icao.charCodeAt(i);
  return types[hash % types.length];
}

function processLeaderboard(rows) {
  let changed = false;
  rows.forEach(a => {
    const id = a.icao24 + '_' + a.timestamp + '_' + a.anomaly;
    if (!seenAnomalies.has(id)) {
      seenAnomalies.add(id); lbTally.total++; changed = true;
      const callsign = (a.callsign && a.callsign !== 'N/A') ? a.callsign : a.icao24.toUpperCase();
      const prefix = callsign.substring(0, 3).toUpperCase();
      const airline = AIRLINES[prefix] || 'Private / Unassigned';
      const flightName = `${callsign} (${airline})`;
      const aircraft = getMockAircraft(a.icao24);
      lbTally.airlines[airline] = (lbTally.airlines[airline] || 0) + 1;
      lbTally.flights[flightName] = (lbTally.flights[flightName] || 0) + 1;
      lbTally.aircrafts[aircraft] = (lbTally.aircrafts[aircraft] || 0) + 1;
    }
  });
  if (changed && lbTally.total > 0) renderLeaderboard();
}

function renderLeaderboard() {
  const sel = document.getElementById('leaderboard-select');
  if (!sel) return;
  const mode = sel.value;
  const data = lbTally[mode];
  if (!data || lbTally.total === 0) return;
  const sorted = Object.entries(data).sort((a, b) => b[1] - a[1]).slice(0, 5);
  const colors = ['var(--red)','var(--yellow)','var(--accent)','var(--green)','var(--text-dim)'];
  const html = sorted.map((item, i) => {
    const count = item[1];
    const pct = Math.round((count / lbTally.total) * 100) || '<1';
    const color = colors[i] || 'var(--text-dim)';
    return `<div class="lb-item" style="animation-delay:${i*60}ms">
      <div class="lb-info">
        <span class="lb-name">${item[0]}</span>
        <span class="lb-stat"><span style="color:${color};font-weight:700">${count}</span> (${pct}%)</span>
      </div>
      <div class="lb-bar-bg"><div class="lb-bar-fill" style="width:${Math.min(pct,100)}%;background:${color};color:${color}"></div></div>
    </div>`;
  }).join('');
  const el = document.getElementById('leaderboard-body');
  if (el) el.innerHTML = html;
}

/* ── SSE ─────────────────────────────────────────────────────── */
const LVL_CLASS = { info:'level-info', warn:'level-warn', alert:'level-alert', ok:'level-ok', error:'level-error' };
const LVL_LINE  = { info:'log-info', warn:'log-warn', alert:'log-alert', ok:'log-ok', error:'log-error' };
const LVL_FLASH = { warn:'log-flash-warn', alert:'log-flash-alert', error:'log-flash-alert' };
let logIndex = 0, packetCount = 0;

function appendLog(d) {
  const logOut = document.getElementById('log-output');
  if (!logOut) return;
  const level = (d.level || 'info').toLowerCase();
  const lineClass = LVL_LINE[level] || 'log-info';
  const flashClass = LVL_FLASH[level] || (level === 'info' ? 'log-flash' : '');
  const row = document.createElement('div');
  row.className = `log-line ${lineClass} ${flashClass}`;
  row.style.animationDelay = `${(logIndex++ % 3) * 8}ms`;
  row.innerHTML = `<span class="log-time">${d.time}</span><span class="log-level ${LVL_CLASS[level] || 'level-info'}">${level.toUpperCase()}</span><span class="log-msg">${d.msg}</span><span class="log-value">${d.val || ''}</span>`;
  logOut.appendChild(row);
  while (logOut.children.length > 80) logOut.removeChild(logOut.firstChild);
  const nearBottom = logOut.scrollHeight - logOut.scrollTop - logOut.clientHeight < 80;
  if (nearBottom) logOut.scrollTop = logOut.scrollHeight;
  packetCount++;
  const pktEl = document.getElementById('packet-count');
  if (pktEl) { pktEl.textContent = `${packetCount} PKT`; pktEl.classList.remove('pop'); void pktEl.offsetWidth; pktEl.classList.add('pop'); }
}

function connectSSE() {
  const es = new EventSource(`${API}/api/stream`);
  const connEl   = document.getElementById('conn-text');
  const streamEl = document.getElementById('stream-status');
  const pill     = document.getElementById('connection-indicator');
  es.addEventListener('log',   e => appendLog(JSON.parse(e.data)));
  es.addEventListener('stats', e => { updateStats(JSON.parse(e.data)); if (typeof loadAnomalies === 'function') loadAnomalies(); });
  es.onopen = () => {
    if (connEl)   connEl.textContent   = 'LIVE · INDIA AIRSPACE';
    if (streamEl) streamEl.textContent = 'RUNNING';
    if (pill)     pill.classList.remove('offline');
  };
  es.onerror = () => {
    if (connEl)   connEl.textContent   = 'RECONNECTING…';
    if (streamEl) streamEl.textContent = 'OFFLINE';
    if (pill)     pill.classList.add('offline');
    es.close(); setTimeout(connectSSE, 5000);
  };
}

/* ── Flight detail helpers ───────────────────────────────────── */
function aircraftSVG() {
  const colour = 'rgba(58,223,255,0.7)';
  return `<svg width="90" height="80" viewBox="0 0 90 80" fill="none" xmlns="http://www.w3.org/2000/svg">
    <ellipse cx="45" cy="40" rx="5" ry="28" fill="${colour}" opacity="0.9"/>
    <path d="M45 38 L8 52 L8 56 L45 46 L82 56 L82 52 Z" fill="${colour}" opacity="0.75"/>
    <path d="M45 64 L28 70 L28 72 L45 68 L62 72 L62 70 Z" fill="${colour}" opacity="0.6"/>
    <ellipse cx="20" cy="52" rx="5" ry="2.5" fill="${colour}" opacity="0.5"/>
    <ellipse cx="70" cy="52" rx="5" ry="2.5" fill="${colour}" opacity="0.5"/>
    <ellipse cx="45" cy="13" rx="3" ry="4" fill="${colour}" opacity="0.95"/>
  </svg>`;
}

function mockDetail(icao, callsign) {
  const routes = [
    {dep:'BOM',depCity:'Mumbai',arr:'DEL',arrCity:'Delhi'},
    {dep:'DEL',depCity:'Delhi',arr:'BLR',arrCity:'Bengaluru'},
    {dep:'MAA',depCity:'Chennai',arr:'HYD',arrCity:'Hyderabad'},
    {dep:'CCU',depCity:'Kolkata',arr:'BOM',arrCity:'Mumbai'},
    {dep:'AMD',depCity:'Ahmedabad',arr:'DEL',arrCity:'Delhi'},
  ];
  const types = ['A320','B737','A321','A319','B777','ATR72'];
  const manuf = {A320:'Airbus',B737:'Boeing',A321:'Airbus',A319:'Airbus',B777:'Boeing',ATR72:'ATR'};
  const ops   = ['IndiGo','Air India','SpiceJet','Vistara','GoFirst','AirAsia India'];
  const r = routes[Math.floor(Math.random()*routes.length)];
  const typ = types[Math.floor(Math.random()*types.length)];
  return { icao24:icao, callsign:callsign||icao.toUpperCase(), aircraft_type:typ, typecode:typ, manufacturer:manuf[typ]||'Unknown', registration:'VT-'+icao.toUpperCase().slice(0,3), operator:ops[Math.floor(Math.random()*ops.length)], origin_country:'India', departure_airport:r.dep, departure_city:r.depCity, arrival_airport:r.arr, arrival_city:r.arrCity, on_ground:false, baro_altitude:Math.round(8000+Math.random()*4000), velocity:Math.round(200+Math.random()*50), true_track:Math.round(Math.random()*360), vertical_rate:Math.round((Math.random()-0.5)*8), squawk:Math.random()>0.85?'7700':String(1000+Math.floor(Math.random()*7000)), wake_category:['MEDIUM','HEAVY','LIGHT'][Math.floor(Math.random()*3)] };
}

function renderDetail(el, d) {
  const onGroundBadge = d.on_ground ? `<span style="color:var(--yellow)"> On Ground</span>` : `<span style="color:var(--green)"> Airborne</span>`;
  const altFt = d.baro_altitude != null ? (d.alt_unit==='ft' ? Math.round(d.baro_altitude) : Math.round(d.baro_altitude*3.281)) : null;
  const velKts = d.velocity != null ? (d.vel_unit==='kts' ? Math.round(d.velocity) : Math.round(d.velocity*1.944)) : null;
  el.innerHTML = `
    <div class="detail-aircraft-art">${aircraftSVG(d.wake_category)}<div class="art-label">${d.typecode||d.aircraft_type||'—'}</div></div>
    <div class="detail-grid">
      <div class="detail-route">
        <div class="route-airport"><div class="route-iata">${d.departure_airport||'???'}</div><div class="route-city">${d.departure_city||'Unknown'}</div></div>
        <div class="route-line"><div class="route-dot"></div><div class="route-dash"></div><div class="route-plane">✈</div><div class="route-dash"></div><div class="route-dot"></div></div>
        <div class="route-airport" style="text-align:right"><div class="route-iata">${d.arrival_airport||'???'}</div><div class="route-city">${d.arrival_city||'Unknown'}</div></div>
      </div>
      <div class="detail-field"><div class="detail-field-label">Aircraft Type</div><div class="detail-field-value accent">${d.aircraft_type||d.typecode||'—'}</div></div>
      <div class="detail-field"><div class="detail-field-label">Manufacturer</div><div class="detail-field-value">${d.manufacturer||'—'}</div></div>
      <div class="detail-field"><div class="detail-field-label">Registration</div><div class="detail-field-value">${d.registration||'—'}</div></div>
      <div class="detail-field"><div class="detail-field-label">Operator</div><div class="detail-field-value">${d.operator||d.owner||'—'}</div></div>
      <div class="detail-field"><div class="detail-field-label">Origin Country</div><div class="detail-field-value">${d.origin_country||'—'}</div></div>
      <div class="detail-field"><div class="detail-field-label">Status</div><div class="detail-field-value">${onGroundBadge}</div></div>
      <div class="detail-field"><div class="detail-field-label">Baro Altitude</div><div class="detail-field-value accent">${altFt?altFt.toLocaleString()+' ft':(d.baro_altitude?d.baro_altitude+' m':'—')}</div></div>
      <div class="detail-field"><div class="detail-field-label">Ground Speed</div><div class="detail-field-value">${velKts?velKts+' kts':(d.velocity?d.velocity+' m/s':'—')}</div></div>
      <div class="detail-field"><div class="detail-field-label">True Heading</div><div class="detail-field-value">${d.true_track?d.true_track+'°':'—'}</div></div>
      <div class="detail-field"><div class="detail-field-label">Vertical Rate</div><div class="detail-field-value ${d.vertical_rate>0?'green':d.vertical_rate<0?'red':''}">${d.vertical_rate!==undefined&&d.vertical_rate!==null?(d.vertical_rate>0?'↑ +':'↓ ')+d.vertical_rate+' m/s':'—'}</div></div>
      <div class="detail-field"><div class="detail-field-label">Squawk</div><div class="detail-field-value ${['7500','7600','7700'].includes(d.squawk)?'red':''}">${d.squawk||'—'}${['7500','7600','7700'].includes(d.squawk)?' ⚠':''}</div></div>
      <div class="detail-field"><div class="detail-field-label">Wake Category</div><div class="detail-field-value">${d.wake_category||'—'}</div></div>
    </div>`;
}

async function loadFlightDetail(icao, callsign) {
  const loadingEl = document.getElementById(`loading-${icao}`);
  const innerEl   = document.getElementById(`inner-${icao}`);
  if (innerEl.dataset.loaded === '1') { loadingEl.style.display='none'; innerEl.style.display='grid'; return; }
  loadingEl.style.display = 'flex'; innerEl.style.display = 'none';
  let detail = null;
  try { const r = await fetch(`${API}/api/flight-detail/${icao}`); if (r.ok) detail = await r.json(); } catch(e) {}
  const mock = mockDetail(icao, callsign);
  detail = detail ? {...mock,...detail} : mock;
  renderDetail(innerEl, detail);
  loadingEl.style.display = 'none'; innerEl.style.display = 'grid'; innerEl.dataset.loaded = '1';
}

/* ── Shared nav SVG logo ─────────────────────────────────────── */
const NAV_LOGO_SVG = `<svg viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" width="28" height="28">
  <circle cx="14" cy="14" r="13" stroke="rgba(58,223,255,0.3)" stroke-width="1"/>
  <circle cx="14" cy="14" r="8" stroke="rgba(58,223,255,0.5)" stroke-width="1"/>
  <circle cx="14" cy="14" r="2.5" fill="rgba(58,223,255,0.9)"/>
  <line x1="14" y1="1" x2="14" y2="27" stroke="rgba(58,223,255,0.2)" stroke-width="0.5"/>
  <line x1="1" y1="14" x2="27" y2="14" stroke="rgba(58,223,255,0.2)" stroke-width="0.5"/>
  <line x1="14" y1="14" x2="24" y2="6" stroke="rgba(58,223,255,0.8)" stroke-width="1.2" stroke-linecap="round">
    <animateTransform attributeName="transform" attributeType="XML" type="rotate" from="0 14 14" to="360 14 14" dur="4s" repeatCount="indefinite"/>
  </line>
  <circle cx="22" cy="8" r="1.5" fill="#FF5050">
    <animate attributeName="opacity" values="0;1;0" dur="4s" repeatCount="indefinite"/>
  </circle>
</svg>`;