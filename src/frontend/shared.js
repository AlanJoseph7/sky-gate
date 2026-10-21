/* shared.js — SkyGate shared utilities */

/* ── Theme Initialization ────────────────────────────────────── */
(function() {
  const savedTheme = localStorage.getItem('sg_theme');
  if (savedTheme) {
    document.documentElement.setAttribute('data-theme', savedTheme);
  }
})();

// API base: use same-origin when served by backend (recommended).
// If opening HTML directly (file://), fall back to the FastAPI default port (5000).
const API = window.location.protocol === 'file:'
  ? 'http://localhost:5000'
  : window.location.origin;

/* ── Auth Utils ──────────────────────────────────────────────── */
async function sgFetch(url, options = {}) {
  const token = localStorage.getItem('sg_token');
  const headers = {
    ...options.headers,
    'Authorization': `Bearer ${token}`
  };
  const response = await fetch(url, { ...options, headers });
  if (response.status === 401 && !window.location.pathname.endsWith('auth.html')) {
    localStorage.removeItem('sg_token');
    window.location.href = 'auth.html';
  }
  return response;
}

function prefetchPages() {
  const pages = ['index.html', 'monitor.html', 'anomalies.html', 'pipeline.html', 'features.html', 'siem.html'];
  const current = window.location.pathname.split('/').pop() || 'index.html';
  pages.forEach(p => {
    if (p !== current && !document.querySelector(`link[href="${p}"]`)) {
      const link = document.createElement('link');
      link.rel = 'prefetch';
      link.href = p;
      document.head.appendChild(link);
    }
  });
}

/* ── Security: HTML Entity Escaping ─────────────────────────── */
function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
window.escapeHtml = escapeHtml;

(function() {
  const page = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-links a').forEach(a => {
    const href = a.getAttribute('href').split('/').pop();
    if (href === page) a.classList.add('active');
  });

  const navRight = document.querySelector('.nav-right');
  const user = localStorage.getItem('sg_name') || localStorage.getItem('sg_user') || 'User';
  // ✅ Guard: only inject nav user on protected pages (shared.js is never loaded on auth.html)
  if (navRight) {
    const userDiv = document.createElement('div');
    userDiv.className = 'nav-user';
    userDiv.innerHTML = `
      <span class="user-greeting">Welcome, <span class="user-name">${escapeHtml(user)}</span>!</span>
      <button onclick="signOut()" class="signout-btn" title="Sign Out">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4M16 17l5-5-5-5M21 12H9"/>
        </svg>
      </button>
    `;
    navRight.insertBefore(userDiv, navRight.firstChild);
  }

  setTimeout(prefetchPages, 1500);
})();

function signOut() {
  localStorage.removeItem('sg_token');
  localStorage.removeItem('sg_user');
  localStorage.removeItem('sg_name');
  window.location.href = 'auth.html';
}

/* ── Nav scroll tint ─────────────────────────────────────────── */
window.addEventListener('scroll', () => {
  const nav = document.getElementById('main-nav');
  if (nav) nav.classList.toggle('scrolled', window.scrollY > 40);
}, { passive: true });

/* ── Theme toggle ─────────────────────────────────────────────── */
function toggleTheme() {
  const html    = document.documentElement;
  const btn     = document.getElementById('theme-btn');
  const ripple  = document.getElementById('theme-ripple');
  const isDark  = html.getAttribute('data-theme') !== 'light';
  const rect    = btn.getBoundingClientRect();
  ripple.style.setProperty('--rx', (rect.left + rect.width  / 2) + 'px');
  ripple.style.setProperty('--ry', (rect.top  + rect.height / 2) + 'px');
  ripple.style.background = isDark ? '#F2EEE8' : '#04040A';
  ripple.className = '';
  void ripple.offsetWidth;
  ripple.className = 'expanding';
  html.classList.add('theme-transitioning');
  const newTheme = isDark ? 'light' : 'dark';
  setTimeout(() => {
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('sg_theme', newTheme);
  }, 320);
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
    const p    = Math.min((t - t0) / dur, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    const val  = start + (target - start) * ease;
    el.textContent = isFloat ? val.toFixed(1) : Math.round(val).toLocaleString();
    if (p < 1) requestAnimationFrame(tick);
  });
}

function updateStats(d) {
  window._activeFlights = d.active_flights;
  const ids = {
    'stat-flights':   d.active_flights !== undefined ? d.active_flights : d.flights_today,
    'stat-anomalies': d.anomalies_flagged,
    'stat-acc':       d.accuracy,
    'stat-lat':       d.avg_latency_ms
  };
  for (const [id, val] of Object.entries(ids)) {
    if (val === undefined) continue;
    const el = document.getElementById(id);
    if (el) {
      if (val === 'N/A') {
        el.textContent = 'N/A';
      } else {
        animateNumber(el, typeof val === 'string' ? parseFloat(val) : val);
      }
    }
  }
  // If we are on the monitor page, refresh the recent list to show empty states
  if (typeof window.renderRecent === 'function') window.renderRecent();
}

async function loadStats() {
  try {
    const r = await sgFetch(`${API}/api/stats`);
    if (!r.ok) return;
    updateStats(await r.json());
  } catch(e) {}
}

/* ── Leaderboard ─────────────────────────────────────────────── */
const AIRLINES = {
  'IGO':'IndiGo','AIC':'Air India','SEJ':'SpiceJet','VTI':'Vistara','AXB':'Air India Express',
  'LLR':'Alliance Air','AKJ':'Akasa Air','UAE':'Emirates','QTR':'Qatar Airways','THA':'Thai Airways',
  'BAW':'British Airways','SIA':'Singapore Airlines','EIN':'Aer Lingus','AAL':'American Airlines',
  'UAL':'United Airlines','DAL':'Delta Air Lines'
};
const lbTally    = { airlines:{}, flights:{}, aircrafts:{}, total:0 };
const seenAnomalies = new Set();

function getMockAircraft(icao) {
  if (!icao || icao === '??????') return 'Unknown Type';
  const types = ['Airbus A320','Airbus A321','Boeing 737-800','Boeing 737 MAX 8','Airbus A350','Boeing 777-300ER','ATR 72-600','Bombardier Q400'];
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
      const prefix   = callsign.substring(0, 3).toUpperCase();
      const airline  = AIRLINES[prefix] || 'Private / Unassigned';
      const flightName = `${callsign} (${airline})`;
      const aircraft = getMockAircraft(a.icao24);
      lbTally.airlines[airline]     = (lbTally.airlines[airline]     || 0) + 1;
      lbTally.flights[flightName]   = (lbTally.flights[flightName]   || 0) + 1;
      lbTally.aircrafts[aircraft]   = (lbTally.aircrafts[aircraft]   || 0) + 1;
    }
  });
  if (changed && lbTally.total > 0) renderLeaderboard();
}

function renderLeaderboard() {
  const sel = document.getElementById('leaderboard-select');
  if (!sel) return;
  const mode   = sel.value;
  const data   = lbTally[mode];
  if (!data || lbTally.total === 0) return;
  const sorted = Object.entries(data).sort((a, b) => b[1] - a[1]).slice(0, 5);
  const colors = ['var(--red)','var(--yellow)','var(--accent)','var(--green)','var(--text-dim)'];
  const html   = sorted.map((item, i) => {
    const count = item[1];
    const pct   = Math.round((count / lbTally.total) * 100) || '<1';
    const color = colors[i] || 'var(--text-dim)';
    return `<div class="lb-item" style="animation-delay:${i * 60}ms">
      <div class="lb-info">
        <span class="lb-name">${escapeHtml(item[0])}</span>
        <span class="lb-stat"><span style="color:${color};font-weight:700">${count}</span> (${pct}%)</span>
      </div>
      <div class="lb-bar-bg">
        <div class="lb-bar-fill" style="width:${Math.min(pct,100)}%;background:${color}"></div>
      </div>
    </div>`;
  }).join('');
  const el = document.getElementById('leaderboard-body');
  if (el) el.innerHTML = html;
}

/* ── SSE ─────────────────────────────────────────────────────── */
const LVL_CLASS = { info:'level-info', warn:'level-warn', alert:'level-alert', ok:'level-ok', error:'level-error' };
const LVL_LINE  = { info:'log-info',   warn:'log-warn',   alert:'log-alert',   ok:'log-ok',   error:'log-error' };
const LVL_FLASH = { warn:'log-flash-warn', alert:'log-flash-alert', error:'log-flash-alert' };
const LVL_ICON  = { info:'◆', warn:'▲', alert:'⚠', ok:'▼', error:'✖' };

let logIndex = 0, packetCount = 0;

function appendLog(d) {
  const logOut = document.getElementById('log-output');
  if (!logOut) return;
  const level      = (d.level || 'info').toLowerCase();
  const lineClass  = LVL_LINE[level]  || 'log-info';
  const flashClass = LVL_FLASH[level] || (level === 'info' ? 'log-flash' : '');
  const icon       = LVL_ICON[level]  || '◆';
  let cleanMsg = escapeHtml(d.msg || '');
  if (level === 'alert' || level === 'warn') {
    cleanMsg = cleanMsg.replace(/(\d+\.?\d*)/g, '<span class="log-value">$1</span>');
  }
  const row = document.createElement('div');
  row.className = `log-line ${lineClass} ${flashClass}`;
  row.style.animationDelay = `${(logIndex++ % 3) * 8}ms`;
  row.innerHTML = `
    <span class="log-time">${escapeHtml(d.time || '')}</span>
    <span class="log-level ${LVL_CLASS[level] || 'level-info'}">${icon} ${escapeHtml(level.toUpperCase())}</span>
    <span class="log-msg">${cleanMsg}</span>
    <span class="log-value">${escapeHtml(d.val || '')}</span>
  `;
  logOut.appendChild(row);
  while (logOut.children.length > 80) logOut.removeChild(logOut.firstChild);
  const nearBottom = logOut.scrollHeight - logOut.scrollTop - logOut.clientHeight < 80;
  if (nearBottom) logOut.scrollTop = logOut.scrollHeight;
  packetCount++;
  const pktEl = document.getElementById('packet-count');
  if (pktEl) {
    pktEl.textContent = `${packetCount} PKT`;
    pktEl.classList.remove('pop');
    void pktEl.offsetWidth;
    pktEl.classList.add('pop');
  }
}

function connectSSE() {
  const token    = localStorage.getItem('sg_token');
  const es       = new EventSource(`${API}/api/stream?token=${token}`);
  const connEl   = document.getElementById('conn-text');
  const streamEl = document.getElementById('stream-status');
  const pill     = document.getElementById('connection-indicator');
  es.addEventListener('log',   e => appendLog(JSON.parse(e.data)));
  es.addEventListener('stats', e => {
    updateStats(JSON.parse(e.data));
    if (typeof loadAnomalies === 'function') loadAnomalies();
  });
  es.addEventListener('airspace_changed', e => {
    const d = JSON.parse(e.data);
    if (window.airspaceSelector) window.airspaceSelector.handleSSEUpdate(d.active_regions);
    // Update the live connection label to reflect the new region(s)
    if (connEl) {
      const label = window.airspaceSelector
        ? window.airspaceSelector._regionLabel()
        : 'GLOBAL AIRSPACE';
      connEl.textContent = `LIVE · ${label}`;
    }
  });
  es.onopen = () => {
    if (connEl)   connEl.textContent   = 'LIVE · ALL REGIONS';
    if (streamEl) streamEl.textContent = 'RUNNING';
    if (pill)     pill.classList.remove('offline');
  };
  es.onerror = () => {
    if (connEl)   connEl.textContent   = 'RECONNECTING…';
    if (streamEl) streamEl.textContent = 'OFFLINE';
    if (pill)     pill.classList.add('offline');
    es.close();
    setTimeout(connectSSE, 5000);
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
  const types = ['A320','B737','A321','A319','B777','ATR72'];
  const manuf = { A320:'Airbus', B737:'Boeing', A321:'Airbus', A319:'Airbus', B777:'Boeing', ATR72:'ATR' };
  const ops   = ['IndiGo','Air India','SpiceJet','Vistara','GoFirst','AirAsia India'];
  const typ   = types[Math.floor(Math.random() * types.length)];
  return {
    icao24: icao,
    callsign: callsign || icao.toUpperCase(),
    aircraft_type: typ,
    typecode: typ,
    manufacturer: manuf[typ] || 'Unknown',
    registration: 'VT-' + icao.toUpperCase().slice(0, 3),
    operator: ops[Math.floor(Math.random() * ops.length)],
    origin_country: 'India',
    departure_airport: null, departure_city: null,
    arrival_airport: null,   arrival_city: null,
    on_ground: false,
    baro_altitude: Math.round(8000 + Math.random() * 4000),
    velocity: Math.round(200 + Math.random() * 50),
    true_track: Math.round(Math.random() * 360),
    vertical_rate: Math.round((Math.random() - 0.5) * 8),
    squawk: Math.random() > 0.85 ? '7700' : String(1000 + Math.floor(Math.random() * 7000)),
    wake_category: ['MEDIUM','HEAVY','LIGHT'][Math.floor(Math.random() * 3)]
  };
}

function renderDetail(el, d) {
  const onGroundBadge = d.on_ground
    ? `<span style="color:var(--yellow)"> On Ground</span>`
    : `<span style="color:var(--green)"> Airborne</span>`;
  const altFt  = d.baro_altitude != null ? (d.alt_unit === 'ft' ? Math.round(d.baro_altitude) : Math.round(d.baro_altitude * 3.281)) : null;
  const velKts = d.velocity      != null ? (d.vel_unit === 'kts' ? Math.round(d.velocity)      : Math.round(d.velocity * 1.944))      : null;
  el.innerHTML = `
    <div class="detail-aircraft-art">${aircraftSVG(d.wake_category)}<div class="art-label">${d.typecode||d.aircraft_type||'—'}</div></div>
    <div class="detail-grid">
      <div class="detail-route">
        <div class="route-airport">
          <div class="route-iata">${d.departure_airport||'—'}</div>
          <div class="route-city">${d.departure_city||(d.departure_airport ? 'Departure' : 'Unknown')}</div>
        </div>
        <div class="route-line">
          <div class="route-dot"></div><div class="route-dash"></div>
          <div class="route-plane">✈</div>
          <div class="route-dash"></div><div class="route-dot"></div>
        </div>
        <div class="route-airport" style="text-align:right">
          <div class="route-iata">${d.arrival_airport||'—'}</div>
          <div class="route-city">${d.arrival_city||(d.arrival_airport ? 'Arrival' : 'Unknown')}</div>
        </div>
      </div>
      <div class="fallback-links" style="grid-column:1/-1;display:flex;gap:10px;margin-top:-10px;font-size:0.8rem;opacity:0.7;">
        ${d.callsign ? `<a href="https://www.flightradar24.com/data/flights/${d.callsign}" target="_blank" style="color:var(--accent);text-decoration:none">🔍 FlightRadar24</a>` : ''}
        ${d.registration && d.registration !== '—' ? `<a href="https://planelogger.com/Aircraft/Registration/${d.registration}" target="_blank" style="color:var(--accent);text-decoration:none">📋 Photo/History</a>` : ''}
      </div>
      <div class="detail-field"><div class="detail-field-label">Aircraft Type</div><div class="detail-field-value accent">${d.aircraft_type||d.typecode||'—'}</div></div>
      <div class="detail-field"><div class="detail-field-label">Manufacturer</div><div class="detail-field-value">${d.manufacturer||'—'}</div></div>
      <div class="detail-field"><div class="detail-field-label">Registration</div><div class="detail-field-value">${d.registration||'—'}</div></div>
      <div class="detail-field"><div class="detail-field-label">Operator</div><div class="detail-field-value">${d.operator||d.owner||'—'}</div></div>
      <div class="detail-field"><div class="detail-field-label">Origin Country</div><div class="detail-field-value">${d.origin_country||'—'}</div></div>
      <div class="detail-field"><div class="detail-field-label">Status</div><div class="detail-field-value">${onGroundBadge}</div></div>
      <div class="detail-field"><div class="detail-field-label">Baro Altitude</div><div class="detail-field-value accent">${altFt ? altFt.toLocaleString()+' ft' : (d.baro_altitude ? d.baro_altitude+' m' : '—')}</div></div>
      <div class="detail-field"><div class="detail-field-label">Ground Speed</div><div class="detail-field-value">${velKts ? velKts+' kts' : (d.velocity ? d.velocity+' m/s' : '—')}</div></div>
      <div class="detail-field"><div class="detail-field-label">True Heading</div><div class="detail-field-value">${d.true_track ? d.true_track+'°' : '—'}</div></div>
      <div class="detail-field"><div class="detail-field-label">Vertical Rate</div><div class="detail-field-value ${d.vertical_rate>0?'green':d.vertical_rate<0?'red':''}">${d.vertical_rate!=null ? (d.vertical_rate>0?'↑ +':'↓ ')+d.vertical_rate+' m/s' : '—'}</div></div>
      <div class="detail-field"><div class="detail-field-label">Squawk</div><div class="detail-field-value ${['7500','7600','7700'].includes(d.squawk)?'red':''}">${d.squawk||'—'}${['7500','7600','7700'].includes(d.squawk)?' ⚠':''}</div></div>
      <div class="detail-field"><div class="detail-field-label">Wake Category</div><div class="detail-field-value">${d.wake_category||'—'}</div></div>
    </div>`;
}

async function loadFlightDetail(icao, callsign) {
  const loadingEl = document.getElementById(`loading-${icao}`);
  const innerEl   = document.getElementById(`inner-${icao}`);
  if (innerEl.dataset.loaded === '1') {
    loadingEl.style.display = 'none';
    innerEl.style.display   = 'grid';
    return;
  }
  loadingEl.style.display = 'flex';
  innerEl.style.display   = 'none';
  let detail = null;
  try {
    const r = await sgFetch(`${API}/api/flight-detail/${icao}`);
    if (r.ok) detail = await r.json();
  } catch(e) {}
  const mock = mockDetail(icao, callsign);
  detail = detail ? { ...mock, ...detail } : mock;
  renderDetail(innerEl, detail);
  loadingEl.style.display   = 'none';
  innerEl.style.display     = 'grid';
  innerEl.dataset.loaded    = '1';
}

/* ── Shared nav SVG logo ─────────────────────────────────────── */
const NAV_LOGO_SVG = `<svg viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg" width="28" height="28">
  <circle cx="14" cy="14" r="13" stroke="rgba(58,223,255,0.3)" stroke-width="1"/>
  <circle cx="14" cy="14" r="8"  stroke="rgba(58,223,255,0.5)" stroke-width="1"/>
  <circle cx="14" cy="14" r="2.5" fill="rgba(58,223,255,0.9)"/>
  <line x1="14" y1="1"  x2="14" y2="27" stroke="rgba(58,223,255,0.2)" stroke-width="0.5"/>
  <line x1="1"  y1="14" x2="27" y2="14" stroke="rgba(58,223,255,0.2)" stroke-width="0.5"/>
  <line x1="14" y1="14" x2="24" y2="6" stroke="rgba(58,223,255,0.8)" stroke-width="1.2" stroke-linecap="round">
    <animateTransform attributeName="transform" attributeType="XML" type="rotate"
      from="0 14 14" to="360 14 14" dur="4s" repeatCount="indefinite"/>
  </line>
  <circle cx="22" cy="8" r="1.5" fill="#FF5050">
    <animate attributeName="opacity" values="0;1;0" dur="4s" repeatCount="indefinite"/>
  </circle>
</svg>`;

/* ── SIEM Event Functions ─────────────────────────────────────── */
let eventStream = null;

function formatEvent(event) {
  const severityColors = {
    critical: 'var(--red)',
    high: 'var(--yellow)',
    medium: 'var(--accent)',
    low: 'var(--green)',
    info: 'var(--text-dim)'
  };
  const color = severityColors[event.severity] || 'var(--text-dim)';
  return {
    time: new Date(event.timestamp).toLocaleTimeString(),
    level: event.severity,
    msg: `${event.service}: ${event.message}`,
    val: event.anomaly_score ? event.anomaly_score.toFixed(2) : '',
    flight_id: event.flight_id,
    color
  };
}

function filterEvents(events, filters) {
  return events.filter(e => {
    if (filters.severity && e.severity !== filters.severity) return false;
    if (filters.layer && e.layer !== filters.layer) return false;
    if (filters.flight_id && !e.flight_id.includes(filters.flight_id)) return false;
    return true;
  });
}

async function loadEventStream() {
  if (eventStream) eventStream.close();
  try {
    const token = localStorage.getItem('sg_token');
    eventStream = new EventSource(`${API}/api/events/stream?token=${token}`);
    eventStream.addEventListener('event', e => {
      const event = JSON.parse(e.data);
      appendEvent(event);
    });
    eventStream.onerror = () => {
      eventStream.close();
      setTimeout(loadEventStream, 5000);
    };
  } catch (e) {}
}

function appendEvent(event) {
  const formatted = formatEvent(event);
  const row = document.createElement('div');
  row.className = 'event-row';
  row.innerHTML = `
    <span class="event-time">${formatted.time}</span>
    <span class="event-severity" style="color:${formatted.color}">${formatted.level.toUpperCase()}</span>
    <span class="event-msg">${formatted.msg}</span>
    <span class="event-score">${formatted.val}</span>
    <span class="event-flight">${formatted.flight_id || '—'}</span>
  `;
  const container = document.getElementById('events-container');
  if (container) {
    container.appendChild(row);
    while (container.children.length > 100) container.removeChild(container.firstChild);
    container.scrollTop = container.scrollHeight;
  }
}

async function escalateAlert(alertId) {
  try {
    const r = await sgFetch(`${API}/api/alerts/${alertId}/escalate`, { method: 'POST' });
    if (r.ok) {
      if (typeof loadAlerts === 'function') loadAlerts();
    }
  } catch (e) {}
}


/* ══════════════════════════════════════════════════════════════
   AirspaceSelector — Airspace region picker component
   Mount: window.airspaceSelector = new AirspaceSelector('airspace-panel');
   Requires: sgFetch, API, connectSSE already defined above.
══════════════════════════════════════════════════════════════ */
class AirspaceSelector {
  constructor(containerId) {
    this.pill = document.getElementById('connection-indicator');
    if (this.pill) {
      this.pill.style.position = 'relative';
      this.pill.style.cursor = 'pointer';
      
      this.container = document.createElement('div');
      this.container.style.position = 'absolute';
      this.container.style.top = '100%';
      this.container.style.right = '0';
      this.container.style.marginTop = '14px';
      this.container.style.width = '340px';
      this.container.style.display = 'none';
      this.container.style.zIndex = '99999';
      this.container.style.cursor = 'default';
      
      this.container.addEventListener('click', e => e.stopPropagation());
      this.pill.addEventListener('click', e => {
        e.stopPropagation();
        const isHidden = this.container.style.display === 'none';
        this.container.style.display = isHidden ? 'block' : 'none';
      });
      document.addEventListener('click', () => {
        this.container.style.display = 'none';
      });
      
      this.pill.appendChild(this.container);
    } else {
      this.container = document.getElementById(containerId);
    }
    this.allRegions = [];       // filled from GET /api/airspaces
    this.active     = new Set();
    this.monitorAll = false;
    this.allGroupsExpanded = true;  // track group expand/collapse state
    if (this.container) this._init();
  }

  // ── Initialise: load catalogue, render, then silently auto-detect ────────
  async _init() {
    try {
      const r    = await sgFetch(`${API}/api/airspaces`);
      const data = await r.json();
      this.allRegions = data.regions || [];
      this.active     = new Set(data.active || []);
      this.monitorAll = (this.active.size === this.allRegions.length && this.allRegions.length > 0);
    } catch(e) {
      console.warn('[AirspaceSelector] init failed:', e);
    }
    this._render();
  }

  // ── Render the full panel HTML with a simple pill grid ──────────────────
  _render() {
    if (!this.container) return;
    const allActive = this.monitorAll;

    // Build the grid of region pills
    const pillsHtml = this.allRegions.map(r => `
      <button class="region-pill ${this.active.has(r.id) ? 'active' : ''}" 
              data-region-id="${r.id}" 
              title="${r.name} — ${r.description}">
        ${r.name}
      </button>
    `).join('');

    this.container.innerHTML = `
      <div class="airspace-selector">
        <div class="airspace-header">
          <span>🌍 Airspace Monitor</span>
        </div>
        
        <input type="text" id="airspace-search" class="airspace-search" placeholder="🔍 Search regions...">
        
        <button id="btn-monitor-all" class="monitor-all-btn ${allActive ? 'active' : ''}">
          ${allActive ? '✓ Monitoring All Regions' : 'Monitor All Regions'}
        </button>
        
        <button id="btn-detect-location" class="detect-location-btn">
          <span class="loc-icon">📍</span> Detect My Location
        </button>
        
        <div class="airspace-divider"></div>
        
        <div class="airspace-grid">
          ${pillsHtml}
        </div>
        
        <div class="airspace-status">${this._statusText()}</div>
      </div>
    `;

    // Event listeners - Monitor All
    const monitorAllBtn = this.container.querySelector('#btn-monitor-all');
    if (monitorAllBtn) {
      monitorAllBtn.addEventListener('click', () => this._toggleAll());
    }
    
    // Event listeners - Detect Location
    const detectBtn = this.container.querySelector('#btn-detect-location');
    if (detectBtn) {
      detectBtn.addEventListener('click', () => this._detectLocation({ silent: false }));
    }
    
    // Event listeners - Region pills
    this.container.querySelectorAll('.region-pill').forEach(item => {
      item.addEventListener('click', () => {
        const regionId = item.dataset.regionId;
        if (regionId) this._toggleRegion(regionId);
      });
    });

    // Event listeners - Search
    const searchInput = this.container.querySelector('#airspace-search');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => this._filterRegions(e.target.value));
    }
  }
  
  // ── Filter regions by search query ───────────────────────────────────────
  _filterRegions(query) {
    const lowerQuery = query.toLowerCase();
    const pills = this.container.querySelectorAll('.region-pill');
    
    pills.forEach(pill => {
      const text = pill.textContent.toLowerCase();
      // Only display the pill if the search string is found in its name
      if (text.includes(lowerQuery)) {
        pill.style.display = 'block';
      } else {
        pill.style.display = 'none';
      }
    });
  }
  


  // ── Toggle a single region pill ──────────────────────────────────────────
  _toggleRegion(regionId) {
    if (this.monitorAll) return;   // locked while "Monitor All" is active
    if (this.active.has(regionId)) {
      if (this.active.size === 1) return;  // must keep at least one
      this.active.delete(regionId);
    } else {
      this.active.add(regionId);
    }
    this.monitorAll = (this.active.size === this.allRegions.length);
    this._pushToServer();
  }

  // ── Toggle Monitor All ───────────────────────────────────────────────────
  _toggleAll() {
    this.monitorAll = !this.monitorAll;
    if (this.monitorAll) {
      this.active = new Set(this.allRegions.map(r => r.id));
    } else {
      this.active = new Set();
    }
    this._pushToServer();
  }

  // ── Detect user location and auto-select the matching region ────────────
  async _detectLocation({ silent = false } = {}) {
    const btn = this.container ? this.container.querySelector('#btn-detect-location') : null;
    if (!navigator.geolocation) {
      if (!silent) alert('Geolocation is not supported by your browser.');
      return;
    }
    if (btn && !silent) {
      btn.innerHTML = '<span class="loc-icon">📍</span> Locating…';
      btn.disabled  = true;
    }

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;
        try {
          const r      = await sgFetch(`${API}/api/locate?lat=${latitude}&lon=${longitude}`);
          const result = await r.json();

          if (result.regions && result.regions.length > 0) {
            // Auto-select all matched regions
            this.monitorAll = false;
            this.active     = new Set(result.regions.map(r => r.id));
            await this._pushToServer();
            const regionNames = result.regions.map(r => r.name).join(', ');
            this._setStatus(`📍 Auto-selected: ${regionNames}`, 'success');
          } else {
            // Outside all defined regions — stay on Monitor All
            this.monitorAll = true;
            this.active     = new Set(this.allRegions.map(r => r.id));
            await this._pushToServer();
            if (!silent) this._setStatus('📍 Location outside monitored regions', 'warn');
          }
        } catch(e) {
          if (!silent) this._setStatus('Location lookup failed.', 'error');
        } finally {
          if (btn && !silent) {
            btn.innerHTML = '<span class="loc-icon">📍</span> Detect My Location';
            btn.disabled  = false;
          }
        }
      },
      (error) => {
        if (btn && !silent) {
          btn.innerHTML = '<span class="loc-icon">📍</span> Detect My Location';
          btn.disabled  = false;
          const msgs = {
            1: 'Location permission denied.',
            2: 'Location unavailable. Check GPS/network.',
            3: 'Location request timed out.',
          };
          this._setStatus(msgs[error.code] || 'Location error.', 'error');
        }
        // Silent mode: stay on current state (Monitor All default)
      },
      { timeout: 8000, maximumAge: 60000 }
    );
  }

  // ── Push current active set to the server ────────────────────────────────
  async _pushToServer() {
    try {
      await sgFetch(`${API}/api/airspace`, {
        method:  'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ active: [...this.active] })
      });
    } catch(e) {
      console.warn('[AirspaceSelector] push failed:', e);
    }
    this._render();
  }

  // ── SSE handler — called by connectSSE() on airspace_changed events ──────
  handleSSEUpdate(activeRegions) {
    this.active     = new Set(activeRegions);
    this.monitorAll = (activeRegions.length === this.allRegions.length && this.allRegions.length > 0);
    this._render();
  }

  // ── Status text helpers ──────────────────────────────────────────────────
  _statusText() {
    if (this.allRegions.length === 0) return 'Loading regions…';
    if (this.monitorAll) return `✅ Monitoring all ${this.allRegions.length} regions`;
    const names = [...this.active]
      .map(id => this.allRegions.find(r => r.id === id)?.name)
      .filter(Boolean);
    return names.length ? `✅ Monitoring: ${names.join(', ')}` : 'No region selected';
  }

  _setStatus(message, type = 'info') {
    const el = this.container ? this.container.querySelector('.airspace-status') : null;
    if (!el) return;
    const colors = { success: 'var(--green)', warn: 'var(--yellow)', error: 'var(--red)', info: 'var(--text-dim)' };
    el.textContent = message;
    el.style.color = colors[type] || 'var(--text-dim)';
    // Reset to dim after 4 seconds
    if (type !== 'info') setTimeout(() => {
      if (el) { el.textContent = this._statusText(); el.style.color = ''; }
    }, 4000);
  }

  // Returns a short human-readable label for the conn-text element
  _regionLabel() {
    if (this.monitorAll || this.active.size === this.allRegions.length) return 'ALL REGIONS';
    const names = [...this.active]
      .map(id => this.allRegions.find(r => r.id === id)?.name)
      .filter(Boolean);
    return names.join(' + ').toUpperCase() || 'CUSTOM';
  }
}

// Globally mount the dropdown selector to the navbar pill
setTimeout(() => {
  if (!window.airspaceSelector) {
    window.airspaceSelector = new AirspaceSelector();
  }
}, 0);