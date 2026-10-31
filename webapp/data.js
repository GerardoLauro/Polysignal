/* PolySignal — live data loader -------------------------------------------
 * Fetches the social-enriched dataset from the backend and publishes it as the
 * app's single data contract:
 *
 *     window.PS_DATA = { markets, system }
 *
 * There is NO mock fallback. If the API is unavailable or returns a malformed
 * body, the app renders an explicit "API unavailable" state — never fabricated
 * numbers.
 *
 * Loaded as an ES module (<script type="module">) so that import.meta.env is
 * available under Vite. In the no-build prototype import.meta.env is undefined
 * and API_BASE falls back to localhost.
 * ------------------------------------------------------------------------- */

/* API base.
 * Production (Vite): set VITE_POLYSIGNAL_API_BASE in your .env — Vite statically
 *   replaces import.meta.env.VITE_POLYSIGNAL_API_BASE at build time.
 * Prototype (no bundler): import.meta.env is undefined, so we fall back to the
 *   local FastAPI dev server. You can also override via window.PS_API_BASE
 *   before this script runs. */
const API_BASE =
  (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_POLYSIGNAL_API_BASE) ||
  (typeof window !== 'undefined' && window.PS_API_BASE) ||
  'http://localhost:8000';

const PS_DATA_ENDPOINT = `${API_BASE.replace(/\/$/, '')}/api/social-enriched/ps-data`;

/* Boot/status channel the UI subscribes to: status = loading | ready | error. */
const boot = (window.PS_BOOT = {
  status: 'loading',
  error: null,
  endpoint: PS_DATA_ENDPOINT,
});

function publish(patch) {
  Object.assign(boot, patch);
  window.dispatchEvent(new CustomEvent('ps-data', { detail: { ...boot } }));
}

/* Minimal shape guard so a malformed body fails loudly instead of half-rendering. */
function validatePayload(payload) {
  if (!payload || typeof payload !== 'object') throw new Error('Empty or non-JSON response body.');
  if (!Array.isArray(payload.markets)) throw new Error('Response is missing a "markets" array.');
  if (payload.markets.length === 0) throw new Error('Response contains zero markets.');
  if (!payload.system || typeof payload.system !== 'object') throw new Error('Response is missing a "system" object.');
  const m = payload.markets[0];
  if (!m.series || !Array.isArray(m.series.poly)) throw new Error('Market is missing series.poly[] timeline.');
  return payload;
}

async function loadPsData() {
  publish({ status: 'loading', error: null });
  try {
    const res = await fetch(PS_DATA_ENDPOINT, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) {
      let detail = '';
      try { detail = (await res.json()).error || ''; } catch (_) { /* HTTP fallback below */ }
      throw new Error(detail || `API responded ${res.status} ${res.statusText || ''}`.trim());
    }

    let payload;
    try {
      payload = await res.json();
    } catch (e) {
      throw new Error('Response was not valid JSON.');
    }

    validatePayload(payload);

    /* Contract preserved exactly: window.PS_DATA = { markets, system } */
    window.PS_DATA = { markets: payload.markets, system: payload.system };
    publish({ status: 'ready', error: null });
  } catch (err) {
    window.PS_DATA = null;
    const message = err && err.message ? err.message : String(err);
    publish({ status: 'error', error: message });
    // eslint-disable-next-line no-console
    console.warn('[PolySignal] data load failed:', message, '·', PS_DATA_ENDPOINT);
  }
}

/* Retry hook used by the error state's "Retry" button. */
window.PS_RELOAD = loadPsData;

loadPsData();
