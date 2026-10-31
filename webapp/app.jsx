/* PolySignal — app shell --------------------------------------------------- */

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accent": "#34d399",
  "density": "regular",
  "bgDepth": "deep",
  "chartPressure": true,
  "chartArea": false
}/*EDITMODE-END*/;

const ACCENTS = {
  '#34d399': 'Terminal green',
  '#5b9dff': 'Electric blue',
  '#f5a524': 'Amber',
};

function applyTheme(t) {
  const root = document.documentElement;
  root.style.setProperty('--accent', t.accent);
  if (t.bgDepth === 'deep') {
    root.style.setProperty('--bg', '#08090a');
    root.style.setProperty('--panel', '#0f1011');
    root.style.setProperty('--border', '#1c1e21');
  } else {
    root.style.setProperty('--bg', '#0e1012');
    root.style.setProperty('--panel', '#15181a');
    root.style.setProperty('--border', '#23272b');
  }
}

function TopBar({ system, accent, view, market, onHome }) {
  return (
    <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 22px', height: 54, borderBottom: '1px solid var(--border)', flexShrink: 0, background: 'var(--bg)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 22 }}>
        <div onClick={onHome} style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }}>
          <svg width="22" height="22" viewBox="0 0 22 22">
            <rect x="1" y="1" width="20" height="20" rx="5" fill="none" stroke={accent} strokeWidth="1.5" opacity="0.5" />
            <circle cx="11" cy="11" r="3.4" fill={accent} />
            <circle cx="11" cy="11" r="6.4" fill="none" stroke={accent} strokeWidth="1.2" opacity="0.4" />
          </svg>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 1 }}>
            <span style={{ font: "600 15px 'IBM Plex Sans', sans-serif", color: '#f0f2f4', letterSpacing: '-0.01em' }}>Poly</span>
            <span style={{ font: "600 15px 'IBM Plex Sans', sans-serif", color: accent, letterSpacing: '-0.01em' }}>Signal</span>
          </div>
        </div>
        <nav style={{ display: 'flex', alignItems: 'center', gap: 4, font: "12px 'IBM Plex Mono', monospace" }}>
          <span style={{ color: '#7d838a' }}>/</span>
          <button onClick={onHome} style={{ background: 'none', border: 'none', cursor: 'pointer', color: view === 'universe' ? '#e8eaed' : '#7d838a', font: "12px 'IBM Plex Mono', monospace" }}>Market Universe</button>
          {view === 'detail' && market && (
            <React.Fragment>
              <span style={{ color: '#5f656c' }}>/</span>
              <span style={{ color: '#9aa0a6', maxWidth: 240, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{market.short}</span>
            </React.Fragment>
          )}
        </nav>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, font: "10.5px 'IBM Plex Mono', monospace", color: '#7d838a' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className="ps-live" style={{ width: 6, height: 6, borderRadius: '50%', background: accent }} />
            <span style={{ color: '#9aa0a6' }}>{system.lastUpdateAgo}</span>
          </span>
          <span style={{ color: '#3a3d42' }}>|</span>
          <span>{system.benchmark}</span>
        </div>
        <span style={{ font: "9.5px 'IBM Plex Mono', monospace", color: accent, border: `1px solid ${accent}44`, background: `${accent}12`, padding: '3px 8px', borderRadius: 4, letterSpacing: '0.04em' }}>{system.mode}</span>
      </div>
    </header>
  );
}

/* Subscribe to the data loader's boot/status channel (data.js). */
function usePsData() {
  const [boot, setBoot] = useState(() => window.PS_BOOT || { status: 'loading', error: null, endpoint: '' });
  useEffect(() => {
    const onEvt = (e) => setBoot({ ...e.detail });
    window.addEventListener('ps-data', onEvt);
    // Catch a load that resolved before this component mounted.
    if (window.PS_BOOT) setBoot({ ...window.PS_BOOT });
    return () => window.removeEventListener('ps-data', onEvt);
  }, []);
  return boot;
}

function Wordmark({ accent, size = 17 }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
      <svg width={size + 6} height={size + 6} viewBox="0 0 22 22">
        <rect x="1" y="1" width="20" height="20" rx="5" fill="none" stroke={accent} strokeWidth="1.5" opacity="0.5" />
        <circle cx="11" cy="11" r="3.4" fill={accent} />
        <circle cx="11" cy="11" r="6.4" fill="none" stroke={accent} strokeWidth="1.2" opacity="0.4" />
      </svg>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 1 }}>
        <span style={{ font: `600 ${size}px 'IBM Plex Sans', sans-serif`, color: '#f0f2f4', letterSpacing: '-0.01em' }}>Poly</span>
        <span style={{ font: `600 ${size}px 'IBM Plex Sans', sans-serif`, color: accent, letterSpacing: '-0.01em' }}>Signal</span>
      </div>
    </div>
  );
}

/* Full-screen gate shown while loading, or when the API is unavailable. */
function BootGate({ boot, accent }) {
  const isError = boot.status === 'error';
  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg)' }}>
      <header style={{ display: 'flex', alignItems: 'center', padding: '0 22px', height: 54, borderBottom: '1px solid var(--border)' }}>
        <Wordmark accent={accent} size={15} />
      </header>
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24 }}>
        <div style={{ width: 'min(480px, 100%)', textAlign: 'center' }}>
          {!isError ? (
            <React.Fragment>
              <div className="ps-spin" style={{ width: 30, height: 30, margin: '0 auto 22px', border: `2px solid ${accent}33`, borderTopColor: accent, borderRadius: '50%' }} />
              <div style={{ font: "13px 'IBM Plex Mono', monospace", color: '#9aa0a6', letterSpacing: '0.04em' }}>Loading social-enriched dataset…</div>
              <div style={{ font: "10.5px 'IBM Plex Mono', monospace", color: '#5f656c', marginTop: 8, wordBreak: 'break-all' }}>{boot.endpoint}</div>
            </React.Fragment>
          ) : (
            <React.Fragment>
              <div style={{ width: 40, height: 40, margin: '0 auto 20px', borderRadius: 9, border: '1px solid #f5a52455', background: 'rgba(245,165,36,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span style={{ color: '#f5a524', font: "600 19px 'IBM Plex Mono', monospace" }}>!</span>
              </div>
              <div style={{ font: "600 15px 'IBM Plex Sans', sans-serif", color: '#f0f2f4', marginBottom: 8 }}>Dashboard data unavailable</div>
              <div style={{ font: "12px/1.6 'IBM Plex Sans', sans-serif", color: '#9aa0a6', marginBottom: 16 }}>
                Configure your local dataset to view the dashboard. Complete data is kept separately from the GitHub repository.
              </div>
              <div style={{ background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 8, padding: '11px 13px', textAlign: 'left', marginBottom: 18 }}>
                <div style={{ font: "9.5px 'IBM Plex Mono', monospace", color: '#5f656c', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 6 }}>Endpoint</div>
                <div style={{ font: "11px 'IBM Plex Mono', monospace", color: '#c4c8cc', wordBreak: 'break-all', marginBottom: 10 }}>GET {boot.endpoint}</div>
                <div style={{ font: "9.5px 'IBM Plex Mono', monospace", color: '#5f656c', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 6 }}>Reason</div>
                <div style={{ font: "11px 'IBM Plex Mono', monospace", color: '#f76d6d', wordBreak: 'break-word' }}>{boot.error || 'Unknown error'}</div>
              </div>
              <button onClick={() => window.PS_RELOAD && window.PS_RELOAD()} style={{
                font: "12px 'IBM Plex Mono', monospace", color: '#08090a', background: accent, border: 'none',
                borderRadius: 7, padding: '9px 18px', cursor: 'pointer', letterSpacing: '0.03em',
              }}>Retry</button>
              <div style={{ font: "10px 'IBM Plex Mono', monospace", color: '#4a4f55', marginTop: 16 }}>
                See docs/RUN_APP.md for setup instructions. Restart the local app after configuring the data, then retry.
              </div>
            </React.Fragment>
          )}
        </div>
      </div>
    </div>
  );
}

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [view, setView] = useState('universe');
  const [market, setMarket] = useState(null);
  const boot = usePsData();

  useEffect(() => { applyTheme(t); }, [t.accent, t.bgDepth]);
  useEffect(() => { applyTheme(t); }, []);

  const accent = t.accent;
  const density = t.density;

  function open(m) { setMarket(m); setView('detail'); window.scrollTo(0, 0); }
  function home() { setView('universe'); }

  // Gate on the live data load — no mock fallback.
  if (boot.status !== 'ready' || !window.PS_DATA) {
    return <BootGate boot={boot} accent={accent} />;
  }

  const { markets, system } = window.PS_DATA;

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg)' }}>
      <TopBar system={system} accent={accent} view={view} market={market} onHome={home} />
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        {view === 'universe'
          ? <MarketUniverse markets={markets} system={system} accent={accent} density={density} onOpen={open} />
          : <MarketDetail market={market} accent={accent} density={density} system={system} chartStyle={{ pressure: t.chartPressure, area: t.chartArea }} onBack={home} />}
      </div>

      <TweaksPanel>
        <TweakSection label="Theme" />
        <TweakColor label="Signal accent" value={t.accent} options={Object.keys(ACCENTS)} onChange={v => setTweak('accent', v)} />
        <TweakRadio label="Background" value={t.bgDepth} options={['deep', 'lifted']} onChange={v => setTweak('bgDepth', v)} />
        <TweakSection label="Layout" />
        <TweakRadio label="Density" value={t.density} options={['compact', 'regular']} onChange={v => setTweak('density', v)} />
        <TweakSection label="Chart defaults" />
        <TweakToggle label="Pressure overlay" value={t.chartPressure} onChange={v => setTweak('chartPressure', v)} />
        <TweakToggle label="Area fill" value={t.chartArea} onChange={v => setTweak('chartArea', v)} />
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
