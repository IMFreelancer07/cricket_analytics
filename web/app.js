const providerEl = document.getElementById('provider');
const modelEl = document.getElementById('model');
const apiKeyEl = document.getElementById('apiKey');
const tempEl = document.getElementById('temperature');
const tempValEl = document.getElementById('tempVal');
const queryEl = document.getElementById('query');
const filtersEl = document.getElementById('filters');
const askBtn = document.getElementById('ask');
const examplesBtn = document.getElementById('examples');
const answerEl = document.getElementById('answer');
const sourcesEl = document.getElementById('sources');
const healthEl = document.getElementById('health');

const DEFAULT_MODELS = {
  openai: 'gpt-3.5-turbo',
  groq: 'mixtral-8x7b-32768',
  gemini: 'gemini-pro',
};

function setDefaultModel() {
  const p = providerEl.value;
  modelEl.value = DEFAULT_MODELS[p] || '';
}

providerEl.addEventListener('change', setDefaultModel);
setDefaultModel();

// Temperature live label
function updateTempLabel() {
  tempValEl.textContent = tempEl.value;
}

tempEl.addEventListener('input', updateTempLabel);
updateTempLabel();

async function checkHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    const ok = data?.status === 'healthy' && data?.services?.rag_system === 'available';
    healthEl.textContent = ok ? 'Healthy' : 'Degraded';
    healthEl.classList.toggle('status-ok', ok);
    healthEl.classList.toggle('status-bad', !ok);
    healthEl.classList.remove('status-unknown');
  } catch (e) {
    healthEl.textContent = 'Unavailable';
    healthEl.classList.add('status-bad');
  }
}

checkHealth();

examplesBtn.addEventListener('click', () => {
  queryEl.value = 'What bowling changes worked best in the last 5 matches at Eden Gardens?';
  filtersEl.value = '{"venue": "Eden Gardens"}';
});

askBtn.addEventListener('click', async () => {
  const query = queryEl.value.trim();
  if (!query) return;
  let filters = undefined;
  if (filtersEl.value.trim()) {
    try { filters = JSON.parse(filtersEl.value); } catch { alert('Invalid JSON in filters'); return; }
  }

  askBtn.disabled = true;
  answerEl.textContent = 'Working on it…';
  sourcesEl.innerHTML = '';

  try {
    const payload = {
      query,
      query_type: 'general',
      filters,
      use_complex_reasoning: false,
      provider: providerEl.value,
      model_name: modelEl.value || undefined,
      temperature: Number(tempEl.value),
      api_key: apiKeyEl.value || undefined,
    };

    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.error || `Request failed: ${res.status}`);
    }

    const data = await res.json();
    answerEl.textContent = data.answer || 'No answer produced.';

    const sources = Array.isArray(data.sources) ? data.sources : [];
    if (sources.length === 0) {
      sourcesEl.innerHTML = '<div class="source-card">No sources returned.</div>';
    } else {
      sourcesEl.innerHTML = sources.map(s => `<div class="source-card">${escapeHtml(s)}</div>`).join('');
    }
  } catch (e) {
    answerEl.textContent = `Error: ${e.message}`;
  } finally {
    askBtn.disabled = false;
  }
});

function escapeHtml(str) {
  return String(str)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}
