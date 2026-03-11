const API_BASE = `${window.location.protocol}//${window.location.host}`;

const els = {
  mode: document.getElementById("modeSelect"),
  hours: document.getElementById("hoursInput"),
  start: document.getElementById("startDate"),
  end: document.getElementById("endDate"),
  status: document.getElementById("statusText"),
  predictBtn: document.getElementById("predictBtn"),
  loadHistoricalBtn: document.getElementById("loadHistoricalBtn"),
  avgPm25: document.getElementById("avgPm25"),
  maxPm25: document.getElementById("maxPm25"),
  aqiCategory: document.getElementById("aqiCategory"),
  predictionCount: document.getElementById("predictionCount"),
};

let latestPredictions = [];
let latestHistorical = [];

function setStatus(text) {
  els.status.textContent = text;
}

function aqiCounts(predictions) {
  const counts = {};
  for (const p of predictions) {
    const key = p.aqi_category || "Unknown";
    counts[key] = (counts[key] || 0) + 1;
  }
  return counts;
}

async function callApi(path, method = "GET", body = null) {
  const options = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body) {
    options.body = JSON.stringify(body);
  }

  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed with ${res.status}`);
  }
  return res.json();
}

function updateMetrics(predictions) {
  if (!predictions.length) {
    els.avgPm25.textContent = "-";
    els.maxPm25.textContent = "-";
    els.aqiCategory.textContent = "-";
    els.predictionCount.textContent = "0";
    return;
  }

  const pm = predictions.map((p) => Number(p["PM2.5"]) || 0);
  const avg = pm.reduce((a, b) => a + b, 0) / pm.length;
  const max = Math.max(...pm);

  els.avgPm25.textContent = avg.toFixed(2);
  els.maxPm25.textContent = max.toFixed(2);
  els.aqiCategory.textContent = predictions[0].aqi_category || "-";
  els.predictionCount.textContent = String(predictions.length);
}

function renderPm25Chart(predictions) {
  Plotly.newPlot(
    "pm25Chart",
    [
      {
        x: predictions.map((p) => p.datetime),
        y: predictions.map((p) => p["PM2.5"]),
        mode: "lines+markers",
        name: "PM2.5",
        line: { color: "#0aa1dd", width: 3 },
      },
    ],
    {
      margin: { t: 20, r: 20, b: 50, l: 50 },
      xaxis: { title: "Datetime" },
      yaxis: { title: "PM2.5" },
    },
    { responsive: true }
  );
}

function renderMultiPollutantChart(predictions) {
  const names = ["PM2.5", "PM10", "NO2", "CO", "SO2"];
  const colors = ["#0aa1dd", "#005b96", "#ff7f0e", "#2ca02c", "#d62728"];

  const traces = names.map((name, i) => ({
    x: predictions.map((p) => p.datetime),
    y: predictions.map((p) => p[name]),
    mode: "lines",
    name,
    line: { width: 2, color: colors[i] },
  }));

  Plotly.newPlot(
    "multiPollutantChart",
    traces,
    {
      margin: { t: 20, r: 20, b: 50, l: 50 },
      xaxis: { title: "Datetime" },
      yaxis: { title: "Concentration" },
    },
    { responsive: true }
  );
}

function renderHistoryVsPredicted(predictions, historical) {
  const traces = [];

  if (historical.length) {
    traces.push({
      x: historical.map((d) => d.datetime),
      y: historical.map((d) => d["PM2.5"]),
      mode: "lines",
      name: "Historical PM2.5",
      line: { color: "#567189", width: 2 },
    });
  }

  traces.push({
    x: predictions.map((d) => d.datetime),
    y: predictions.map((d) => d["PM2.5"]),
    mode: "lines+markers",
    name: "Predicted PM2.5",
    line: { color: "#0aa1dd", width: 3, dash: "dot" },
  });

  Plotly.newPlot(
    "historyVsPredChart",
    traces,
    {
      margin: { t: 20, r: 20, b: 50, l: 50 },
      xaxis: { title: "Datetime" },
      yaxis: { title: "PM2.5" },
    },
    { responsive: true }
  );
}

function renderAqiChart(predictions) {
  const counts = aqiCounts(predictions);
  Plotly.newPlot(
    "aqiChart",
    [
      {
        type: "pie",
        labels: Object.keys(counts),
        values: Object.values(counts),
        hole: 0.4,
      },
    ],
    {
      margin: { t: 20, r: 20, b: 20, l: 20 },
    },
    { responsive: true }
  );
}

function renderAll(predictions) {
  updateMetrics(predictions);
  renderPm25Chart(predictions);
  renderMultiPollutantChart(predictions);
  renderHistoryVsPredicted(predictions, latestHistorical);
  renderAqiChart(predictions);
}

async function loadHistorical() {
  try {
    setStatus("Loading historical data...");
    const data = await callApi("/historical?hours=168");
    latestHistorical = data.historical || [];
    setStatus(`Historical points loaded: ${latestHistorical.length}`);
    if (latestPredictions.length) {
      renderHistoryVsPredicted(latestPredictions, latestHistorical);
    }
  } catch (err) {
    setStatus(`Historical data error: ${err.message}`);
  }
}

async function requestPrediction() {
  try {
    setStatus("Requesting prediction...");

    const mode = els.mode.value;
    let payload = {};
    let path = "/predict-hour";

    if (mode === "hour") {
      payload = { hours: Number(els.hours.value || 6) };
      path = "/predict-hour";
    } else if (mode === "day") {
      payload = {};
      path = "/predict-day";
    } else {
      const startDate = els.start.value;
      const endDate = els.end.value;
      if (!startDate || !endDate) {
        setStatus("Please select both start and end dates for range mode.");
        return;
      }
      payload = { start_date: startDate, end_date: endDate };
      path = "/predict-range";
    }

    const data = await callApi(path, "POST", payload);
    latestPredictions = data.predictions || [];

    if (!latestPredictions.length) {
      setStatus("No predictions returned.");
      return;
    }

    setStatus(`Prediction loaded: ${latestPredictions.length} points`);
    renderAll(latestPredictions);
  } catch (err) {
    setStatus(`Prediction error: ${err.message}`);
  }
}

els.predictBtn.addEventListener("click", requestPrediction);
els.loadHistoricalBtn.addEventListener("click", loadHistorical);

// Initial data pull.
requestPrediction();
loadHistorical();
