import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dir    = dirname(fileURLToPath(import.meta.url));
const BTC_CACHE = join(__dir, "../btc-mcp/data/hourly_24h.json");
const OUT_DIR   = join(__dir, "data");
const OUT_FILE  = join(OUT_DIR, "summary.json");

if (!existsSync(OUT_DIR)) mkdirSync(OUT_DIR, { recursive: true });

// ─── Analysis helpers ─────────────────────────────────────────────────────────

function avg(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function rsi(closes, period = 14) {
  if (closes.length < period + 1) return null;
  const slice = closes.slice(-(period + 1));
  let gains = 0, losses = 0;
  for (let i = 1; i < slice.length; i++) {
    const diff = slice[i] - slice[i - 1];
    if (diff >= 0) gains += diff; else losses -= diff;
  }
  const rs = losses === 0 ? Infinity : gains / losses;
  return +(100 - 100 / (1 + rs)).toFixed(2);
}

function ema(closes, period) {
  if (closes.length < period) return null;
  const k = 2 / (period + 1);
  let val = avg(closes.slice(0, period));
  for (let i = period; i < closes.length; i++) {
    val = closes[i] * k + val * (1 - k);
  }
  return +val.toFixed(2);
}

function supportResistance(candles) {
  const highs  = candles.map(c => c.high);
  const lows   = candles.map(c => c.low);
  return {
    resistance: +Math.max(...highs).toFixed(2),
    support:    +Math.min(...lows).toFixed(2),
  };
}

function trendLabel(pct) {
  if (pct > 2)  return "strong_uptrend";
  if (pct > 0.5) return "uptrend";
  if (pct < -2) return "strong_downtrend";
  if (pct < -0.5) return "downtrend";
  return "sideways";
}

function volumeProfile(candles) {
  const vols = candles.map(c => c.volume);
  const avgVol = avg(vols);
  const lastVol = vols[vols.length - 1];
  return {
    avg_volume:  +avgVol.toFixed(2),
    last_volume: +lastVol.toFixed(2),
    volume_ratio: +(lastVol / avgVol).toFixed(2),
  };
}

function buildSummary(data) {
  const candles = data.candles;
  const closes  = candles.map(c => c.close);
  const first   = closes[0];
  const last    = closes[closes.length - 1];
  const changePct = +((last - first) / first * 100).toFixed(4);

  const ema7    = ema(closes, 7);
  const ema14   = ema(closes, 14);
  const rsi14   = rsi(closes, 14);
  const { resistance, support } = supportResistance(candles);
  const vol     = volumeProfile(candles);
  const trend   = trendLabel(changePct);

  // ── Forecast logic ────────────────────────────────────────────────────────
  // Simple rule-based forecast combining trend, RSI, EMA cross and volume
  const signals = [];
  let bullScore = 0;
  let bearScore = 0;

  if (changePct > 0) { bullScore += 1; signals.push("24h price change positive"); }
  else               { bearScore += 1; signals.push("24h price change negative"); }

  if (rsi14 !== null) {
    if (rsi14 < 30)      { bullScore += 2; signals.push(`RSI ${rsi14} oversold → reversal likely`); }
    else if (rsi14 > 70) { bearScore += 2; signals.push(`RSI ${rsi14} overbought → pullback likely`); }
    else if (rsi14 > 55) { bullScore += 1; signals.push(`RSI ${rsi14} bullish momentum`); }
    else if (rsi14 < 45) { bearScore += 1; signals.push(`RSI ${rsi14} bearish momentum`); }
  }

  if (ema7 !== null && ema14 !== null) {
    if (ema7 > ema14) { bullScore += 1; signals.push("EMA7 > EMA14 — golden cross"); }
    else              { bearScore += 1; signals.push("EMA7 < EMA14 — death cross"); }
  }

  if (last > resistance * 0.99) { bearScore += 1; signals.push("Price near 24h resistance"); }
  if (last < support  * 1.01)   { bullScore += 1; signals.push("Price near 24h support (bounce zone)"); }

  if (vol.volume_ratio > 1.5) {
    if (changePct > 0) { bullScore += 1; signals.push("High volume on up move"); }
    else               { bearScore += 1; signals.push("High volume on down move"); }
  }

  let direction, confidence;
  const diff = bullScore - bearScore;
  if      (diff >= 3)  { direction = "UP";      confidence = "high"; }
  else if (diff >= 1)  { direction = "UP";      confidence = "moderate"; }
  else if (diff <= -3) { direction = "DOWN";    confidence = "high"; }
  else if (diff <= -1) { direction = "DOWN";    confidence = "moderate"; }
  else                 { direction = "SIDEWAYS"; confidence = "low"; }

  const priceTarget = direction === "UP"
    ? +(last * 1.02).toFixed(2)
    : direction === "DOWN"
      ? +(last * 0.98).toFixed(2)
      : last;

  return {
    generatedAt: new Date().toISOString(),
    dataFetchedAt: data.fetchedAt,
    current_price: last,
    open_24h:      first,
    change_24h_pct: changePct,
    high_24h:      resistance,
    low_24h:       support,
    ema7,
    ema14,
    rsi14,
    volume: vol,
    trend,
    forecast: {
      direction,
      confidence,
      bull_score: bullScore,
      bear_score: bearScore,
      signals,
      estimated_next_target: priceTarget,
      note: `Based on 24h hourly candles. ${direction} with ${confidence} confidence.`,
    },
  };
}

// ─── MCP Server ───────────────────────────────────────────────────────────────

const server = new Server(
  { name: "mcp-summarize", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "summarize_btc",
      description:
        "Read cached 24h hourly BTC data from btc-mcp, compute technical indicators " +
        "(EMA7/14, RSI14, support/resistance, volume), and generate a price forecast " +
        "(UP / DOWN / SIDEWAYS with confidence level). Saves result to disk.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "get_summary",
      description:
        "Return the last saved BTC summary and forecast without recomputing.",
      inputSchema: { type: "object", properties: {} },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name } = req.params;

  switch (name) {

    case "summarize_btc": {
      if (!existsSync(BTC_CACHE)) {
        return text(
          "No 24h data found. Run btc_fetch_24h in btc-mcp first."
        );
      }
      let data;
      try {
        data = JSON.parse(readFileSync(BTC_CACHE, "utf8"));
      } catch (err) {
        return text(`Failed to read btc-mcp cache: ${err.message}`);
      }
      if (!data.candles || data.candles.length === 0) {
        return text("Cache is empty. Fetch data first with btc_fetch_24h.");
      }

      const summary = buildSummary(data);
      writeFileSync(OUT_FILE, JSON.stringify(summary, null, 2), "utf8");
      return text(JSON.stringify(summary, null, 2));
    }

    case "get_summary": {
      if (!existsSync(OUT_FILE)) {
        return text("No summary yet. Call summarize_btc first.");
      }
      try {
        return text(readFileSync(OUT_FILE, "utf8"));
      } catch (err) {
        return text(`Error reading summary: ${err.message}`);
      }
    }

    default:
      return text(`Unknown tool: ${name}`);
  }
});

function text(content) {
  return { content: [{ type: "text", text: String(content) }] };
}

const transport = new StdioServerTransport();
await server.connect(transport);
