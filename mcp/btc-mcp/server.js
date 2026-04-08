import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

// ─── Paths ────────────────────────────────────────────────────────────────────
const __dir = dirname(fileURLToPath(import.meta.url));
const DATA_DIR     = join(__dir, "data");
const STATS_FILE   = join(DATA_DIR, "stats.json");
const TICKS_FILE   = join(DATA_DIR, "ticks.json");
const CANDLES_FILE = join(DATA_DIR, "candles.json");

if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });

// ─── Config ───────────────────────────────────────────────────────────────────
const TICK_BUFFER     = 3600;   // keep last 1 hour of 1-sec ticks in memory
const CANDLE_BUFFER   = 1440;   // keep last 24 hours of 1-min candles
const SOURCE_URL      = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT";

// ─── State ────────────────────────────────────────────────────────────────────
/** @type {{ ts: number, price: number }[]} */
let ticks   = [];         // ring buffer of { ts, price }
/** @type {{ time: string, open: number, high: number, low: number, close: number, ticks: number }[]} */
let candles = [];         // 1-minute OHLC candles
let currentCandle = null; // candle being built for the current minute

let timer     = null;
let running   = false;
let tickCount = 0;
let lastError = null;
let lastPrice = null;
let lastTs    = null;

// ─── Persistence helpers ──────────────────────────────────────────────────────
function loadTicks() {
  if (existsSync(TICKS_FILE)) {
    try { return JSON.parse(readFileSync(TICKS_FILE, "utf8")); } catch { /* */ }
  }
  return [];
}

function loadCandles() {
  if (existsSync(CANDLES_FILE)) {
    try { return JSON.parse(readFileSync(CANDLES_FILE, "utf8")); } catch { /* */ }
  }
  return [];
}

function flush() {
  writeFileSync(TICKS_FILE,   JSON.stringify(ticks,   null, 2), "utf8");
  writeFileSync(CANDLES_FILE, JSON.stringify(candles, null, 2), "utf8");
  writeFileSync(STATS_FILE,   JSON.stringify(buildStats(), null, 2), "utf8");
}

// ─── Statistics builder ───────────────────────────────────────────────────────
function windowStats(seconds) {
  const cutoff = Date.now() - seconds * 1000;
  const slice  = ticks.filter(t => t.ts >= cutoff).map(t => t.price);
  if (!slice.length) return null;
  const min = Math.min(...slice);
  const max = Math.max(...slice);
  const avg = slice.reduce((a, b) => a + b, 0) / slice.length;
  const first = slice[0];
  const last  = slice[slice.length - 1];
  return {
    min: +min.toFixed(2),
    max: +max.toFixed(2),
    avg: +avg.toFixed(2),
    change_abs: +(last - first).toFixed(2),
    change_pct: +(((last - first) / first) * 100).toFixed(4),
    ticks: slice.length,
  };
}

function ma(n) {
  if (ticks.length < n) return null;
  const slice = ticks.slice(-n).map(t => t.price);
  return +(slice.reduce((a, b) => a + b, 0) / n).toFixed(2);
}

function buildStats() {
  return {
    updatedAt:  lastTs,
    price:      lastPrice,
    last_error: lastError,
    total_ticks: tickCount,
    ma_20:      ma(20),
    ma_60:      ma(60),
    "1m":  windowStats(60),
    "5m":  windowStats(300),
    "15m": windowStats(900),
    "1h":  windowStats(3600),
  };
}

// ─── Candle aggregation ───────────────────────────────────────────────────────
function minuteKey(ts) {
  const d = new Date(ts);
  d.setSeconds(0, 0);
  return d.toISOString();
}

function updateCandle(price, ts) {
  const key = minuteKey(ts);

  if (!currentCandle || currentCandle.time !== key) {
    // Close previous candle
    if (currentCandle) {
      candles.push(currentCandle);
      if (candles.length > CANDLE_BUFFER) candles.shift();
    }
    // Open new candle
    currentCandle = { time: key, open: price, high: price, low: price, close: price, ticks: 1 };
  } else {
    currentCandle.high  = Math.max(currentCandle.high, price);
    currentCandle.low   = Math.min(currentCandle.low,  price);
    currentCandle.close = price;
    currentCandle.ticks++;
  }
}

// ─── Fetch & tick ─────────────────────────────────────────────────────────────
async function tick() {
  let price;
  try {
    const res = await fetch(SOURCE_URL, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    price = parseFloat(data.price);
    if (!isFinite(price)) throw new Error("Invalid price: " + data.price);
  } catch (err) {
    lastError = `${new Date().toISOString()} ${err.message}`;
    return;
  }

  const ts = Date.now();
  lastPrice = price;
  lastTs    = new Date(ts).toISOString();
  lastError = null;
  tickCount++;

  // Ring-buffer push
  ticks.push({ ts, price });
  if (ticks.length > TICK_BUFFER) ticks.shift();

  // Update current 1-min candle
  updateCandle(price, ts);

  // Flush to disk every tick
  flush();
}

// ─── Scheduler ────────────────────────────────────────────────────────────────
function start() {
  if (timer) clearInterval(timer);
  timer   = setInterval(tick, 1000);
  running = true;
  tick(); // immediate first tick
}

function stop() {
  if (timer) { clearInterval(timer); timer = null; }
  running = false;
  flush();
}

// Restore from disk and auto-start
ticks   = loadTicks();
candles = loadCandles();
start();

// ─── MCP Server ───────────────────────────────────────────────────────────────
const server = new Server(
  { name: "btc-mcp", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "btc_status",
      description: "Scheduler status, current BTC price, tick count, last error.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "btc_stats",
      description: "Full statistics: current price, moving averages, min/max/avg/change for 1m, 5m, 15m, 1h windows.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "btc_get_price",
      description: "Fetch BTC price right now (bypass scheduler, direct API call).",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "btc_get_ticks",
      description: "Return recent raw price ticks (1-second resolution).",
      inputSchema: {
        type: "object",
        properties: {
          last_n: { type: "number", description: "Number of recent ticks to return (default: 60, max: 3600)" },
        },
      },
    },
    {
      name: "btc_get_candles",
      description: "Return 1-minute OHLC candles (up to 1440 = 24 hours).",
      inputSchema: {
        type: "object",
        properties: {
          last_n: { type: "number", description: "Number of recent candles (default: 60)" },
        },
      },
    },
    {
      name: "btc_scheduler_start",
      description: "Start the 1-second price polling.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "btc_scheduler_stop",
      description: "Stop the 1-second price polling and flush data to disk.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "btc_flush",
      description: "Force-write all in-memory data (ticks, candles, stats) to disk immediately.",
      inputSchema: { type: "object", properties: {} },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name, arguments: args = {} } = req.params;

  switch (name) {

    case "btc_status": {
      return text(JSON.stringify({
        running,
        price:       lastPrice,
        updatedAt:   lastTs,
        lastError,
        total_ticks: tickCount,
        buffered_ticks: ticks.length,
        candles:     candles.length + (currentCandle ? 1 : 0),
        source:      SOURCE_URL,
      }, null, 2));
    }

    case "btc_stats": {
      return text(JSON.stringify(buildStats(), null, 2));
    }

    case "btc_get_price": {
      await tick();
      if (lastError) return text(`Error: ${lastError}`);
      return text(`BTC/USDT: $${lastPrice.toLocaleString("en-US", { minimumFractionDigits: 2 })}  (${lastTs})`);
    }

    case "btc_get_ticks": {
      const n = Math.min(Number(args.last_n) || 60, TICK_BUFFER);
      const result = ticks.slice(-n).map(t => ({
        time:  new Date(t.ts).toISOString(),
        price: t.price,
      }));
      return text(JSON.stringify(result, null, 2));
    }

    case "btc_get_candles": {
      const n = Math.min(Number(args.last_n) || 60, CANDLE_BUFFER);
      const all = currentCandle ? [...candles, currentCandle] : [...candles];
      return text(JSON.stringify(all.slice(-n), null, 2));
    }

    case "btc_scheduler_start": {
      start();
      return text("Scheduler started — polling BTC/USDT every second.");
    }

    case "btc_scheduler_stop": {
      stop();
      return text("Scheduler stopped and data flushed to disk.");
    }

    case "btc_flush": {
      flush();
      return text(`Flushed. ticks: ${ticks.length}, candles: ${candles.length}.`);
    }

    default:
      return text(`Unknown tool: ${name}`);
  }
});

function text(content) {
  return { content: [{ type: "text", text: String(content) }] };
}

// Flush on clean exit
process.on("SIGINT",  () => { flush(); process.exit(0); });
process.on("SIGTERM", () => { flush(); process.exit(0); });

const transport = new StdioServerTransport();
await server.connect(transport);
