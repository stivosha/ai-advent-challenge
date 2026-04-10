import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { writeFileSync, readFileSync, existsSync, mkdirSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dir = dirname(fileURLToPath(import.meta.url));
const DATA_DIR   = join(__dir, "data");
const CACHE_FILE = join(DATA_DIR, "hourly_24h.json");

if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });

const KLINES_URL =
  "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=24";
const PRICE_URL =
  "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT";

// ─── Fetch helpers ────────────────────────────────────────────────────────────

async function fetchCurrentPrice() {
  const res = await fetch(PRICE_URL, { signal: AbortSignal.timeout(5000) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  const price = parseFloat(data.price);
  if (!isFinite(price)) throw new Error("Invalid price");
  return price;
}

async function fetchHourly24h() {
  const res = await fetch(KLINES_URL, { signal: AbortSignal.timeout(8000) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const raw = await res.json();

  // raw row: [openTime, open, high, low, close, volume, closeTime, ...]
  const candles = raw.map((r) => ({
    time:   new Date(r[0]).toISOString(),
    open:   parseFloat(r[1]),
    high:   parseFloat(r[2]),
    low:    parseFloat(r[3]),
    close:  parseFloat(r[4]),
    volume: parseFloat(r[5]),
  }));

  const payload = { fetchedAt: new Date().toISOString(), candles };
  writeFileSync(CACHE_FILE, JSON.stringify(payload, null, 2), "utf8");
  return payload;
}

function loadCache() {
  if (!existsSync(CACHE_FILE)) return null;
  try { return JSON.parse(readFileSync(CACHE_FILE, "utf8")); } catch { return null; }
}

// ─── MCP Server ───────────────────────────────────────────────────────────────

const server = new Server(
  { name: "btc-mcp", version: "2.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "btc_fetch_24h",
      description:
        "Fetch BTC/USDT hourly OHLC candles for the last 24 hours from Binance. " +
        "Saves to disk and returns the data.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "btc_get_24h",
      description:
        "Return cached 24h hourly data (last result of btc_fetch_24h). " +
        "Returns null if no cache exists yet.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "btc_get_price",
      description: "Fetch the current BTC/USDT spot price from Binance.",
      inputSchema: { type: "object", properties: {} },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name } = req.params;

  switch (name) {

    case "btc_fetch_24h": {
      try {
        const data = await fetchHourly24h();
        return text(JSON.stringify(data, null, 2));
      } catch (err) {
        return text(`Error fetching 24h data: ${err.message}`);
      }
    }

    case "btc_get_24h": {
      const cache = loadCache();
      if (!cache) return text("No cached data. Call btc_fetch_24h first.");
      return text(JSON.stringify(cache, null, 2));
    }

    case "btc_get_price": {
      try {
        const price = await fetchCurrentPrice();
        return text(
          `BTC/USDT: $${price.toLocaleString("en-US", { minimumFractionDigits: 2 })}` +
          `  (${new Date().toISOString()})`
        );
      } catch (err) {
        return text(`Error: ${err.message}`);
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
