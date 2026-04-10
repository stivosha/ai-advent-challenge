import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { writeFileSync, readFileSync, existsSync, mkdirSync, readdirSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dir      = dirname(fileURLToPath(import.meta.url));
const REPORTS_DIR = join(__dir, "../reports");
const SUMMARY_FILE = join(__dir, "../mcp-summarize/data/summary.json");

if (!existsSync(REPORTS_DIR)) mkdirSync(REPORTS_DIR, { recursive: true });

// ─── Formatting ────────────────────────────────────────────────────────────────

function formatReport(summary) {
  const f = summary.forecast;
  const ts = new Date(summary.generatedAt).toLocaleString("ru-RU", { timeZone: "UTC" }) + " UTC";

  const arrow = f.direction === "UP" ? "▲" : f.direction === "DOWN" ? "▼" : "→";
  const lines = [
    "═══════════════════════════════════════════════════════",
    "        BTC/USDT — АНАЛИЗ ЗА 24 ЧАСА",
    "═══════════════════════════════════════════════════════",
    `Дата анализа : ${ts}`,
    `Данные от    : ${new Date(summary.dataFetchedAt).toLocaleString("ru-RU", { timeZone: "UTC" })} UTC`,
    "───────────────────────────────────────────────────────",
    "ЦЕНОВЫЕ ДАННЫЕ",
    "───────────────────────────────────────────────────────",
    `Текущая цена : $${summary.current_price.toLocaleString("en-US", { minimumFractionDigits: 2 })}`,
    `Открытие 24h : $${summary.open_24h.toLocaleString("en-US", { minimumFractionDigits: 2 })}`,
    `Изменение    : ${summary.change_24h_pct >= 0 ? "+" : ""}${summary.change_24h_pct}%`,
    `Максимум 24h : $${summary.high_24h.toLocaleString("en-US", { minimumFractionDigits: 2 })}`,
    `Минимум 24h  : $${summary.low_24h.toLocaleString("en-US", { minimumFractionDigits: 2 })}`,
    "───────────────────────────────────────────────────────",
    "ИНДИКАТОРЫ",
    "───────────────────────────────────────────────────────",
    `EMA(7)       : $${summary.ema7 ?? "—"}`,
    `EMA(14)      : $${summary.ema14 ?? "—"}`,
    `RSI(14)      : ${summary.rsi14 ?? "—"}`,
    `Тренд        : ${summary.trend}`,
    `Объём (avg)  : ${summary.volume.avg_volume.toLocaleString("en-US", { maximumFractionDigits: 2 })}`,
    `Объём (last) : ${summary.volume.last_volume.toLocaleString("en-US", { maximumFractionDigits: 2 })} (×${summary.volume.volume_ratio})`,
    "───────────────────────────────────────────────────────",
    "ПРОГНОЗ",
    "───────────────────────────────────────────────────────",
    `Направление  : ${arrow} ${f.direction}`,
    `Уверенность  : ${f.confidence.toUpperCase()}`,
    `Цель         : $${f.estimated_next_target.toLocaleString("en-US", { minimumFractionDigits: 2 })}`,
    `Bull score   : ${f.bull_score}`,
    `Bear score   : ${f.bear_score}`,
    "",
    "Сигналы:",
    ...f.signals.map(s => `  • ${s}`),
    "───────────────────────────────────────────────────────",
    `Примечание   : ${f.note}`,
    "═══════════════════════════════════════════════════════",
  ];
  return lines.join("\n");
}

// ─── MCP Server ───────────────────────────────────────────────────────────────

const server = new Server(
  { name: "mcp-save", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "save_report",
      description:
        "Read the latest BTC summary from mcp-summarize and save it as a " +
        "formatted .txt report in the reports/ folder. Returns the file path.",
      inputSchema: {
        type: "object",
        properties: {
          filename: {
            type: "string",
            description:
              "Optional custom filename (without extension). " +
              "Defaults to btc_report_<timestamp>.",
          },
        },
      },
    },
    {
      name: "list_reports",
      description: "List all saved BTC report files.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "read_report",
      description: "Read a specific report file by filename.",
      inputSchema: {
        type: "object",
        properties: {
          filename: {
            type: "string",
            description: "Filename (with or without .txt extension).",
          },
        },
        required: ["filename"],
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name, arguments: args = {} } = req.params;

  switch (name) {

    case "save_report": {
      if (!existsSync(SUMMARY_FILE)) {
        return text(
          "No summary found. Run summarize_btc in mcp-summarize first."
        );
      }
      let summary;
      try {
        summary = JSON.parse(readFileSync(SUMMARY_FILE, "utf8"));
      } catch (err) {
        return text(`Failed to read summary: ${err.message}`);
      }

      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      const base  = args.filename
        ? String(args.filename).replace(/[^a-zA-Z0-9_-]/g, "_")
        : `btc_report_${stamp}`;
      const filepath = join(REPORTS_DIR, `${base}.txt`);

      const content = formatReport(summary);
      writeFileSync(filepath, content, "utf8");
      return text(`Report saved to:\n${filepath}\n\n${content}`);
    }

    case "list_reports": {
      if (!existsSync(REPORTS_DIR)) return text("No reports directory found.");
      const files = readdirSync(REPORTS_DIR)
        .filter(f => f.endsWith(".txt"))
        .sort()
        .reverse();
      if (files.length === 0) return text("No reports yet.");
      return text(files.map((f, i) => `${i + 1}. ${f}`).join("\n"));
    }

    case "read_report": {
      let fname = String(args.filename || "");
      if (!fname.endsWith(".txt")) fname += ".txt";
      const filepath = join(REPORTS_DIR, fname);
      if (!existsSync(filepath)) return text(`File not found: ${fname}`);
      try {
        return text(readFileSync(filepath, "utf8"));
      } catch (err) {
        return text(`Error reading file: ${err.message}`);
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
