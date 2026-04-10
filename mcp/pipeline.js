#!/usr/bin/env node
/**
 * BTC Pipeline
 * Шаг 1: btc_fetch_24h    → btc-mcp
 * Шаг 2: summarize_btc    → mcp-summarize
 * Шаг 3: save_report      → mcp-save
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dir = dirname(fileURLToPath(import.meta.url));

// ─── Helpers ──────────────────────────────────────────────────────────────────

function log(step, msg) {
  const time = new Date().toLocaleTimeString("ru-RU");
  console.log(`\n[${time}] ── Step ${step} ──────────────────────────────`);
  console.log(msg);
}

function hr() {
  console.log("─".repeat(55));
}

async function runTool(serverPath, toolName, toolArgs = {}) {
  const client = new Client(
    { name: "btc-pipeline", version: "1.0.0" },
    { capabilities: {} }
  );

  const transport = new StdioClientTransport({
    command: "node",
    args: [serverPath],
  });

  await client.connect(transport);

  let result;
  try {
    result = await client.callTool({ name: toolName, arguments: toolArgs });
  } finally {
    await client.close();
  }

  // Extract text from MCP content array
  const text = result?.content
    ?.filter(c => c.type === "text")
    .map(c => c.text)
    .join("\n") ?? "";

  return text;
}

// ─── Pipeline ─────────────────────────────────────────────────────────────────

async function run() {
  console.log("╔═══════════════════════════════════════════════════╗");
  console.log("║          BTC ANALYSIS PIPELINE                   ║");
  console.log("╚═══════════════════════════════════════════════════╝");

  // ── Step 1: Fetch 24h hourly data ─────────────────────────────────────────
  hr();
  console.log("STEP 1/3  btc_fetch_24h  (btc-mcp)");
  hr();

  let fetchResult;
  try {
    fetchResult = await runTool(
      join(__dir, "btc-mcp/server.js"),
      "btc_fetch_24h"
    );
    // Parse to show a compact summary instead of dumping 24 candles
    const parsed = JSON.parse(fetchResult);
    const candles = parsed.candles;
    const first = candles[0];
    const last  = candles[candles.length - 1];
    log(1,
      `Fetched ${candles.length} hourly candles\n` +
      `  From  : ${first.time}\n` +
      `  To    : ${last.time}\n` +
      `  Open  : $${first.open.toLocaleString("en-US", { minimumFractionDigits: 2 })}\n` +
      `  Close : $${last.close.toLocaleString("en-US", { minimumFractionDigits: 2 })}\n` +
      `  Data saved to btc-mcp/data/hourly_24h.json`
    );
  } catch (err) {
    console.error(`STEP 1 FAILED: ${err.message}`);
    process.exit(1);
  }

  // ── Step 2: Summarize & forecast ──────────────────────────────────────────
  hr();
  console.log("STEP 2/3  summarize_btc  (mcp-summarize)");
  hr();

  let summary;
  try {
    const summaryRaw = await runTool(
      join(__dir, "mcp-summarize/server.js"),
      "summarize_btc"
    );
    summary = JSON.parse(summaryRaw);
    const f = summary.forecast;
    const arrow = f.direction === "UP" ? "▲" : f.direction === "DOWN" ? "▼" : "→";
    log(2,
      `Technical analysis complete\n` +
      `  Price   : $${summary.current_price.toLocaleString("en-US", { minimumFractionDigits: 2 })}\n` +
      `  Change  : ${summary.change_24h_pct >= 0 ? "+" : ""}${summary.change_24h_pct}%\n` +
      `  EMA7    : $${summary.ema7}  EMA14: $${summary.ema14}\n` +
      `  RSI14   : ${summary.rsi14}\n` +
      `  Trend   : ${summary.trend}\n` +
      `  ──────────────────────────────────\n` +
      `  Forecast: ${arrow} ${f.direction}  (confidence: ${f.confidence.toUpperCase()})\n` +
      `  Target  : $${f.estimated_next_target.toLocaleString("en-US", { minimumFractionDigits: 2 })}\n` +
      `  Signals :\n` +
      f.signals.map(s => `    • ${s}`).join("\n") +
      `\n  Summary saved to mcp-summarize/data/summary.json`
    );
  } catch (err) {
    console.error(`STEP 2 FAILED: ${err.message}`);
    process.exit(1);
  }

  // ── Step 3: Save report ────────────────────────────────────────────────────
  hr();
  console.log("STEP 3/3  save_report  (mcp-save)");
  hr();

  try {
    const saveResult = await runTool(
      join(__dir, "mcp-save/server.js"),
      "save_report"
    );
    // First line is "Report saved to:\n<path>"
    const firstLine = saveResult.split("\n").slice(0, 2).join(" ");
    log(3, firstLine);
    console.log("\n" + saveResult.split("\n").slice(2).join("\n"));
  } catch (err) {
    console.error(`STEP 3 FAILED: ${err.message}`);
    process.exit(1);
  }

  console.log("\n╔═══════════════════════════════════════════════════╗");
  console.log("║           PIPELINE COMPLETE ✓                     ║");
  console.log("╚═══════════════════════════════════════════════════╝\n");
}

run().catch(err => {
  console.error("Pipeline error:", err);
  process.exit(1);
});
