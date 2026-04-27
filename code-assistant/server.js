/**
 * Code Assistant — MCP server
 *
 * Tools:
 *   help({ question, project_path? })     — RAG answer (Ollama/gemma3) over generated docs
 *   generate_docs({ path? })              — generate docs via GigaChat Pro
 *   index_project({ path, extensions? })  — index docs for RAG (BM25)
 *   git_branch/status/log/list_files      — git helpers
 *
 * Env:
 *   GIGACHAT_CREDENTIALS — base64(clientId:secret), required for generate_docs
 *   GIGACHAT_SCOPE       (default: GIGACHAT_API_CORP)
 *   GIGACHAT_MODEL       (default: GigaChat-Pro)
 *   OLLAMA_URL           (default: http://localhost:11434)
 *   OLLAMA_MODEL         (default: gemma3)
 *   PROJECT_PATH         — pre-set project path (optional)
 */

import { randomUUID } from "crypto";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { execFile } from "child_process";
import { promisify } from "util";
import { readdir, readFile, stat, writeFile, mkdir } from "fs/promises";
import { join, extname } from "path";

const execFileAsync = promisify(execFile);

// ─── Config ───────────────────────────────────────────────────────────────────

const OLLAMA_URL   = process.env.OLLAMA_URL   ?? "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL ?? "gemma3";

const GIGACHAT_CREDENTIALS = process.env.GIGACHAT_CREDENTIALS ?? process.env.GIGACHAT_API_KEY ?? null;
const GIGACHAT_SCOPE       = process.env.GIGACHAT_SCOPE       ?? "GIGACHAT_API_CORP";
const GIGACHAT_MODEL       = process.env.GIGACHAT_MODEL       ?? "GigaChat-Pro";

const DEFAULT_EXTENSIONS = [".md", ".txt", ".py", ".js", ".ts", ".yaml", ".yml", ".json",
                            ".kt", ".kts", ".java", ".xml", ".gradle"];
const IGNORE_DIRS        = new Set(["node_modules", ".git", "__pycache__", "venv", ".venv",
                                    "dist", "build", "index", ".mypy_cache", "coverage"]);
const MAX_FILE_BYTES     = 120_000;
const CHUNK_SIZE         = 900;
const CHUNK_OVERLAP      = 120;

// ─── BM25 ─────────────────────────────────────────────────────────────────────

class BM25 {
  constructor(k1 = 1.5, b = 0.75) {
    this.k1 = k1;
    this.b  = b;
    this.docs   = [];
    this.idf    = {};
    this.avgdl  = 0;
  }

  _tok(text) {
    return text.toLowerCase().replace(/[^\p{L}\p{N}]/gu, " ").split(/\s+/).filter(t => t.length > 2);
  }

  index(docs) {
    this.docs = docs.map(d => ({ ...d, toks: this._tok(d.text) }));
    this.avgdl = this.docs.reduce((s, d) => s + d.toks.length, 0) / (this.docs.length || 1);

    const df = {};
    for (const d of this.docs) {
      for (const t of new Set(d.toks)) df[t] = (df[t] ?? 0) + 1;
    }
    const N = this.docs.length;
    for (const [t, f] of Object.entries(df)) {
      this.idf[t] = Math.log((N - f + 0.5) / (f + 0.5) + 1);
    }
  }

  search(query, topK = 5) {
    const qToks = this._tok(query);
    return this.docs
      .map((d, i) => {
        const tf = {};
        for (const t of d.toks) tf[t] = (tf[t] ?? 0) + 1;
        let score = 0;
        for (const t of qToks) {
          if (!this.idf[t]) continue;
          const f  = tf[t] ?? 0;
          const dl = d.toks.length;
          score += this.idf[t] * (f * (this.k1 + 1)) /
                   (f + this.k1 * (1 - this.b + this.b * dl / this.avgdl));
        }
        return { score, idx: i };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .filter(r => r.score > 0)
      .map(r => this.docs[r.idx]);
  }
}

// ─── Chunker ──────────────────────────────────────────────────────────────────

function chunk(text, source) {
  const parts = [];
  // Split on markdown headings to preserve section context
  const sections = text.split(/\n(?=#{1,4} )/);
  let idx = 0;
  for (const sec of sections) {
    if (sec.length <= CHUNK_SIZE) {
      const t = sec.trim();
      if (t.length > 40) parts.push({ source, idx: idx++, text: t });
    } else {
      let s = 0;
      while (s < sec.length) {
        const t = sec.slice(s, s + CHUNK_SIZE).trim();
        if (t.length > 40) parts.push({ source, idx: idx++, text: t });
        s += CHUNK_SIZE - CHUNK_OVERLAP;
      }
    }
  }
  return parts;
}

// ─── File walker ──────────────────────────────────────────────────────────────

async function walkFiles(dir, exts) {
  const found = [];
  async function walk(cur) {
    let entries;
    try { entries = await readdir(cur, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      if (IGNORE_DIRS.has(e.name)) continue;
      const full = join(cur, e.name);
      if (e.isDirectory()) { await walk(full); }
      else if (e.isFile() && exts.has(extname(e.name).toLowerCase())) {
        try {
          const { size } = await stat(full);
          if (size <= MAX_FILE_BYTES) found.push(full);
        } catch {}
      }
    }
  }
  await walk(dir);
  return found;
}

// ─── Index state ─────────────────────────────────────────────────────────────

let bm25             = null;
let indexedPath      = process.env.PROJECT_PATH ?? null;
let indexStats       = null;
const docsGeneratedFor = new Set();

// Replaced with real MCP logging after server is created
let log = (_msg) => Promise.resolve();

async function buildIndex(projectPath, exts = DEFAULT_EXTENSIONS) {
  const extSet = new Set(exts);
  const files  = await walkFiles(projectPath, extSet);
  const chunks = [];

  for (const f of files) {
    try {
      const text = await readFile(f, "utf8");
      const rel  = f.slice(projectPath.length).replace(/^[\\/]/, "");
      chunks.push(...chunk(text, rel));
    } catch {}
  }

  if (!chunks.length) throw new Error(`No indexable files found in: ${projectPath}`);

  bm25        = new BM25();
  bm25.index(chunks);
  indexedPath = projectPath;
  indexStats  = { files: files.length, chunks: chunks.length };
  return indexStats;
}

// ─── Git ─────────────────────────────────────────────────────────────────────

async function git(args, cwd) {
  try {
    const { stdout } = await execFileAsync("git", args, { cwd, windowsHide: true, maxBuffer: 2_000_000 });
    return stdout.trim();
  } catch (err) {
    return (err.stderr ?? err.message ?? "git error").trim();
  }
}

// ─── Ollama ───────────────────────────────────────────────────────────────────

async function askOllama(system, user, timeoutMs = 55_000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(`${OLLAMA_URL}/api/chat`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      signal:  controller.signal,
      body: JSON.stringify({
        model:    OLLAMA_MODEL,
        messages: [
          { role: "system", content: system },
          { role: "user",   content: user   },
        ],
        stream:  false,
        options: { temperature: 0.1, num_predict: 300, num_ctx: 1024 },
      }),
    });
    if (!resp.ok) throw new Error(`Ollama ${resp.status}: ${await resp.text()}`);
    return (await resp.json()).message.content.trim();
  } finally {
    clearTimeout(timer);
  }
}

// ─── Dir tree ────────────────────────────────────────────────────────────────

async function dirTree(dir, depth = 2, prefix = "") {
  const lines = [];
  let entries;
  try { entries = await readdir(dir, { withFileTypes: true }); } catch { return lines; }
  const dirs  = entries.filter(e => e.isDirectory() && !IGNORE_DIRS.has(e.name));
  const files = entries.filter(e => e.isFile());
  for (const d of dirs) {
    lines.push(`${prefix}${d.name}/`);
    if (depth > 1) lines.push(...await dirTree(join(dir, d.name), depth - 1, prefix + "  "));
  }
  for (const f of files.slice(0, 10)) lines.push(`${prefix}${f.name}`);
  return lines;
}

// ─── GigaChat API ────────────────────────────────────────────────────────────

let _gcToken        = null;
let _gcExpiry       = 0;
let _gcTokenFlight  = null;

async function getGigaChatToken() {
  if (_gcToken && Date.now() < _gcExpiry) return _gcToken;
  // Dedup concurrent requests — only one OAuth call at a time
  if (_gcTokenFlight) return _gcTokenFlight;
  if (!GIGACHAT_CREDENTIALS) throw new Error("GIGACHAT_CREDENTIALS not set");

  _gcTokenFlight = (async () => {
    const resp = await fetch("https://ngw.devices.sberbank.ru:9443/api/v2/oauth", {
      method:  "POST",
      headers: {
        "Authorization": `Basic ${GIGACHAT_CREDENTIALS}`,
        "RqUID":         randomUUID(),
        "Content-Type":  "application/x-www-form-urlencoded",
      },
      body: `scope=${GIGACHAT_SCOPE}`,
    });
    if (!resp.ok) throw new Error(`GigaChat auth ${resp.status}: ${await resp.text()}`);
    const data = await resp.json();
    _gcToken   = data.access_token;
    _gcExpiry  = (data.expires_at ?? Date.now() + 29 * 60 * 1000) - 10_000;
    return _gcToken;
  })().finally(() => { _gcTokenFlight = null; });

  return _gcTokenFlight;
}

async function askGigaChat(system, user, timeoutMs = 120_000) {
  const controller = new AbortController();
  const timer      = setTimeout(() => controller.abort(), timeoutMs);
  try {
    for (let attempt = 0; attempt < 4; attempt++) {
      const token = await getGigaChatToken();
      const resp  = await fetch("https://gigachat.devices.sberbank.ru/api/v1/chat/completions", {
        method:  "POST",
        headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
        signal:  controller.signal,
        body: JSON.stringify({
          model:       GIGACHAT_MODEL,
          messages:    [{ role: "system", content: system }, { role: "user", content: user }],
          temperature: 0.1,
          max_tokens:  2048,
          stream:      false,
        }),
      });
      if (resp.status === 429) {
        await new Promise(r => setTimeout(r, 2000 * (attempt + 1)));
        continue;
      }
      if (!resp.ok) throw new Error(`GigaChat ${resp.status}: ${await resp.text()}`);
      return (await resp.json()).choices[0].message.content.trim();
    }
    throw new Error("GigaChat: too many retries (rate limit)");
  } finally {
    clearTimeout(timer);
  }
}

// ─── RAG answer (Ollama) ──────────────────────────────────────────────────────

async function ragAnswer(question, projectPath) {
  if (!bm25 || indexedPath !== projectPath) {
    await buildIndex(projectPath);
  }

  const hits    = bm25.search(question, 3);
  const context = hits.length
    ? hits.map(h => `### ${h.source}\n${h.text.slice(0, 500)}`).join("\n\n---\n\n")
    : "(no matches)";

  const system  = "You are a developer assistant. Answer briefly (2-4 sentences) in the same language as the question. Use only the provided context.";
  const userMsg = `${context}\n\nQuestion: ${question}`;

  return askOllama(system, userMsg);
}

// ─── Docs generation ─────────────────────────────────────────────────────────

async function generateDocs(projectPath) {
  await log("📂 Scanning project files...");
  const extSet  = new Set(DEFAULT_EXTENSIONS);
  const files   = await walkFiles(projectPath, extSet);
  const tree    = (await dirTree(projectPath, 3)).join("\n");
  const branch  = await git(["rev-parse", "--abbrev-ref", "HEAD"], projectPath);
  const gitLog  = await git(["log", "--oneline", "-10"],            projectPath);

  const sample = files.slice(0, 20);
  let codeDump = "";
  for (const f of sample) {
    try {
      const rel  = f.slice(projectPath.length).replace(/^[\\/]/, "");
      const text = (await readFile(f, "utf8")).slice(0, 2000);
      codeDump  += `\n### ${rel}\n\`\`\`\n${text}\n\`\`\`\n`;
    } catch {}
  }

  const sysPrompt =
    "Ты технический писатель. Генерируй чёткую и лаконичную Markdown-документацию для программного проекта. " +
    "Опирайся строго на предоставленный код и структуру. Пиши на русском языке.";

  await log("✍️  [1/4] Generating README.md via GigaChat...");
  const readme = await askGigaChat(sysPrompt,
    `Project tree:\n${tree}\n\nGit log:\n${gitLog}\n\nCode samples:\n${codeDump}\n\n` +
    "Generate a README.md with sections: Project Overview, Features, Architecture, Getting Started, Project Structure.");

  await log("✍️  [2/4] Generating docs/architecture.md...");
  const arch = await askGigaChat(sysPrompt,
    `Project tree:\n${tree}\n\nCode samples:\n${codeDump}\n\n` +
    "Generate docs/architecture.md describing: layers, modules, key design patterns, data flow, dependencies.");

  await log("✍️  [3/4] Generating docs/data-models.md...");
  const dataModels = await askGigaChat(sysPrompt,
    `Code samples:\n${codeDump}\n\n` +
    "Generate docs/data-models.md listing all data classes, entities, enums with their fields and purpose. " +
    "Use Markdown tables where appropriate.");

  await log("✍️  [4/4] Generating docs/api.md...");
  const api = await askGigaChat(sysPrompt,
    `Code samples:\n${codeDump}\n\n` +
    "Generate docs/api.md describing public interfaces, repositories, ViewModels, use-cases and their methods.");

  await log("💾 Writing documentation files...");
  const docsDir = join(projectPath, "docs");
  await mkdir(docsDir, { recursive: true });
  await writeFile(join(projectPath, "README.md"),       readme,     "utf8");
  await writeFile(join(docsDir,     "architecture.md"), arch,       "utf8");
  await writeFile(join(docsDir,     "data-models.md"),  dataModels, "utf8");
  await writeFile(join(docsDir,     "api.md"),           api,        "utf8");

  await log("🔍 Indexing project for RAG...");
  docsGeneratedFor.add(projectPath);
  return await buildIndex(projectPath);
}

// ─── Tools ────────────────────────────────────────────────────────────────────

const TOOLS = [
  {
    name: "help",
    description:
      "Answer a question about the project using RAG over its docs/code and the current git context. " +
      "The project path is pre-configured via PROJECT_PATH env — just pass the question.",
    inputSchema: {
      type: "object",
      required: ["question"],
      properties: {
        question:     { type: "string", description: "Your question about the project" },
        project_path: { type: "string", description: "Override project path (optional, uses PROJECT_PATH env by default)" },
      },
    },
    async handler({ question, project_path }) {
      const path = project_path ?? indexedPath;
      if (!path) return err("Pass project_path or set PROJECT_PATH env var, then call index_project.");

      if (!bm25 || indexedPath !== path) {
        await log("🔍 Indexing project...");
        await buildIndex(path);
      }

      await log("🤔 Searching for relevant context...");
      return ok(await ragAnswer(question, path));
    },
  },

  {
    name: "index_project",
    description: "Index a project's documentation and source files for RAG search.",
    inputSchema: {
      type: "object",
      required: ["path"],
      properties: {
        path: { type: "string", description: "Absolute path to the project root" },
        extensions: {
          type: "array",
          items: { type: "string" },
          description: "Extensions to index (default: .md .txt .py .js .ts .yaml .yml .json)",
        },
      },
    },
    async handler({ path, extensions }) {
      const stats = await buildIndex(path, extensions ?? DEFAULT_EXTENSIONS);
      return ok(`Indexed ${stats.files} files → ${stats.chunks} chunks\nProject: ${path}`);
    },
  },

  {
    name: "git_branch",
    description: "Get the current git branch of a project.",
    inputSchema: {
      type: "object",
      required: ["path"],
      properties: { path: { type: "string" } },
    },
    async handler({ path }) {
      return ok(await git(["rev-parse", "--abbrev-ref", "HEAD"], path));
    },
  },

  {
    name: "git_status",
    description: "Get git working-tree status (staged, unstaged, untracked files).",
    inputSchema: {
      type: "object",
      required: ["path"],
      properties: { path: { type: "string" } },
    },
    async handler({ path }) {
      const branch = await git(["rev-parse", "--abbrev-ref", "HEAD"], path);
      const status = await git(["status", "--short"], path);
      return ok(`Branch: ${branch}\n${status || "(clean)"}`);
    },
  },

  {
    name: "git_log",
    description: "Show recent git commits.",
    inputSchema: {
      type: "object",
      required: ["path"],
      properties: {
        path:  { type: "string" },
        count: { type: "number", description: "Number of commits (default 10)" },
      },
    },
    async handler({ path, count = 10 }) {
      return ok(await git(["log", "--oneline", `-${count}`], path));
    },
  },

  {
    name: "generate_docs",
    description:
      "Analyse project source code and generate README.md, docs/architecture.md, " +
      "docs/data-models.md, and docs/api.md. Writes files to the project, then re-indexes for RAG.",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "Project root (uses PROJECT_PATH env by default)" },
      },
    },
    async handler({ path }) {
      const projectPath = path ?? indexedPath;
      if (!projectPath) return err("Pass path or set PROJECT_PATH env var.");
      const stats = await generateDocs(projectPath);
      return ok(
        `Generated and written:\n` +
        `  README.md\n` +
        `  docs/architecture.md\n` +
        `  docs/data-models.md\n` +
        `  docs/api.md\n\n` +
        `Re-indexed: ${stats.files} files → ${stats.chunks} chunks`
      );
    },
  },

  {
    name: "list_files",
    description: "List tracked files in the project.",
    inputSchema: {
      type: "object",
      required: ["path"],
      properties: {
        path:    { type: "string" },
        pattern: { type: "string", description: "Glob, e.g. '*.py'" },
      },
    },
    async handler({ path, pattern }) {
      const args = ["ls-files"];
      if (pattern) args.push("--", pattern);
      return ok(await git(args, path) || "(no files)");
    },
  },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function ok(text) {
  return { content: [{ type: "text", text: String(text) }] };
}
function err(text) {
  return { content: [{ type: "text", text: String(text) }], isError: true };
}

// ─── Server ───────────────────────────────────────────────────────────────────

const server = new Server(
  { name: "code-assistant", version: "1.0.0" },
  { capabilities: { tools: {}, logging: {} } }
);

log = (data) => server.sendLoggingMessage({ level: "info", data }).catch(() => {});

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: TOOLS.map(({ name, description, inputSchema }) => ({ name, description, inputSchema })),
}));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const tool = TOOLS.find(t => t.name === req.params.name);
  if (!tool) return err(`Unknown tool: ${req.params.name}`);
  try {
    return await tool.handler(req.params.arguments ?? {});
  } catch (e) {
    return err(e.message);
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
