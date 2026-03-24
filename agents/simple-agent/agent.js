import { randomUUID } from "crypto";

// GigaChat uses a certificate signed by Russian NCA — disable TLS verification for dev.
process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

// ─── Hardcoded agent parameters ───────────────────────────────────────────────
const CONFIG = {
  model:       "GigaChat",
  system:      "Ты — полезный AI-ассистент. Отвечай чётко и по делу.",
  temperature: 0.7,
  max_tokens:  1024,
  top_p:       0.9,
};

const GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth";
const GIGACHAT_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions";

// ─── Token cache ───────────────────────────────────────────────────────────────
let cachedToken   = null;
let tokenExpiresAt = 0;

async function getAccessToken() {
  if (cachedToken && Date.now() / 1000 < tokenExpiresAt - 60) {
    return cachedToken;
  }

  const apiKey = process.env.GIGACHAT_API_KEY;
  if (!apiKey) throw new Error("GIGACHAT_API_KEY environment variable is not set");

  const res = await fetch(GIGACHAT_AUTH_URL, {
    method: "POST",
    headers: {
      "Authorization":  `Basic ${apiKey}`,
      "RqUID":          randomUUID(),
      "Content-Type":   "application/x-www-form-urlencoded",
    },
    body: "scope=GIGACHAT_API_PERS",
  });

  if (!res.ok) throw new Error(`GigaChat auth failed: ${res.status} ${await res.text()}`);

  const data     = await res.json();
  cachedToken    = data.access_token;
  tokenExpiresAt = data.expires_at;
  return cachedToken;
}

// ─── Core: send query, return full response text ───────────────────────────────
export async function ask(userQuery) {
  if (!userQuery?.trim()) throw new Error("Query must not be empty");

  const messages = [
    { role: "system", content: CONFIG.system },
    { role: "user",   content: userQuery },
  ];

  const payload = {
    model:       CONFIG.model,
    messages,
    temperature: CONFIG.temperature,
    max_tokens:  CONFIG.max_tokens,
    top_p:       CONFIG.top_p,
    stream:      true,
  };

  const token  = await getAccessToken();
  const apiRes = await fetch(GIGACHAT_CHAT_URL, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Content-Type":  "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!apiRes.ok) throw new Error(`GigaChat API error: ${apiRes.status} ${await apiRes.text()}`);

  // Stream → collect full text
  const reader  = apiRes.body.getReader();
  const decoder = new TextDecoder();
  let buf    = "";
  let result = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    const lines = buf.split("\n");
    buf = lines.pop();

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;

      const raw = trimmed.slice(5).trim();
      if (raw === "[DONE]") continue;

      let chunk;
      try { chunk = JSON.parse(raw); } catch { continue; }

      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) result += delta;
    }
  }

  return result;
}

// ─── CLI entry point ───────────────────────────────────────────────────────────
// Usage: node agent.js "Твой вопрос здесь"
if (process.argv[1] && new URL(import.meta.url).pathname.endsWith(process.argv[1].replace(/\\/g, "/"))) {
  const query = process.argv.slice(2).join(" ");
  if (!query) {
    console.error("Usage: node agent.js \"<your question>\"");
    process.exit(1);
  }

  ask(query)
    .then((answer) => console.log(answer))
    .catch((err)   => { console.error(err.message); process.exit(1); });
}
