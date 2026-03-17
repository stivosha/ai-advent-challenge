import express from "express";
import { randomUUID } from "crypto";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

// GigaChat uses a certificate signed by Russian NCA (НУЦ Минцифры),
// not trusted by default in Node.js — disable verification for dev.
process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();

const GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth";
const GIGACHAT_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions";

// Token cache — refresh 1 min before expiry
let cachedToken = null;
let tokenExpiresAt = 0; // Unix seconds

async function getAccessToken() {
  if (cachedToken && Date.now() / 1000 < tokenExpiresAt - 60) {
    return cachedToken;
  }
  const res = await fetch(GIGACHAT_AUTH_URL, {
    method: "POST",
    headers: {
      "Authorization": `Basic ${process.env.GIGACHAT_API_KEY}`,
      "RqUID": randomUUID(),
      "Content-Type": "application/x-www-form-urlencoded"
    },
    body: "scope=GIGACHAT_API_PERS",
  });

  if (!res.ok) {
    throw new Error(`GigaChat auth failed: ${res.status} ${await res.text()}`);
  }

  const data = await res.json();
  cachedToken = data.access_token;
  tokenExpiresAt = data.expires_at; // seconds
  return cachedToken;
}

app.use(express.json());
app.use(express.static(join(__dirname, "public")));

app.post("/api/chat", async (req, res) => {
  const {
    model,
    prompt,
    system,
    temperature,
    max_tokens,
    top_p,
    stop_sequences,
  } = req.body;

  const messages = [];
  if (system) messages.push({ role: "system", content: system });
  messages.push({ role: "user", content: prompt });

  const params = {
    model: model || "GigaChat",
    messages,
    stream: true,
    max_tokens: max_tokens || 1024,
  };

  if (temperature !== undefined && temperature !== "") params.temperature = parseFloat(temperature);
  if (top_p !== undefined && top_p !== "")           params.top_p = parseFloat(top_p);
  if (stop_sequences && stop_sequences.length > 0)   params.stop = stop_sequences;

  console.log("[REQUEST PARAMS]", JSON.stringify(params, null, 2));

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  const sendEvent = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);

  try {
    const token = await getAccessToken();

    const apiRes = await fetch(GIGACHAT_CHAT_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(params),
    });

    if (!apiRes.ok) {
      throw new Error(`GigaChat API error: ${apiRes.status} ${await apiRes.text()}`);
    }

    const reader = apiRes.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let started = false;
    let inputTokens = 0;
    let outputTokens = 0;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const lines = buf.split("\n");
      buf = lines.pop(); // keep last incomplete line

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;

        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") continue;

        let chunk;
        try { chunk = JSON.parse(payload); } catch { continue; }

        const delta = chunk.choices?.[0]?.delta?.content;
        if (delta != null) {
          if (!started) { sendEvent({ type: "text_start" }); started = true; }
          if (delta)    sendEvent({ type: "text_delta", text: delta });
        }

        if (chunk.usage) {
          inputTokens  = chunk.usage.prompt_tokens     ?? inputTokens;
          outputTokens = chunk.usage.completion_tokens ?? outputTokens;
        }
      }
    }

    sendEvent({ type: "usage", input_tokens: inputTokens, output_tokens: outputTokens });
  } catch (err) {
    sendEvent({ type: "error", message: err.message || String(err) });
  }

  res.end();
});

const PORT = 3000;
app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
