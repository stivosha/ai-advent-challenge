import express from "express";
import Anthropic from "@anthropic-ai/sdk";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
const client = new Anthropic({
  baseURL: "https://api.aiguoguo199.com",
});

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
    top_k,
    stop_sequences,
    thinking,
  } = req.body;

  // Build params
  const params = {
    model,
    max_tokens: max_tokens || 1024,
    messages: [{ role: "user", content: prompt }],
  };

  if (system) params.system = system;
  if (temperature !== undefined && temperature !== "") params.temperature = parseFloat(temperature);
  if (top_p !== undefined && top_p !== "") params.top_p = parseFloat(top_p);
  if (top_k !== undefined && top_k !== "") params.top_k = parseInt(top_k);
  if (stop_sequences && stop_sequences.length > 0) params.stop_sequences = stop_sequences;
  if (thinking) params.thinking = { type: "adaptive" };

  // SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  const sendEvent = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);

  try {
    const stream = client.messages.stream(params);

    let inThinking = false;

    for await (const event of stream) {
      if (event.type === "content_block_start") {
        if (event.content_block.type === "thinking") {
          inThinking = true;
          sendEvent({ type: "thinking_start" });
        } else if (event.content_block.type === "text") {
          inThinking = false;
          sendEvent({ type: "text_start" });
        }
      } else if (event.type === "content_block_delta") {
        if (event.delta.type === "thinking_delta") {
          sendEvent({ type: "thinking_delta", text: event.delta.thinking });
        } else if (event.delta.type === "text_delta") {
          sendEvent({ type: "text_delta", text: event.delta.text });
        }
      } else if (event.type === "content_block_stop") {
        if (inThinking) sendEvent({ type: "thinking_end" });
      } else if (event.type === "message_delta") {
        sendEvent({
          type: "done",
          stop_reason: event.delta.stop_reason,
          usage: event.usage,
        });
      }
    }

    const final = await stream.finalMessage();
    sendEvent({
      type: "usage",
      input_tokens: final.usage.input_tokens,
      output_tokens: final.usage.output_tokens,
    });
  } catch (err) {
    sendEvent({ type: "error", message: err.message || String(err) });
  }

  res.end();
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
