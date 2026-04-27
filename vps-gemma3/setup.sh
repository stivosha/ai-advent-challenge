#!/bin/bash
# Run on VPS: bash setup.sh
set -e

MODEL="${MODEL:-gemma3}"

echo "=== Installing ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

echo "=== Pulling model: $MODEL ==="
ollama pull "$MODEL"

echo "=== Installing Python deps ==="
pip3 install fastapi uvicorn requests --quiet

echo "=== Creating systemd service for ollama ==="
cat > /etc/systemd/system/ollama.service <<EOF
[Unit]
Description=Ollama
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
Environment=OLLAMA_HOST=127.0.0.1:11434
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now ollama

echo "=== Creating systemd service for gemma3-api ==="
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cat > /etc/systemd/system/gemma3-api.service <<EOF
[Unit]
Description=Gemma3 API Server
After=ollama.service

[Service]
ExecStart=/usr/bin/python3 $SCRIPT_DIR/server.py
Environment=OLLAMA_MODEL=$MODEL
Restart=always
RestartSec=3
WorkingDirectory=$SCRIPT_DIR

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now gemma3-api

echo ""
echo "=== Done ==="
echo "API is running on http://0.0.0.0:8000"
