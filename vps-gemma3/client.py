#!/usr/bin/env python3
"""CLI client for the VPS Gemma3 API."""

import json
import os
import sys

from typing import List

import requests

API_URL = os.getenv("GEMMA3_API_URL", "http://localhost:8000")

HEADERS = {"Content-Type": "application/json"}


def stream_chat(messages: List[dict]) -> str:
    payload = {"messages": messages, "stream": True}
    full = ""

    with requests.post(f"{API_URL}/v1/chat", json=payload, headers=HEADERS,
                       stream=True, timeout=120) as resp:
        resp.raise_for_status()

        print("\n\033[92mGemma3:\033[0m ", end="", flush=True)
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode() if isinstance(line, bytes) else line
            if not line.startswith("data: "):
                continue
            payload_str = line[6:]
            if payload_str == "[DONE]":
                break
            data = json.loads(payload_str)
            chunk = data.get("content", "")
            print(chunk, end="", flush=True)
            full += chunk

    print("\n")
    return full


def check_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        info = r.json()
        print(f"\033[90mСервер: {API_URL} | модель: {info.get('model')}\033[0m\n")
        return True
    except Exception as e:
        print(f"\033[91mСервер недоступен: {e}\033[0m")
        return False


def main():
    if not check_health():
        sys.exit(1)

    print("\033[96m╔════════════════════════════════╗\033[0m")
    print("\033[96m║  Gemma3 VPS Client             ║\033[0m")
    print("\033[96m║  /clear — очистить историю     ║\033[0m")
    print("\033[96m║  /exit  — выйти                ║\033[0m")
    print("\033[96m╚════════════════════════════════╝\033[0m\n")

    history: List[dict] = []

    while True:
        try:
            user_input = input("\033[93mВы:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("Пока!")
            break
        if user_input == "/clear":
            history.clear()
            print("\033[90mИстория очищена.\033[0m\n")
            continue

        history.append({"role": "user", "content": user_input})
        reply = stream_chat(history)
        if reply:
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
