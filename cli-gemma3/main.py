import sys
import time
import json
import subprocess
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_HEALTH_URL = "http://localhost:11434"
MODEL = "gemma3"


def ensure_ollama_running():
    try:
        requests.get(OLLAMA_HEALTH_URL, timeout=2)
        return
    except requests.exceptions.ConnectionError:
        pass

    print("\033[90mЗапускаю ollama...\033[0m", flush=True)
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(20):
        time.sleep(0.5)
        try:
            requests.get(OLLAMA_HEALTH_URL, timeout=1)
            print("\033[90mollama запущена.\033[0m\n", flush=True)
            return
        except requests.exceptions.ConnectionError:
            pass

    print("\033[91mНе удалось запустить ollama. Попробуй вручную: ollama serve\033[0m")
    sys.exit(1)


def stream_response(messages: list[dict]) -> str:
    payload = {"model": MODEL, "messages": messages, "stream": True}
    full_response = ""

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            print(f"\n\033[92mGemma3:\033[0m ", end="", flush=True)
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                print(chunk, end="", flush=True)
                full_response += chunk
                if data.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        print("\n\033[91mОшибка соединения с ollama.\033[0m")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n\033[91mОшибка HTTP: {e}\033[0m")
        sys.exit(1)

    print("\n")
    return full_response


def main():
    ensure_ollama_running()

    print(f"\033[96m╔══════════════════════════════╗\033[0m")
    print(f"\033[96m║   Gemma3 CLI  (ollama)       ║\033[0m")
    print(f"\033[96m║   /clear — очистить историю  ║\033[0m")
    print(f"\033[96m║   /exit  — выйти             ║\033[0m")
    print(f"\033[96m╚══════════════════════════════╝\033[0m\n")

    history: list[dict] = []

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
        assistant_reply = stream_response(history)
        if assistant_reply:
            history.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    main()
