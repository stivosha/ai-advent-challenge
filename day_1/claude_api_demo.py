import anthropic

client = anthropic.Anthropic()  # читает ANTHROPIC_API_KEY из окружения

MODEL = "claude-opus-4-6"
PROMPT = "Напиши короткое стихотворение о программировании (2-4 строки)."


def run(label: str, **kwargs):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    for k, v in kwargs.items():
        if k != "messages":
            print(f"  {k}: {v}")
    print()

    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        **kwargs,
    )

    for block in response.content:
        if block.type == "text":
            print(block.text)

    usage = response.usage
    print(f"\n  [input: {usage.input_tokens} tok | output: {usage.output_tokens} tok | stop: {response.stop_reason}]")


# 1. Базовый запрос
run("Базовый запрос (max_tokens=256)", max_tokens=256)

# 2. Высокая температура — более творческие ответы
run("Высокая температура (temperature=1.0)", max_tokens=256, temperature=1.0)

# 3. Низкая температура — более детерминированные ответы
run("Низкая температура (temperature=0.0)", max_tokens=256, temperature=0.0)

# 4. С системным промптом
run(
    "С системным промптом",
    max_tokens=256,
    temperature=0.7,
    system="Ты поэт-минималист. Отвечай только одной строкой.",
)

# 5. С ограничением по токенам
run("Ограничение max_tokens=50", max_tokens=50)

# 6. Со стоп-последовательностью
run(
    "Стоп-последовательность (stop при 'программ')",
    max_tokens=256,
    stop_sequences=["программ"],
)

# 7. top_p (nucleus sampling)
run("top_p=0.5 (только верхние 50% вероятностей)", max_tokens=256, top_p=0.5)

# 8. top_k (только top-k токенов)
run("top_k=10 (только топ-10 токенов)", max_tokens=256, top_k=10)

# 9. Расширенное мышление (adaptive thinking)
print(f"\n{'=' * 60}")
print("  Adaptive Thinking (claude-opus-4-6)")
print(f"{'=' * 60}")
print()

response = client.messages.create(
    model=MODEL,
    max_tokens=2048,
    thinking={"type": "adaptive"},
    messages=[{"role": "user", "content": "Сколько простых чисел меньше 100? Посчитай и объясни."}],
)

for block in response.content:
    if block.type == "thinking":
        print(f"[THINKING]\n{block.thinking[:300]}{'...' if len(block.thinking) > 300 else ''}\n")
    elif block.type == "text":
        print(f"[ОТВЕТ]\n{block.text}")

usage = response.usage
print(f"\n  [input: {usage.input_tokens} tok | output: {usage.output_tokens} tok]")

print("\nГотово!")
