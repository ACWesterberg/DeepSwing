from __future__ import annotations

import argparse
import os
import sys


def ping_openai(model: str) -> tuple[bool, str]:
    import openai

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return False, "OPENAI_API_KEY unset"
    try:
        client = openai.OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=16,
            messages=[{"role": "user", "content": "ping"}],
        )
        return True, (resp.choices[0].message.content or "").strip() or "(empty reply)"
    except Exception as exc:
        return False, str(exc)


def ping_anthropic(model: str) -> tuple[bool, str]:
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return False, "ANTHROPIC_API_KEY unset"
    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            max_tokens=8,
            messages=[{"role": "user", "content": "ping"}],
        )
        text = "".join(getattr(b, "text", "") for b in resp.content).strip()
        return True, text or "(empty reply)"
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ping a model ID to check it is valid/reachable.")
    parser.add_argument("model", nargs="?", default="gpt-5.6-sol", help="Model ID to test")
    parser.add_argument(
        "--provider", choices=["openai", "anthropic"], default="openai", help="API provider"
    )
    args = parser.parse_args()

    ping = ping_openai if args.provider == "openai" else ping_anthropic
    ok, detail = ping(args.model)
    status = "OK" if ok else "FAIL"
    print(f"Preflight {status}: {args.provider}/{args.model} — {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
