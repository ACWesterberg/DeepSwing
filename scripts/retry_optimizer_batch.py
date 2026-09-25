"""Explicitly retry failed optimizer Batch requests (paid operation)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--authorized-by", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--max-requests", type=int, help="New cumulative request ceiling, including prior attempts")
    parser.add_argument("--max-reserved-output-tokens", type=int, help="New cumulative token ceiling")
    parser.add_argument("--submit", action="store_true", help="Required acknowledgement that this sends paid requests")
    args = parser.parse_args()
    if not args.submit:
        parser.error("Use --submit to authorize a paid retry")
    from openai import OpenAI
    from config.settings import settings
    from src.agent.batch_retry import submit_failed_retry
    print(submit_failed_retry(args.run_dir, authorized_by=args.authorized_by, reason=args.reason,
        output_tokens=args.output_tokens, max_requests=args.max_requests,
        max_reserved_output_tokens=args.max_reserved_output_tokens,
        client=OpenAI(api_key=settings.openai_api_key, max_retries=0)))


if __name__ == "__main__":
    main()
