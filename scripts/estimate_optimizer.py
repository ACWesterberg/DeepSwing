"""Estimate a frozen optimizer corpus locally, without requests or state changes."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="Saved evaluation corpus.json containing datasets")
    parser.add_argument("--track", choices=("gpt", "claude"), required=True)
    parser.add_argument("--incumbent", type=Path, help="Saved program; omit for baseline")
    parser.add_argument("--candidate", type=Path, help="Saved candidate; omit to estimate its size allowance")
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()
    # This standalone command must never fetch tokenizers or call a provider.
    def offline(*args, **kwargs):
        raise RuntimeError("Offline estimate: tokenizer data must already be installed/cached")
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import dspy
    import tiktoken
    from src.agent.decision import TradeDecision
    from src.agent.search_estimate import estimate_requests, search_requests
    incumbent = dspy.Predict(TradeDecision)
    if args.incumbent:
        incumbent.load(str(args.incumbent))
    candidate = None
    if args.candidate:
        candidate = dspy.Predict(TradeDecision)
        candidate.load(str(args.candidate))
    requests = search_requests(json.loads(args.corpus.read_text())["datasets"], track=args.track,
                               incumbent=incumbent, candidate=candidate, batch=args.batch)
    print(json.dumps(estimate_requests(requests, encoder_for_model=tiktoken.encoding_for_model), indent=2))


if __name__ == "__main__":
    main()
