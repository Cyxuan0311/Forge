"""
CLI entry point for generating synthetic test models.

Usage:
    python -m tools.cli_create_test --output model.ninf [options]
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.create_test_model import create_test_model


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic .ninf test model")
    parser.add_argument("--output", default="test_model.ninf", help="Output .ninf file path")
    parser.add_argument("--vocab-size", type=int, default=3200)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--intermediate-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    args = parser.parse_args()

    create_test_model(
        args.output,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )


if __name__ == "__main__":
    main()
