"""
CLI entry point for GGUF → .ninf conversion.

Usage:
    python -m tools.cli_convert input.gguf output.ninf [--dtype fp32|q4_0]
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.convert import convert_gguf_to_ninf


def main():
    parser = argparse.ArgumentParser(
        description='Convert GGUF model to NanoInfer .ninf format')
    parser.add_argument('input', help='Input GGUF file path')
    parser.add_argument('output', help='Output .ninf file path')
    parser.add_argument('--dtype', choices=['fp32', 'q4_0'], default='q4_0',
                        help='Target dtype for quantized tensors (default: q4_0)')
    args = parser.parse_args()
    convert_gguf_to_ninf(args.input, args.output, args.dtype)


if __name__ == '__main__':
    main()
