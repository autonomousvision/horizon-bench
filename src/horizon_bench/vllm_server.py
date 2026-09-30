"""
vLLM OpenAI-compatible API server.

Usage:
    python -m horizon_bench.vllm_server                          # defaults
    python -m horizon_bench.vllm_server --model meta-llama/Llama-3.1-8B-Instruct --port 8000

The server exposes an OpenAI-compatible API at http://0.0.0.0:{port}/v1
so existing OpenAI SDK clients can connect by setting:
    base_url = "http://localhost:8000/v1"
"""

import os

import uvloop
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse_args():
    parser = FlexibleArgumentParser(description="vLLM OpenAI-compatible API server")
    parser = make_arg_parser(parser)
    # Set sensible defaults (override on CLI as needed)
    parser.set_defaults(
        model="Qwen/Qwen2.5-7B-Instruct",
        host="0.0.0.0",
        port=8001,
        dtype="auto",
        quantization="fp8",
        max_model_len=None,  # auto-detect from model config
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        tensor_parallel_size=2,
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",  # Qwen2.5 uses the Hermes tool-call format
        api_key=os.environ.get("VLLM_API_KEY"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Print key settings so the user can verify
    print(f"Starting vLLM server:")
    print(f"  Model           : {args.model}")
    print(f"  Host            : {args.host}")
    print(f"  Port            : {args.port}")
    print(f"  Tensor parallel : {args.tensor_parallel_size}")
    print(f"  GPU mem util    : {args.gpu_memory_utilization}")
    print(f"  Auth token      : {'set' if args.api_key else 'NONE (open access)'}")
    print(f"  Endpoint        : http://{args.host}:{args.port}/v1")
    print()

    uvloop.run(run_server(args))

# python horizon_bench/vllm_server.py --max-model-len 4096 --gpu-memory-utilization 0.80


if __name__ == "__main__":
    main()
