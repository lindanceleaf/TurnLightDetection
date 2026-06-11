from __future__ import annotations

import sys


def main() -> None:
    try:
        import torch
    except ImportError:
        print("torch is not installed in this Python environment.")
        print(f"python: {sys.executable}")
        raise SystemExit(1)

    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    else:
        print("This torch build cannot use CUDA. Reinstall torch with a CUDA wheel.")


if __name__ == "__main__":
    main()
