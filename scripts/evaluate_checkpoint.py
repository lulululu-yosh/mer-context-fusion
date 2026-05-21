from __future__ import annotations

import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    raise NotImplementedError(
        "Evaluation checkpoint loading will be implemented after the first text baseline is trained."
    )


if __name__ == "__main__":
    main()
