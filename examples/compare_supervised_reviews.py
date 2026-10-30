"""Compare separately saved full fine-tuning and supervised PEFT reviews."""

import argparse
from pathlib import Path

from amortized_gnn_tta.review_comparison import compare_supervised_reviews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--references", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = compare_supervised_reviews(args.full, args.references, output_dir=args.output_dir)
    for row in report["summary"]:
        if row["shift"] == "all_shifted":
            print(f"{row['method']}: {100*row['before_accuracy']:.2f}% -> {100*row['after_accuracy']:.2f}% "
                  f"({row['gain_pp']:+.2f} points; {row['pairs']} matched development pairs)")


if __name__ == "__main__":
    main()
