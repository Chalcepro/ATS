"""Export ATS runtime logs into Virus ATS profile corpus."""

import argparse
from pathlib import Path

import config_rl
from ats_virus_adapter import VirusAdapter


def export_file(input_path):
    adapter = VirusAdapter()
    lines = []
    with open(input_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or "|" not in line:
                continue
            inp, out = line.split("|", 1)
            lines.append(f"### INPUT: {inp.strip()}")
            lines.append(f"### OUTPUT: {out.strip()}")
            lines.append("")
    if not lines:
        print("No exportable lines found.")
        return 1
    out_path = adapter.export_runtime_corpus(lines)
    print(f"Exported {len(lines)} lines to {out_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Export ATS logs to Virus ATS corpus")
    parser.add_argument("input", help="Log file with INPUT|OUTPUT lines")
    args = parser.parse_args()
    raise SystemExit(export_file(args.input))


if __name__ == "__main__":
    main()
