import os
import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <modelnet_root> <output_root> <convex_hull_binary>")
        sys.exit(1)

    modelnet_root = Path(sys.argv[1])
    output_root   = Path(sys.argv[2])
    binary        = sys.argv[3]

    total, success, failed = 0, 0, 0

    for category_dir in sorted(modelnet_root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        for split in ["train", "test"]:
            split_dir = category_dir / split
            if not split_dir.is_dir():
                continue

            out_split_dir = output_root / category / split
            out_split_dir.mkdir(parents=True, exist_ok=True)

            for input_off in sorted(split_dir.glob("*.off")):
                output_off = out_split_dir / input_off.name
                total += 1

                result = subprocess.run(
                    [binary, str(input_off), str(output_off)],
                    capture_output=True, text=True
                )

                if result.returncode == 0:
                    success += 1
                    print(f"[OK]     {input_off}")
                else:
                    failed += 1
                    print(f"[FAILED] {input_off}")
                    print(f"         {result.stderr.strip()}")

    print("=" * 50)
    print(f"Done. Total: {total}, Success: {success}, Failed: {failed}")
    print(f"Output at: {output_root}")

if __name__ == "__main__":
    main()