#!/usr/bin/env python3

"""
Split combined dual-unit pleth CSV into two separate pleth CSVs
compatible with pleth_overlay.py.

Input format:
  timestamp,elapsed_s,unit,pleth_unit1,pleth_unit2

When unit=Unit1, pleth_unit1 is the fresh sample.
When unit=Unit2, pleth_unit2 is the fresh sample.

Output: two files with the standard pleth format (timestamp, elapsed_s, pleth)

Usage:
  python split_pleth.py combined.csv
  python split_pleth.py combined.csv --suffix1 R4 --suffix2 L4
  python split_pleth.py combined.csv --overlay  (also runs pleth_overlay.py)

J. Beale  2026-02
"""

import csv
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Split dual-unit pleth CSV')
    parser.add_argument('input_csv', help='Combined pleth CSV file')
    parser.add_argument('--suffix1', type=str, default='unit1',
                        help='Suffix for Unit1 output file (default: unit1)')
    parser.add_argument('--suffix2', type=str, default='unit2',
                        help='Suffix for Unit2 output file (default: unit2)')
    parser.add_argument('--overlay', action='store_true',
                        help='Run pleth_overlay.py after splitting')
    parser.add_argument('--slowest', type=int, default=None,
                        help='Pass --slowest to pleth_overlay.py')
    parser.add_argument('--post', type=float, default=1000,
                        help='Pass --post to pleth_overlay.py (default: 1000)')
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    stem = input_path.stem
    parent = input_path.parent

    out1_path = parent / f'{stem}_{args.suffix1}_pleth.csv'
    out2_path = parent / f'{stem}_{args.suffix2}_pleth.csv'

    with open(args.input_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Separate by unit, using the correct pleth column for each
    n1 = n2 = 0
    with open(out1_path, 'w', newline='') as f1, \
         open(out2_path, 'w', newline='') as f2:

        w1 = csv.writer(f1)
        w2 = csv.writer(f2)
        w1.writerow(['timestamp', 'elapsed_s', 'pleth'])
        w2.writerow(['timestamp', 'elapsed_s', 'pleth'])

        for r in rows:
            if r['unit'] == 'Unit1':
                w1.writerow([r['timestamp'], r['elapsed_s'], r['pleth_unit1']])
                n1 += 1
            elif r['unit'] == 'Unit2':
                w2.writerow([r['timestamp'], r['elapsed_s'], r['pleth_unit2']])
                n2 += 1

    elapsed1 = float(rows[-1]['elapsed_s']) if rows else 0
    print(f"Unit1 ({args.suffix1}): {n1} samples -> {out1_path}")
    print(f"Unit2 ({args.suffix2}): {n2} samples -> {out2_path}")
    print(f"Duration: {elapsed1:.1f}s")

    if n1 > 0 and n2 > 0:
        rate1 = n1 / elapsed1
        rate2 = n2 / elapsed1
        print(f"Sample rates: Unit1={rate1:.1f} Hz, Unit2={rate2:.1f} Hz")

    if args.overlay:
        script_dir = Path(__file__).parent
        overlay_script = script_dir / 'pleth_overlay.py'
        if not overlay_script.exists():
            overlay_script = Path('pleth_overlay.py')

        cmd = [sys.executable, str(overlay_script),
               str(out1_path), str(out2_path),
               '--post', str(args.post)]
        if args.slowest:
            cmd.extend(['--slowest', str(args.slowest)])

        print(f"\nRunning overlay: {' '.join(cmd)}")
        subprocess.run(cmd)


if __name__ == '__main__':
    main()
