"""
MECE Cart Abandoner Segmentation Script
======================================

What this script does
- Generates a mock dataset (or reads input CSV if provided)
- Filters universe: users who abandoned carts in the last 7 days
- Builds a decision-tree inspired MECE segmentation (mutually exclusive, collectively exhaustive)
- Enforces min/max segment size (folds small segments into parent ELSE)
- Computes per-segment scores: conversion potential, lift_vs_control (simulated), size(normalized), profitability, strategic_fit
- Aggregates into overall score and exports CSV/JSON

How to run
- As a script: python MECE_Cart_Abandoner_Segmentation.py --input data.csv --output_dir ./out
- To use mock data: python MECE_Cart_Abandoner_Segmentation.py --mock

Requirements
- pandas, numpy

"""

import argparse
import json
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from logic import MECE_Segmenter
from utilities import generate_mock_data

def main(args):
    if args.mock:
        print('Generating mock data...')
        df = generate_mock_data(n=args.mock_size, seed=42)
    else:
        if args.input is None:
            raise ValueError('Either provide --input path or use --mock')
        df = pd.read_csv(args.input, parse_dates=['cart_abandoned_date', 'last_order_date'])

    # Universe: cart abandoned in last 7 days
    now = datetime.utcnow()
    df['cart_abandoned_date'] = pd.to_datetime(df['cart_abandoned_date'])
    universe_cutoff = now - timedelta(days=7)
    df_universe = df[df['cart_abandoned_date'] >= universe_cutoff].reset_index(drop=True)
    print(f'Universe size (abandoned in last 7 days): {len(df_universe)}')

    seg = MECE_Segmenter(df_universe, min_size=args.min_size, max_size=args.max_size)
    seg.build_decision_tree()
    seg.enforce_size_constraints()
    assignments = seg.export_segment_assignments()
    scores = seg.compute_segment_scores()

    # save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    assignments.to_csv(os.path.join(args.output_dir, 'segment_assignments.csv'), index=False)
    scores.to_csv(os.path.join(args.output_dir, 'segment_scores.csv'), index=False)
    with open(os.path.join(args.output_dir, 'segment_scores.json'), 'w') as f:
        f.write(scores.to_json(orient='records', date_format='iso'))

    print('Saved outputs to', args.output_dir)
    print(scores.head(50).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to input CSV. If omitted use --mock')
    parser.add_argument('--output_dir', type=str, default='./out', help='Output directory')
    parser.add_argument('--mock', action='store_true', help='Use mock data')
    parser.add_argument('--mock_size', type=int, default=50000, help='Number of mock users to generate')
    parser.add_argument('--min_size', type=int, default=500, help='Minimum allowed segment size')
    parser.add_argument('--max_size', type=int, default=20000, help='Maximum allowed segment size')
    args = parser.parse_args()
    main(args)
