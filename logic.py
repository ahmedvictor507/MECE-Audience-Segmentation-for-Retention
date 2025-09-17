import json
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class MECE_Segmenter:
    def __init__(self, df_universe, min_size=500, max_size=20000):
        self.df = df_universe.copy()
        self.min_size = min_size
        self.max_size = max_size
        self.segments = {}  # name -> boolean mask
        self.segment_order = []  # deterministic order for mutual exclusivity

    def build_decision_tree(self):
        # Thresholds (tweakable)
        # AOV tiers
        HIGH_AOV = 3000
        MID_AOV = 1000

        # Engagement tiers
        ENG_HIGH = 0.6
        ENG_MED = 0.3

        # Profitability tiers
        PROF_HIGH = 0.7
        PROF_MED = 0.4

        # We'll construct segments in a deterministic top-down order so they are MECE.
        # The rules are hierarchical: AOV -> Engagement -> Profitability

        # 1) High AOV branch
        mask_high_aov = self.df['avg_order_value'] > HIGH_AOV
        sub = self.df[mask_high_aov]
        self._add_segment('High_AOV', mask_high_aov, parent=None)

        self._add_segment('High_AOV_High_Engage', mask_high_aov & (self.df['engagement_score'] > ENG_HIGH), parent='High_AOV')
        self._add_segment('High_AOV_Med_Engage', mask_high_aov & (self.df['engagement_score'] <= ENG_HIGH) & (self.df['engagement_score'] > ENG_MED), parent='High_AOV')
        self._add_segment('High_AOV_Low_Engage', mask_high_aov & (self.df['engagement_score'] <= ENG_MED), parent='High_AOV')

        # 2) Mid AOV branch
        mask_mid_aov = (self.df['avg_order_value'] > MID_AOV) & (self.df['avg_order_value'] <= HIGH_AOV)
        self._add_segment('Mid_AOV', mask_mid_aov, parent=None)
        self._add_segment('Mid_AOV_High_Engage', mask_mid_aov & (self.df['engagement_score'] > ENG_HIGH), parent='Mid_AOV')
        self._add_segment('Mid_AOV_Med_Engage', mask_mid_aov & (self.df['engagement_score'] <= ENG_HIGH) & (self.df['engagement_score'] > ENG_MED), parent='Mid_AOV')
        self._add_segment('Mid_AOV_Low_Engage', mask_mid_aov & (self.df['engagement_score'] <= ENG_MED), parent='Mid_AOV')

        # 3) Low AOV branch
        mask_low_aov = self.df['avg_order_value'] <= MID_AOV
        self._add_segment('Low_AOV', mask_low_aov, parent=None)
        self._add_segment('Low_AOV_High_Engage', mask_low_aov & (self.df['engagement_score'] > ENG_HIGH), parent='Low_AOV')
        self._add_segment('Low_AOV_Med_Engage', mask_low_aov & (self.df['engagement_score'] <= ENG_HIGH) & (self.df['engagement_score'] > ENG_MED), parent='Low_AOV')
        self._add_segment('Low_AOV_Low_Engage', mask_low_aov & (self.df['engagement_score'] <= ENG_MED), parent='Low_AOV')

        # 4) Special profitability-based segment inside high engagement but high profitability
        self._add_segment('High_Profit_High_Engage', (self.df['profitability_score'] > PROF_HIGH) & (self.df['engagement_score'] > ENG_HIGH), parent=None)

        # 5) ELSE - fallback bucket that catches remaining unassigned users
        # We'll create the ELSE mask as those not captured by any earlier segment
        assigned_mask = pd.Series(False, index=self.df.index)
        for name in self.segment_order:
            assigned_mask = assigned_mask | self.segments[name][0]
        else_mask = ~assigned_mask
        self._add_segment('ELSE', else_mask, parent=None)

        # Now make them mutually exclusive by applying order: first matching segment takes the user
        self._materialize_exclusive_segments()

    def _add_segment(self, name, mask, parent=None):
        # store tuple (mask, parent)
        if name in self.segments:
            raise ValueError(f"Duplicate segment name: {name}")
        self.segments[name] = (mask.fillna(False), parent)
        self.segment_order.append(name)

    def _materialize_exclusive_segments(self):
        # Create exclusive masks by iterating segment_order: assign user to first matching segment
        n = len(self.df)
        assigned = pd.Series(False, index=self.df.index)
        exclusive = {}
        for name in self.segment_order:
            mask, parent = self.segments[name]
            # only assign those not already assigned
            this_mask = mask & (~assigned)
            exclusive[name] = (this_mask, parent)
            assigned = assigned | this_mask
        # any unassigned (should be none because of ELSE) remain unassigned
        self.segments = exclusive

    def enforce_size_constraints(self):
        # Fold small segments (size < min_size) into parent ELSE bucket.
        # We'll iterate bottom-up: leaf segments first (heuristic: names with underscores are children)
        sizes = {name: int(mask.sum()) for name, (mask, parent) in self.segments.items()}

        # Identify order for fold: segments with parent not None fold into parent
        # We'll loop until no changes
        changed = True
        while changed:
            changed = False
            for name, (mask, parent) in list(self.segments.items()):
                size = int(mask.sum())
                if name == 'ELSE':
                    continue
                if size > 0 and size < self.min_size:
                    # fold into parent if parent exists, else into ELSE
                    target = parent if parent is not None else 'ELSE'
                    if target not in self.segments:
                        # create target as empty
                        self.segments[target] = (pd.Series(False, index=self.df.index), None)
                    # move users
                    src_mask = mask
                    tgt_mask, tgt_parent = self.segments[target]
                    new_tgt_mask = tgt_mask | src_mask
                    self.segments[target] = (new_tgt_mask, tgt_parent)
                    # clear source
                    self.segments[name] = (pd.Series(False, index=self.df.index), parent)
                    changed = True
            if changed:
                # re-materialize exclusivity to ensure no overlaps
                self._materialize_exclusive_segments()
        # Optionally, we could also cap large segments (not required by problem), but we keep as-is

    def export_segment_assignments(self):
        # produce df with segment name for each user
        seg_series = pd.Series(index=self.df.index, dtype=object)
        for name, (mask, parent) in self.segments.items():
            seg_series[mask] = name
        out = self.df.copy()
        out['segment'] = seg_series.fillna('UNASSIGNED')
        return out

    def compute_segment_scores(self):
        # For each segment compute metrics
        df_assigned = self.df.copy()
        assigned_df = self.export_segment_assignments()

        # Prepare per-user conversion_potential = engagement * recency_score
        now = datetime.utcnow()
        days_since = (now - pd.to_datetime(assigned_df['cart_abandoned_date'])).dt.days
        recency_score = np.clip(1 - (days_since / 7.0), 0, 1)  # 7 day window
        assigned_df['recency_score'] = recency_score
        assigned_df['conversion_potential_user'] = assigned_df['engagement_score'] * assigned_df['recency_score']

        # lift_vs_control simulated: base on engagement & profitability + noise
        rng = np.random.default_rng(123)
        assigned_df['lift_vs_control_user'] = (0.2 * assigned_df['engagement_score'] + 0.6 * assigned_df['profitability_score']) + rng.normal(0, 0.05, size=len(assigned_df))
        assigned_df['lift_vs_control_user'] = assigned_df['lift_vs_control_user'].clip(0, 2)

        # Strategic fit: placeholder combining avg_order_value (log scaled) and profitability
        assigned_df['log_aov'] = np.log1p(assigned_df['avg_order_value'])
        assigned_df['strategic_fit_user'] = (assigned_df['log_aov'] / assigned_df['log_aov'].max()) * 0.6 + assigned_df['profitability_score'] * 0.4

        # Group by segment
        grouped = assigned_df.groupby('segment').agg(
            Size=('user_id', 'count'),
            Conv_Pot=('conversion_potential_user', 'mean'),
            Lift_vs_Control=('lift_vs_control_user', 'mean'),
            Profitability=('profitability_score', 'mean'),
            Strategic_Fit=('strategic_fit_user', 'mean')
        ).reset_index()

        # Size normalized
        min_s = grouped['Size'].min()
        max_s = grouped['Size'].max()
        if max_s == min_s:
            grouped['Size_norm'] = 1.0
        else:
            grouped['Size_norm'] = (grouped['Size'] - min_s) / (max_s - min_s)

        # Overall score: weighted sum (weights can be tuned)
        weights = {
            'Conv_Pot': 0.35,
            'Lift_vs_Control': 0.25,
            'Size_norm': 0.15,
            'Profitability': 0.15,
            'Strategic_Fit': 0.10
        }

        grouped['Overall_Score'] = (
            weights['Conv_Pot'] * grouped['Conv_Pot'] +
            weights['Lift_vs_Control'] * grouped['Lift_vs_Control'] +
            weights['Size_norm'] * grouped['Size_norm'] +
            weights['Profitability'] * grouped['Profitability'] +
            weights['Strategic_Fit'] * grouped['Strategic_Fit']
        )

        # Valid flag: if segment has >0 users and Size within min/max, else False (merged/unusable)
        grouped['Valid'] = grouped['Size'].apply(lambda s: (s >= self.min_size) and (s <= self.max_size))

        # If merged into ELSE, mark Valid True for ELSE if its size meets constraints
        # We'll also attach 'Rules Applied' by reconstructing simple rules from segment name
        def rules_from_name(name):
            if name == 'ELSE':
                return 'ELSE - fallback'
            parts = name.split('_')
            # basic decode
            if parts[0] in ('High', 'Mid', 'Low') and parts[1] == 'AOV':
                rule = f'AOV_{parts[0]}'
                if len(parts) > 2:
                    rule += f' & Engagement_{parts[2]}'
                return rule
            if name == 'High_Profit_High_Engage':
                return 'Profitability > 0.7 & Engagement > 0.6'
            return name

        grouped['Rules_Applied'] = grouped['segment'].apply(rules_from_name)

        # Sort by Overall_Score desc
        grouped = grouped.sort_values('Overall_Score', ascending=False).reset_index(drop=True)

        return grouped