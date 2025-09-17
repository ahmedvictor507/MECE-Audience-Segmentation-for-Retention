import json
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def generate_mock_data(n=50000, seed=42):
    np.random.seed(seed)
    now = datetime.utcnow()

    # Randomly assign cart_abandoned_date in last 30 days; we'll filter last 7 days
    days_ago = np.random.randint(0, 30, size=n)
    cart_abandoned_date = [now - timedelta(days=int(d)) for d in days_ago]

    # last_order_date sometimes null
    last_order_gap = np.random.exponential(scale=30, size=n).astype(int)
    last_order_date = [now - timedelta(days=int(g)) if np.random.rand() < 0.8 else pd.NaT for g in last_order_gap]

    avg_order_value = np.round(np.random.lognormal(mean=7.5, sigma=0.9, size=n))  # skewed AOV
    sessions_last_30d = np.random.poisson(lam=3, size=n)
    num_cart_items = np.random.randint(1, 10, size=n)

    # engagement_score in [0,1]
    engagement_score = np.clip(np.random.beta(a=2, b=2, size=n), 0, 1)
    # profitability_score in [0,1]
    profitability_score = np.clip(np.random.beta(a=2.2, b=1.8, size=n), 0, 1)

    df = pd.DataFrame({
        'user_id': [f'user_{i:06d}' for i in range(n)],
        'cart_abandoned_date': cart_abandoned_date,
        'last_order_date': last_order_date,
        'avg_order_value': avg_order_value,
        'sessions_last_30d': sessions_last_30d,
        'num_cart_items': num_cart_items,
        'engagement_score': engagement_score,
        'profitability_score': profitability_score
    })
    return df