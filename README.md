# MECE Cart Abandoner Segmentation

## 📌 Overview
This project implements a **MECE (Mutually Exclusive, Collectively Exhaustive)** segmentation system for a **Cart Abandoner Retention Strategy**.  
The goal is to create **clear, non-overlapping audience segments** of users who abandoned carts in the last 7 days, compute **audience scores**, and output results that marketers can use to run targeted campaigns.

---

## 🚀 Features
- Generates **mock dataset** (or reads input CSV if provided).
- Filters **universe**: users who abandoned carts in the last 7 days.
- Builds **decision-tree inspired MECE segmentation**:
  - Low AOV, Mid AOV, High AOV buckets.
  - Mutually Exclusive (no overlaps).
  - Collectively Exhaustive (every user assigned).
- Enforces **min/max segment sizes** (merges small segments into ELSE bucket).
- Computes **audience scores**:
  - Conversion Potential  
  - Lift vs Control (simulated)  
  - Profitability  
  - Strategic Fit  
  - Size (normalized)  
  - Overall Score (aggregate)
- Exports results to **CSV + JSON**.

---

## 📂 Project Structure
├── main.py # Entry point script
├── logic.py # MECE_Segmenter class (segmentation + scoring logic)
├── utilities.py # Mock dataset generator
├── out/ # Generated outputs
│ ├── segment_assignments.csv
│ ├── segment_scores.csv
│ └── segment_scores.json
└── README.md # Documentation

yaml
Copy code

---

## ▶️ How to Run

### 1. Install requirements
`pip install pandas numpy`

### 2. Run with mock data
`python main.py --mock --mock_size 10000`

### 3. Run with real dataset
Provide a CSV file containing user data (with required fields such as `user_id`, `cart_abandoned_date`, `avg_order_value`, etc.) and specify an output directory:
`python main.py --input path/to/your_data.csv --output_dir ./out`

This will:
- Load your dataset instead of generating mock data.
- Apply segmentation rules.
- Save outputs (`segment_assignments.csv`, `segment_scores.csv`, `segment_scores.json`) in the specified `--output_dir`.

---

## 📊 Example Output

Sample output of `segment_scores.csv`:

| Segment  | Size | Conv_Pot | Lift_vs_Control | Profitability | Strategic_Fit | Size_norm | Overall_Score | Valid | Rules_Applied |
|----------|------|----------|-----------------|---------------|---------------|-----------|---------------|-------|---------------|
| Mid_AOV  | 985  | 0.28     | 0.43            | 0.55          | 0.63          | 1.00      | 0.50          | True  | AOV_Mid       |
| High_AOV | 664  | 0.29     | 0.43            | 0.56          | 0.69          | 0.16      | 0.39          | True  | AOV_High      |
| Low_AOV  | 603  | 0.30     | 0.43            | 0.54          | 0.57          | 0.00      | 0.35          | True  | AOV_Low       |

---

## 🎯 Why Segmentation?
- **MECE Principle**: ensures no overlap and full coverage.  
- **Marketing Use Case**: marketers can tailor campaigns by segment.  
- **Scoring**: provides a data-driven way to prioritize audiences.  

Without segmentation, all cart abandoners would be lumped together, making it hard to design effective retention strategies.

---

## 📝 Deliverables
This project provides:
- **Code / Workflow**: Python implementation of segmentation pipeline.  
- **Outputs**: CSV and JSON files with assignments and scores.  
- **Video Walkthrough (to be recorded)**: explain logic, run demo, personal intro.  

---

## 📈 Limitations & Future Improvements
- Thresholds for AOV and engagement are static → can be optimized with clustering.  
- Lift vs Control values are simulated → replace with actual experiment results.  
- Segmentation can be extended with RFM analysis or ML-based audience modeling.  

---

## 👤 Author
Developed as part of a **Data Science Assignment** on retention strategy and MECE segmentation.
