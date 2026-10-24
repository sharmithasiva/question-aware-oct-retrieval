from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS_DIR = ROOT / "OLIVES_Dataset_Labels" / "full_labels"
CLINICAL_XLSX = LABELS_DIR / "Clinical_Data_Images.xlsx"
BIOMARKER_CSV = LABELS_DIR / "Biomarker_Clinical_Data_Images.csv"
OCT_DR_XLSX = LABELS_DIR / "OCT-DR.xlsx"

PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"

SEED = 42
