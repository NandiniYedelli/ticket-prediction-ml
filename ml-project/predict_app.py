"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         WaitSure — Interactive Prediction App (predict_app.py)              ║
║         Run: python predict_app.py                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Steps:
  1. Auto-merges 3 CSVs (waitsure_200_merged + Railway Ticket Confirmation
     + waiting list) into one analysis-ready DataFrame.
  2. Loads the pre-trained model from models/ folder.
  3. Shows dropdown menus so you pick Train No, Date, Class, etc.
  4. Prints the WaitSure prediction banner.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

# ──────────────────────────────────────────────────────────────────────────────
# PATHS  (edit if your folder layout is different)
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

PATH_MAIN  = os.path.join(DATA_DIR, "waitsure_200_merged.csv")
PATH_CONF  = os.path.join(DATA_DIR, "Railway Ticket Confirmation.csv")
PATH_WL    = os.path.join(DATA_DIR, "waiting list.csv")

MODEL_FILE = os.path.join(MODEL_DIR, "waitsure_best_model.pkl")
SCALER_FILE= os.path.join(MODEL_DIR, "scaler.pkl")
ENC_FILE   = os.path.join(MODEL_DIR, "encoders.pkl")
FEAT_FILE  = os.path.join(MODEL_DIR, "feat_names.json")

CAT_COLS   = ["train_type", "travel_class", "quota",
              "season", "source_station", "destination"]

PALETTE = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — MERGE THE 3 CSVs
# ──────────────────────────────────────────────────────────────────────────────
def merge_datasets() -> pd.DataFrame:
    """
    Load and merge all 3 source CSVs into a single DataFrame.
    waitsure_200_merged  →  high-quality labelled core
    Railway Ticket Confirmation → 30 k IRCTC records
    waiting list         →  status timeline features
    """
    print(f"\n{PALETTE['cyan']}📂  Merging datasets …{PALETTE['reset']}")

    # ① Core dataset (already merged & labelled)
    df_main = pd.read_csv(PATH_MAIN)
    print(f"  ✓ waitsure_200_merged        → {df_main.shape}")

    # ② Large IRCTC confirmation dataset
    df_conf = pd.read_csv(PATH_CONF)
    df_conf.columns = [c.strip() for c in df_conf.columns]
    print(f"  ✓ Railway Ticket Confirmation → {df_conf.shape}")

    rename_map = {
        "Train Number":           "train_no",
        "Class of Travel":        "travel_class",
        "Quota":                  "quota",
        "Source Station":         "source_station",
        "Destination Station":    "destination",
        "Date of Journey":        "journey_date",
        "Booking Date":           "booking_date",
        "Waitlist Position":      "waitlist_raw",
        "Travel Distance":        "travel_distance",
        "Seat Availability":      "total_seats",
        "Holiday or Peak Season": "season",
        "Train Type":             "train_type",
        "Confirmation Status":    "confirmed_str",
    }
    df_conf = df_conf.rename(columns=rename_map)
    df_conf["confirmed"] = (
        df_conf["confirmed_str"].str.strip() == "Confirmed"
    ).astype(int)
    df_conf["waitlist_number"] = (
        pd.to_numeric(
            df_conf["waitlist_raw"].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce",
        ).fillna(0).astype(int)
    )
    df_conf["journey_date"]  = pd.to_datetime(df_conf["journey_date"],  errors="coerce")
    df_conf["booking_date"]  = pd.to_datetime(df_conf["booking_date"],  errors="coerce")
    df_conf["days_before_travel"] = (
        df_conf["journey_date"] - df_conf["booking_date"]
    ).dt.days
    df_conf["historical_confirm_rate"] = np.nan
    df_conf["cancellation_trend"]      = np.nan

    # ③ Waiting-list status timeline dataset
    df_wl = pd.read_csv(PATH_WL)
    print(f"  ✓ waiting list               → {df_wl.shape}")

    # ④ Combine on common columns
    COMMON = [
        "train_no", "train_type", "travel_class", "quota",
        "source_station", "destination", "season",
        "days_before_travel", "waitlist_number", "total_seats",
        "historical_confirm_rate", "cancellation_trend", "confirmed",
    ]
    df_m = df_main[[c for c in COMMON if c in df_main.columns]].copy()
    df_c = df_conf[[c for c in COMMON if c in df_conf.columns]].copy()
    df   = pd.concat([df_m, df_c], ignore_index=True).dropna(subset=["confirmed"])

    print(f"\n  {PALETTE['green']}► Final merged shape : {df.shape}{PALETTE['reset']}")
    print(f"  ► Confirmed: {(df['confirmed']==1).sum()}  |  Not confirmed: {(df['confirmed']==0).sum()}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — LOAD PRE-TRAINED MODEL
# ──────────────────────────────────────────────────────────────────────────────
def load_model():
    print(f"\n{PALETTE['cyan']}🤖  Loading saved model …{PALETTE['reset']}")
    model     = joblib.load(MODEL_FILE)
    scaler    = joblib.load(SCALER_FILE)
    encoders  = joblib.load(ENC_FILE)
    with open(FEAT_FILE) as f:
        feat_names = json.load(f)
    print(f"  ✓ Model    : {type(model).__name__}")
    print(f"  ✓ Features : {len(feat_names)} features loaded")
    return model, scaler, encoders, feat_names


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — DROPDOWN INPUT HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def pick(prompt: str, options: list, default_idx: int = 0):
    """Show numbered dropdown, return chosen value."""
    print(f"\n{PALETTE['bold']}{prompt}{PALETTE['reset']}")
    for i, opt in enumerate(options, 1):
        marker = f"  {PALETTE['cyan']}[{i}]{PALETTE['reset']}"
        print(f"{marker} {opt}")
    while True:
        raw = input(f"  Enter number (default={default_idx+1}): ").strip()
        if raw == "":
            return options[default_idx]
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("  ⚠  Invalid choice, try again.")


def pick_number(prompt: str, default: float, min_val: float = 0, max_val: float = 9999):
    """Ask for a numeric value with a default."""
    while True:
        raw = input(f"  {PALETTE['bold']}{prompt}{PALETTE['reset']} (default={default}): ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if min_val <= val <= max_val:
                return val
        except ValueError:
            pass
        print(f"  ⚠  Please enter a number between {min_val} and {max_val}.")


def pick_date(prompt: str, default: str = "2025-01-01"):
    """Ask for a date in YYYY-MM-DD format."""
    while True:
        raw = input(f"  {PALETTE['bold']}{prompt}{PALETTE['reset']} (default={default}, format YYYY-MM-DD): ").strip()
        if raw == "":
            return default
        try:
            pd.to_datetime(raw)
            return raw
        except Exception:
            print("  ⚠  Invalid date. Use YYYY-MM-DD format (e.g. 2025-06-15).")


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — COLLECT INPUTS VIA DROPDOWNS
# ──────────────────────────────────────────────────────────────────────────────
def collect_inputs(df_merged: pd.DataFrame) -> dict:
    """
    Show dropdown menus so user can select all ticket details.
    Values are pulled dynamically from the merged dataset.
    """
    print(f"\n{'='*65}")
    print(f"  {PALETTE['bold']}🎫  Enter your IRCTC Ticket Details{PALETTE['reset']}")
    print(f"{'='*65}")

    # Dynamic options from merged data (sorted for readability)
    train_nos    = sorted(df_merged["train_no"].dropna().unique().tolist())
    train_nos    = [int(x) for x in train_nos]
    train_types  = sorted(df_merged["train_type"].dropna().unique().tolist())
    stations     = sorted(set(
        df_merged["source_station"].dropna().unique().tolist() +
        df_merged["destination"].dropna().unique().tolist()
    ))
    classes      = sorted(df_merged["travel_class"].dropna().unique().tolist())
    quotas       = sorted(df_merged["quota"].dropna().unique().tolist())
    seasons      = sorted(df_merged["season"].dropna().unique().tolist())

    train_no      = pick("🚂  Select Train Number",       train_nos,   default_idx=4)
    train_type    = pick("🚂  Select Train Type",         train_types, default_idx=0)
    source        = pick("📍  Select Source Station",     stations,    default_idx=4)
    destination   = pick("📍  Select Destination Station",stations,    default_idx=5)

    journey_date  = pick_date("📅  Journey Date")
    booking_date  = pick_date("📅  Booking Date (date you booked the ticket)", default="2025-01-01")

    # Auto-compute days_before_travel from dates
    jd  = pd.to_datetime(journey_date)
    bd  = pd.to_datetime(booking_date)
    dbt = max(0, (jd - bd).days)
    print(f"  ✓ Days before travel computed: {dbt} days")

    travel_class  = pick("🪑  Select Travel Class",       classes,     default_idx=0)
    quota         = pick("🎟️   Select Quota",              quotas,      default_idx=0)
    season        = pick("🗓️   Select Season",             seasons,     default_idx=0)

    waitlist_num  = int(pick_number("📋  Enter Waitlist Number (e.g. 14)",          default=14,   min_val=1,  max_val=500))
    total_seats   = int(pick_number("💺  Enter Total Seats in Coach (e.g. 72)",     default=72,   min_val=10, max_val=500))
    hist_rate     = pick_number("📈  Historical Confirm Rate (0.0–1.0, e.g. 0.72)", default=0.72, min_val=0,  max_val=1)
    cancel_trend  = pick_number("📉  Cancellation Trend (avg cancellations, e.g. 10)", default=10, min_val=0, max_val=200)

    return {
        "train_no":                train_no,
        "train_type":              train_type,
        "source_station":          source,
        "destination":             destination,
        "journey_date":            journey_date,
        "booking_date":            booking_date,
        "days_before_travel":      dbt,
        "travel_class":            travel_class,
        "quota":                   quota,
        "season":                  season,
        "waitlist_number":         waitlist_num,
        "total_seats":             total_seats,
        "historical_confirm_rate": hist_rate,
        "cancellation_trend":      cancel_trend,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — PREDICT
# ──────────────────────────────────────────────────────────────────────────────
def predict_ticket(ticket: dict, model, scaler, encoders, feat_names) -> dict:
    """Run the WaitSure prediction for one ticket."""
    df = pd.DataFrame([ticket])

    # Encode categoricals
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip().str.upper()
        if col in encoders:
            le    = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])

    # Derived features (same as training pipeline)
    df["wl_seat_ratio"]    = (df["waitlist_number"] / df["total_seats"].replace(0, np.nan)).fillna(0)
    df["booking_urgency"]  = pd.cut(
        df["days_before_travel"].fillna(0),
        bins=[-1, 3, 7, 14, 30, 365],
        labels=[4, 3, 2, 1, 0],
    ).astype(float)
    df["confirm_signal"]   = (
        df["historical_confirm_rate"] /
        (df["waitlist_number"].replace(0, 0.5) ** 0.5)
    ).fillna(0)
    df["cancel_pressure"]  = (
        df["cancellation_trend"] / df["total_seats"].replace(0, np.nan)
    ).fillna(0)
    df["is_premium_class"] = df["travel_class"].isin([0, 1]).astype(int)

    # Align to training features
    for col in feat_names:
        if col not in df.columns:
            df[col] = 0.0
    X = df[feat_names].fillna(0).values.astype(float)
    X = scaler.transform(X)

    prob  = model.predict_proba(X)[0, 1]

    # Verdict
    if prob >= 0.75:
        verdict = f"{PALETTE['green']}✅  LIKELY CONFIRMED{PALETTE['reset']}"
        advice  = "Great chance of getting a seat! Keep monitoring chart closer to departure."
    elif prob >= 0.50:
        verdict = f"{PALETTE['yellow']}⚠️   POSSIBLE CONFIRMATION{PALETTE['reset']}"
        advice  = "Borderline. Consider booking an alternate train or RAC seat as backup."
    else:
        verdict = f"{PALETTE['red']}❌  UNLIKELY TO CONFIRM{PALETTE['reset']}"
        advice  = "Recommend booking an alternative. Try Tatkal quota or a nearby date."

    bar_len = 30
    filled  = int(prob * bar_len)
    prog    = "█" * filled + "░" * (bar_len - filled)

    print(f"\n{'='*65}")
    print(f"  {PALETTE['bold']}🚂  WaitSure  v1.0 — Prediction Result{PALETTE['reset']}")
    print(f"{'='*65}")
    print(f"  Train      : {ticket['train_no']}  ({ticket['train_type']})")
    print(f"  Route      : {ticket['source_station']} → {ticket['destination']}")
    print(f"  Class      : {ticket['travel_class']}   |  Quota : {ticket['quota']}")
    print(f"  Journey    : {ticket['journey_date']}   |  Season: {ticket['season']}")
    print(f"  WL Pos.    : WL/{ticket['waitlist_number']}  |  Seats : {ticket['total_seats']}")
    print(f"  Days ahead : {ticket['days_before_travel']}")
    print(f"  {'─'*58}")
    print(f"  Probability : {prob:.1%}  [{prog}]")
    print(f"  Verdict     : {verdict}")
    print(f"  {'─'*58}")
    print(f"  💡 {advice}")
    print(f"{'='*65}\n")

    return {"probability": prob, "verdict": verdict, "advice": advice}


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{PALETTE['bold']}{'='*65}")
    print("   🚂  WaitSure — IRCTC Waitlist Confirmation Predictor")
    print(f"{'='*65}{PALETTE['reset']}")

    # 1. Merge datasets
    df_merged = merge_datasets()

    # 2. Load saved model
    model, scaler, encoders, feat_names = load_model()

    # 3. Interactive loop
    while True:
        ticket = collect_inputs(df_merged)
        predict_ticket(ticket, model, scaler, encoders, feat_names)

        again = input("  🔁  Predict another ticket? (y/n, default=n): ").strip().lower()
        if again != "y":
            print(f"\n  {PALETTE['green']}✅  Thank you for using WaitSure! Safe travels 🚂{PALETTE['reset']}\n")
            break


if __name__ == "__main__":
    main()