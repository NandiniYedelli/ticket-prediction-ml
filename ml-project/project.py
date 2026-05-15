"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          WaitSure — IRCTC Waitlist Confirmation Predictor                    ║
║          Version 1.0  |  Full ML Pipeline                                    ║
║                                                                              ║
║  Algorithms used:                                                            ║
║    ► Supervised   : Logistic Regression, Decision Tree, Random Forest        ║
║                     XGBoost, LightGBM, Stacking & Voting Ensembles          ║
║    ► Unsupervised : K-Means Clustering, PCA                                  ║
║    ► Evaluation   : MAE, RMSE, R², AUC-ROC, F1, CV-AUC                      ║
║    ► Explainability: SHAP (SHapley Additive exPlanations)                    ║
║    ► Resampling   : SMOTE (handle class imbalance)                           ║
║                                                                              ║
║  Datasets:                                                                   ║
║    1. waitsure_200_merged.csv            (core labelled data)                ║
║    2. Railway Ticket Confirmation.csv    (30,000 IRCTC records)              ║
║    3. Railway Ticket WaitingList Data.csv (status timeline)                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE
─────
# Full pipeline (train + evaluate + visualise + predict):
    python waitsure_complete_project.py

# Predict a single ticket (after training):
    from waitsure_complete_project import predict_ticket
    predict_ticket(train_no=12301, waitlist_number=14, ...)
"""

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 0 — IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os, json
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing & Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.impute import SimpleImputer

# Supervised Learning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier

# Unsupervised Learning
from sklearn.cluster    import KMeans
from sklearn.decomposition import PCA

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

# Resampling (class imbalance)
from imblearn.over_sampling import SMOTE

# Explainability
import shap


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
SEED       = 42
OUTPUT_DIR = "outputs"
MODEL_DIR  = "models"

# ← Update these paths to your local copies of the data files ←
DATA_PATHS = {
    "main":         "data/waitsure_200_merged.csv",
    "confirmation": "data/Railway Ticket Confirmation.csv",
    "waitlist":     "data/Railway Ticket WaitingList Data.csv",
}

PALETTE = {
    "primary":   "#FF6B35",
    "secondary": "#1E3A5F",
    "accent":    "#00B4D8",
    "success":   "#06D6A0",
    "danger":    "#EF233C",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)
sns.set_theme(style="whitegrid")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATA LOADING & MERGING
# ──────────────────────────────────────────────────────────────────────────────
def load_and_merge(paths: dict) -> pd.DataFrame:
    """
    Load the three datasets and merge into a single analysis-ready DataFrame.

    Pipeline:
        ① Load waitsure_200_merged (small, high-quality, labelled)
        ② Load Railway Ticket Confirmation (30 k IRCTC PNR records)
        ③ Standardise column names, derive days_before_travel, encode target
        ④ Concat both frames on common feature columns
    """
    print("📂  Loading datasets …")

    # ① Core dataset
    df_main = pd.read_csv(paths["main"])
    print(f"  ✓ waitsure_200_merged        → {df_main.shape}")

    # ② Large IRCTC confirmation dataset
    df_conf = pd.read_csv(paths["confirmation"])
    df_conf.columns = [c.strip() for c in df_conf.columns]
    print(f"  ✓ Railway Ticket Confirmation → {df_conf.shape}")

    # ③ Standardise confirmation DF
    rename_map = {
        "Train Number":          "train_no",
        "Class of Travel":       "travel_class",
        "Quota":                 "quota",
        "Source Station":        "source_station",
        "Destination Station":   "destination",
        "Date of Journey":       "journey_date",
        "Booking Date":          "booking_date",
        "Waitlist Position":     "waitlist_raw",
        "Travel Distance":       "travel_distance",
        "Seat Availability":     "total_seats",
        "Holiday or Peak Season":"season",
        "Train Type":            "train_type",
        "Confirmation Status":   "confirmed_str",
    }
    df_conf = df_conf.rename(columns=rename_map)
    df_conf["confirmed"] = (
        df_conf["confirmed_str"].str.strip() == "Confirmed"
    ).astype(int)
    df_conf["waitlist_number"] = (
        pd.to_numeric(
            df_conf["waitlist_raw"].astype(str)
                .str.extract(r"(\d+)")[0], errors="coerce"
        ).fillna(0).astype(int)
    )
    df_conf["journey_date"] = pd.to_datetime(df_conf["journey_date"], errors="coerce")
    df_conf["booking_date"] = pd.to_datetime(df_conf["booking_date"], errors="coerce")
    df_conf["days_before_travel"] = (
        df_conf["journey_date"] - df_conf["booking_date"]
    ).dt.days
    df_conf["historical_confirm_rate"] = np.nan
    df_conf["cancellation_trend"]      = np.nan

    # ④ Merge on common columns
    COMMON = [
        "train_no", "train_type", "travel_class", "quota",
        "source_station", "destination", "season",
        "days_before_travel", "waitlist_number", "total_seats",
        "historical_confirm_rate", "cancellation_trend", "confirmed",
    ]
    df_m = df_main[[c for c in COMMON if c in df_main.columns]].copy()
    df_c = df_conf[[c for c in COMMON if c in df_conf.columns]].copy()
    df   = pd.concat([df_m, df_c], ignore_index=True).dropna(subset=["confirmed"])

    print(f"\n  ► Final merged shape : {df.shape}")
    print(f"  ► Class distribution : {df['confirmed'].value_counts().to_dict()}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────
CAT_COLS = [
    "train_type", "travel_class", "quota",
    "season", "source_station", "destination",
]

def engineer_features(df: pd.DataFrame,
                       encoders: dict = None,
                       fit: bool = True) -> tuple:
    """
    Create domain-informed features from raw IRCTC columns.

    New features:
        wl_seat_ratio    — waitlist position ÷ total seats
        booking_urgency  — categorical bucket: last-minute vs early
        confirm_signal   — historical rate discounted by WL depth
        cancel_pressure  — cancellation rate per seat
        is_premium_class — flag for 1A / 2A classes

    Returns (X_array, y_array, encoders, feature_names, scaler)
    """
    df = df.copy()

    # ── categorical encoding ──────────────────────────────────────
    if encoders is None:
        encoders = {}
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip().str.upper()
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])

    # ── derived numeric features ──────────────────────────────────
    df["wl_seat_ratio"] = (
        df["waitlist_number"] / df["total_seats"].replace(0, np.nan)
    ).fillna(0)

    df["booking_urgency"] = pd.cut(
        df["days_before_travel"].fillna(0),
        bins=[-1, 3, 7, 14, 30, 365],
        labels=[4, 3, 2, 1, 0],
    ).astype(float)

    df["confirm_signal"] = (
        df["historical_confirm_rate"] /
        (df["waitlist_number"].replace(0, 0.5) ** 0.5)
    ).fillna(0)

    df["cancel_pressure"] = (
        df["cancellation_trend"] / df["total_seats"].replace(0, np.nan)
    ).fillna(0)

    df["is_premium_class"] = df["travel_class"].isin([0, 1]).astype(int)

    # ── split X / y ───────────────────────────────────────────────
    TARGET       = "confirmed"
    feat_names   = [c for c in df.columns if c != TARGET]
    X_raw        = df[feat_names].fillna(df[feat_names].median(numeric_only=True)).values.astype(float)
    y            = df[TARGET].values

    # ── scale ─────────────────────────────────────────────────────
    scaler = RobustScaler()
    X      = scaler.fit_transform(X_raw) if fit else scaler.transform(X_raw)

    return X, y, encoders, feat_names, scaler


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MODEL DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
def build_models() -> dict:
    """
    Return all ML models used in WaitSure.

    Module mapping from syllabus:
        Supervised-1  : Logistic Regression
        Supervised-2  : Decision Tree, Random Forest, Ensemble Methods
        Gradient/Boost: XGBoost, LightGBM  (state-of-the-art)
        Stacking      : StackingClassifier (meta-learner)
        Voting        : VotingClassifier   (soft ensemble)
    """
    lr  = LogisticRegression(
        C=1.0, max_iter=1000,
        class_weight="balanced", solver="lbfgs", random_state=SEED,
    )
    dt  = DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED,
    )
    rf  = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=SEED, verbosity=0,
    )
    lgbm = LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        class_weight="balanced", random_state=SEED, verbose=-1,
    )
    # Stacking: RF + XGB + LR as meta
    stacking = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lr", lr)],
        final_estimator=LogisticRegression(max_iter=500, random_state=SEED),
        cv=5, passthrough=False, n_jobs=-1,
    )
    # Voting: RF + XGB + LGBM soft vote
    voting = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        voting="soft", n_jobs=-1,
    )
    return {
        "Logistic Regression": lr,
        "Decision Tree":       dt,
        "Random Forest":       rf,
        "XGBoost":             xgb,
        "LightGBM":            lgbm,
        "Stacking Ensemble":   stacking,
        "Voting Ensemble":     voting,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — TRAINING & EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test,
                       feat_names: list) -> tuple:
    """
    Train all models, collect metrics, identify best model.

    Metrics reported (from syllabus):
        Supervised-1 : MAE, RMSE, R²  (on predicted probabilities)
        Supervised-2 : Accuracy, F1, AUC-ROC, 5-fold CV AUC
    """
    models  = build_models()
    results = {}
    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("\n🤖  Training Models …\n")
    for name, model in models.items():
        print(f"  → {name:<26}", end="", flush=True)
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba") else np.zeros(len(y_test))
        )
        # Classification metrics
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_proba) if y_proba.sum() > 0 else 0.5

        # Regression-style metrics on probabilities (syllabus Module-1)
        mae  = mean_absolute_error(y_test, y_proba)
        rmse = mean_squared_error(y_test, y_proba) ** 0.5
        r2   = r2_score(y_test, y_proba)

        # Cross-validation
        cv_auc = cross_val_score(
            model, X_train, y_train,
            cv=skf, scoring="roc_auc", n_jobs=-1,
        )

        results[name] = {
            "model":       model,
            "y_pred":      y_pred,
            "y_proba":     y_proba,
            "accuracy":    acc,
            "f1":          f1,
            "auc":         auc,
            "mae":         mae,
            "rmse":        rmse,
            "r2":          r2,
            "cv_auc_mean": cv_auc.mean(),
            "cv_auc_std":  cv_auc.std(),
        }
        print(f"ACC={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}  "
              f"CV_AUC={cv_auc.mean():.3f}±{cv_auc.std():.3f}")

    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\n🏆  Best Model → {best_name}  (AUC={results[best_name]['auc']:.4f})")
    return results, best_name


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — UNSUPERVISED ANALYSIS (K-Means + PCA)
# ──────────────────────────────────────────────────────────────────────────────
def run_unsupervised(X: np.ndarray, y: np.ndarray,
                     feat_names: list, n_clusters: int = 4):
    """
    K-Means: group routes/tickets by confirmation behaviour.
    PCA:     reduce dimensionality, identify key influencers.
    """
    print("\n🔍  Unsupervised Analysis …")

    # ── PCA ───────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X)
    exp_var = pca.explained_variance_ratio_
    print(f"  PCA: PC1={exp_var[0]:.2%}  PC2={exp_var[1]:.2%}  "
          f"total={sum(exp_var):.2%}")

    # Top PCA loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feat_names, columns=["PC1", "PC2"]
    ).abs()
    print(f"  Top PC1 features: {loadings['PC1'].nlargest(3).index.tolist()}")

    # ── K-Means ───────────────────────────────────────────────────
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    cluster_labels = km.fit_predict(X)
    print(f"  K-Means inertia (K={n_clusters}): {km.inertia_:.2f}")

    # Elbow curve
    inertias = []
    for k in range(2, 11):
        kk = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        kk.fit(X[:10000])          # subsample for speed
        inertias.append(kk.inertia_)

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Unsupervised Analysis — PCA & K-Means", fontsize=14, fontweight="bold")

    # PCA by label
    sc = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                         c=y, cmap="RdYlGn", alpha=0.4, s=8, edgecolors="none")
    axes[0].set_title("PCA — By Confirmation Label")
    axes[0].set_xlabel(f"PC1 ({exp_var[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({exp_var[1]:.1%})")
    plt.colorbar(sc, ax=axes[0])

    # PCA by cluster
    sc2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                          c=cluster_labels, cmap="tab10", alpha=0.4, s=8, edgecolors="none")
    axes[1].set_title(f"PCA — K-Means Clusters (K={n_clusters})")
    axes[1].set_xlabel(f"PC1 ({exp_var[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({exp_var[1]:.1%})")
    plt.colorbar(sc2, ax=axes[1])

    # Elbow curve
    axes[2].plot(range(2, 11), inertias, "o-", color=PALETTE["primary"], lw=2, ms=8)
    axes[2].axvline(n_clusters, ls="--", color=PALETTE["accent"],
                    label=f"Chosen K={n_clusters}")
    axes[2].set_title("K-Means Elbow Curve")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Inertia")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/unsupervised_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ unsupervised_analysis.png")

    return cluster_labels, X_pca, km, pca


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — VISUALISATIONS
# ──────────────────────────────────────────────────────────────────────────────
def generate_all_plots(results: dict, best_name: str,
                       X_test: np.ndarray, y_test: np.ndarray,
                       feat_names: list, df_raw: pd.DataFrame = None):
    """Generate all evaluation and insight plots."""
    print("\n📊  Generating Visualisations …")

    _plot_model_comparison(results, best_name)
    _plot_confusion_matrix(results, best_name, y_test)
    _plot_roc_curves(results, y_test)
    _plot_pr_curves(results, y_test)
    _plot_feature_importance(results, feat_names)
    _plot_shap(results, best_name, X_test, y_test, feat_names)
    _plot_shap_beeswarm(results, best_name, X_test, y_test, feat_names)
    _plot_shap_waterfall(results, best_name, X_test, y_test, feat_names)
    if df_raw is not None:
        _plot_correlation_heatmap(df_raw)
    _plot_domain_insights()
    print("  ✓ All plots saved to outputs/")


def _plot_model_comparison(results, best_name):
    names   = list(results.keys())
    metrics = ["accuracy", "f1", "auc", "cv_auc_mean"]
    labels  = ["Accuracy", "F1 Score", "AUC-ROC", "CV AUC"]
    colors  = [PALETTE["primary"], PALETTE["secondary"],
               PALETTE["accent"], PALETTE["success"]]

    x, w = np.arange(len(names)), 0.2
    fig, ax = plt.subplots(figsize=(15, 6))
    for i, (met, lab, col) in enumerate(zip(metrics, labels, colors)):
        vals = [results[n][met] for n in names]
        bars = ax.bar(x + i * w, vals, w, label=lab, color=col, alpha=0.87)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold")

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=9)
    ax.set_ylim(0.92, 1.03)
    ax.set_title("WaitSure — Model Performance Comparison",
                 fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    best_idx = names.index(best_name)
    ax.axvline(best_idx + w * 1.5, color=PALETTE["danger"],
               ls="--", lw=1.5, label=f"Best: {best_name}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ model_comparison.png")


def _plot_confusion_matrix(results, best_name, y_test):
    y_pred = results[best_name]["y_pred"]
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Confirmed", "Confirmed"], fontsize=10)
    ax.set_yticklabels(["Not Confirmed", "Confirmed"], fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=22, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ confusion_matrix.png")


def _plot_shap_beeswarm(results, best_name, X_test, y_test, feat_names):
    """
    SHAP Beeswarm plot (Image 4).
    Shows distribution of SHAP values per feature, colour-coded by feature value.
    """
    try:
        model    = results[best_name]["model"]
        rng      = np.random.RandomState(SEED)
        idx      = rng.choice(len(X_test), min(300, len(X_test)), replace=False)
        X_sample = X_test[idx]

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            sv_raw    = explainer.shap_values(X_sample)
            sv        = sv_raw[1] if isinstance(sv_raw, list) else sv_raw
        else:
            explainer = shap.LinearExplainer(model, X_sample)
            sv        = explainer.shap_values(X_sample)

        # Sort features by mean |SHAP| (ascending so most important is on top)
        order = np.argsort(np.abs(sv).mean(axis=0))

        fig, ax = plt.subplots(figsize=(10, 7))

        for rank, fi in enumerate(order):
            shap_vals   = sv[:, fi]
            feat_vals   = X_sample[:, fi]
            # Normalise feature value → [0, 1] for colour
            fv_norm     = (feat_vals - feat_vals.min()) / (feat_vals.ptp() + 1e-9)
            # Jitter on y-axis
            y_jitter    = rank + rng.uniform(-0.3, 0.3, size=len(shap_vals))
            sc = ax.scatter(
                shap_vals, y_jitter,
                c=fv_norm, cmap="coolwarm",
                alpha=0.6, s=14, linewidths=0,
                vmin=0, vmax=1,
            )

        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_yticks(range(len(feat_names)))
        ax.set_yticklabels([feat_names[i] for i in order], fontsize=9)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
        ax.set_title(f"SHAP Beeswarm — {best_name}",
                     fontsize=13, fontweight="bold", pad=12)
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Feature value", fontsize=9)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✓ shap_beeswarm.png")
    except Exception as e:
        print(f"  ⚠  shap_beeswarm skipped: {e}")


def _plot_shap_waterfall(results, best_name, X_test, y_test, feat_names):
    """
    SHAP Waterfall plot for the single highest-probability prediction (Image 1).
    Shows how each feature pushes the prediction above/below the base value.
    """
    try:
        model    = results[best_name]["model"]
        y_proba  = results[best_name]["y_proba"]
        # Pick the sample closest to the highest-confidence confirmation
        top_idx  = int(np.argmax(y_proba))
        X_single = X_test[[top_idx]]

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            sv_raw    = explainer.shap_values(X_single)
            sv_1d     = (sv_raw[1] if isinstance(sv_raw, list) else sv_raw)[0]
            base_val  = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )
        else:
            explainer = shap.LinearExplainer(model, X_test)
            sv_1d     = explainer.shap_values(X_single)[0]
            base_val  = explainer.expected_value

        # Sort by absolute impact
        order    = np.argsort(np.abs(sv_1d))
        n_show   = min(12, len(feat_names))
        order    = order[-n_show:]

        names_show  = [feat_names[i] for i in order]
        shap_show   = sv_1d[order]
        feat_show   = X_single[0, order]

        # Build waterfall
        running = base_val
        lefts, widths, colors_bar = [], [], []
        for s in shap_show:
            if s >= 0:
                lefts.append(running)
                colors_bar.append(PALETTE["danger"])
            else:
                lefts.append(running + s)
                colors_bar.append(PALETTE["accent"])
            widths.append(abs(s))
            running += s

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(
            names_show, widths, left=lefts,
            color=colors_bar, edgecolor="white", height=0.55,
        )
        for bar, s in zip(bars, shap_show):
            sign = "+" if s >= 0 else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{s:.2f}",
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white",
            )

        ax.axvline(base_val, color="gray", lw=1, ls="--", alpha=0.7)
        ax.axvline(running,  color=PALETTE["secondary"], lw=1.5, ls=":")
        ax.set_title(
            f"SHAP Waterfall — {best_name}\n"
            f"Sample prediction: f(x) = {running:.3f}   |   E[f(x)] = {base_val:.3f}",
            fontsize=12, fontweight="bold", pad=12,
        )
        ax.set_xlabel("Model output value", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Feature-value annotations on y-axis labels
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(
            [f"{n}  =  {feat_show[i]:.2g}" for i, n in enumerate(names_show)],
            fontsize=9,
        )
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✓ shap_waterfall.png")
    except Exception as e:
        print(f"  ⚠  shap_waterfall skipped: {e}")


# ── FIXED: ROC Curves ─────────────────────────────────────────────────────────
def _plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(9, 7))
    # Shaded diagonal (random baseline region)
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5,
            label="Random baseline  (AUC = 0.500)")
    colors = plt.cm.tab10.colors
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, lw=2, color=colors[i % 10],
                label=f"{name}  (AUC = {res['auc']:.3f})")
    ax.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ roc_curves.png")


# ── FIXED: Precision-Recall Curves ────────────────────────────────────────────
def _plot_pr_curves(results, y_test):
    baseline = y_test.mean()          # positive class prevalence
    fig, ax  = plt.subplots(figsize=(9, 7))
    # Random / no-skill baseline
    ax.axhline(baseline, color="k", lw=1.2, ls="--", alpha=0.5,
               label=f"Random baseline  (AP = {baseline:.3f})")
    colors = plt.cm.tab10.colors
    for i, (name, res) in enumerate(results.items()):
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        ap = average_precision_score(y_test, res["y_proba"])
        ax.plot(rec, prec, lw=2, color=colors[i % 10],
                label=f"{name}  (AP = {ap:.3f})")
        ax.fill_between(rec, prec, alpha=0.04, color=colors[i % 10])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves — All Models",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ pr_curves.png")


def _plot_feature_importance(results, feat_names):
    rf = results.get("Random Forest", {}).get("model")
    if rf is None or not hasattr(rf, "feature_importances_"):
        return
    imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values()
    bar_colors = [
        PALETTE["danger"] if v > imp.median() else PALETTE["accent"]
        for v in imp.values
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(imp.index, imp.values, color=bar_colors, edgecolor="white")
    for bar, v in zip(bars, imp.values):
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8)
    ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ feature_importance.png")


def _plot_shap(results, best_name, X_test, y_test, feat_names):
    """SHAP explainability — global bar + dependence scatter."""
    try:
        model = results[best_name]["model"]
        idx = np.random.RandomState(SEED).choice(len(X_test), min(300, len(X_test)), replace=False)
        X_sample = X_test[idx]

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            sv_raw    = explainer.shap_values(X_sample)
            sv        = sv_raw[1] if isinstance(sv_raw, list) else sv_raw
        else:
            explainer = shap.LinearExplainer(model, X_sample)
            sv        = explainer.shap_values(X_sample)

        mean_shap = np.abs(sv).mean(axis=0)
        top_idx   = np.argsort(mean_shap)[-12:]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"SHAP Explainability — {best_name}",
                     fontsize=13, fontweight="bold")

        axes[0].barh(
            [feat_names[i] for i in top_idx],
            mean_shap[top_idx],
            color=PALETTE["accent"],
        )
        axes[0].set_title("Mean |SHAP| — Global Feature Impact")
        axes[0].set_xlabel("|SHAP value|")
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        top_i = top_idx[-1]
        sc = axes[1].scatter(
            X_sample[:, top_i], sv[:, top_i],
            c=y_test[idx], cmap="RdYlGn",
            alpha=0.6, edgecolors="k", linewidths=0.2, s=40,
        )
        axes[1].axhline(0, color="k", lw=0.8, ls="--")
        axes[1].set_xlabel(f"Feature value: {feat_names[top_i]}")
        axes[1].set_ylabel("SHAP value (impact on prediction)")
        axes[1].set_title(f"SHAP Dependence — '{feat_names[top_i]}'")
        plt.colorbar(sc, ax=axes[1], label="Confirmed (1=Yes)")

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✓ shap_analysis.png")

    except Exception as e:
        print(f"  ⚠  SHAP skipped: {e}")


# ── NEW: Correlation Heatmap ───────────────────────────────────────────────────
def _plot_correlation_heatmap(df: pd.DataFrame):
    """
    Full-matrix feature correlation heatmap (diverging coolwarm palette).
    Matches Image 2 — symmetric matrix, annotated values, centred at 0.
    """
    num_df = df.select_dtypes(include=[np.number]).drop(
        columns=["confirmed"], errors="ignore"
    )
    if num_df.shape[1] < 2:
        print("  ⚠  correlation_heatmap skipped: not enough numeric columns")
        return
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, ax=ax,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        square=True, cbar_kws={"shrink": 0.75},
    )
    ax.set_title("Feature Correlation Heatmap",
                 fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ correlation_heatmap.png")


# ── NEW: SHAP Beeswarm ─────────────────────────────────────────────────────────
def _plot_shap_beeswarm(results, best_name, X_test, y_test, feat_names):
    """
    SHAP Beeswarm plot (matches Image 4).
    Each dot = one sample; x = SHAP value; colour = feature magnitude.
    Features sorted top-to-bottom by mean |SHAP| (most important on top).
    """
    try:
        model    = results[best_name]["model"]
        rng      = np.random.RandomState(SEED)
        idx      = rng.choice(len(X_test), min(300, len(X_test)), replace=False)
        X_sample = X_test[idx]

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            sv_raw    = explainer.shap_values(X_sample)
            sv        = sv_raw[1] if isinstance(sv_raw, list) else sv_raw
        else:
            explainer = shap.LinearExplainer(model, X_sample)
            sv        = explainer.shap_values(X_sample)

        # Sort features by mean |SHAP| ascending (so most important ends up on top)
        order = np.argsort(np.abs(sv).mean(axis=0))

        fig, ax = plt.subplots(figsize=(10, 7))

        for rank, fi in enumerate(order):
            shap_vals = sv[:, fi]
            feat_vals = X_sample[:, fi]
            # Normalise feature value → [0, 1] for colour mapping
            ptp = feat_vals.ptp()
            fv_norm = (feat_vals - feat_vals.min()) / (ptp if ptp > 0 else 1)
            # Y-jitter so points don't stack
            y_jitter = rank + rng.uniform(-0.3, 0.3, size=len(shap_vals))
            sc = ax.scatter(
                shap_vals, y_jitter,
                c=fv_norm, cmap="coolwarm",
                alpha=0.6, s=14, linewidths=0,
                vmin=0, vmax=1,
            )

        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_yticks(range(len(feat_names)))
        ax.set_yticklabels([feat_names[i] for i in order], fontsize=9)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
        ax.set_title(f"SHAP Beeswarm — {best_name}",
                     fontsize=13, fontweight="bold", pad=12)

        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Feature value", fontsize=9)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✓ shap_beeswarm.png")
    except Exception as e:
        print(f"  ⚠  shap_beeswarm skipped: {e}")


# ── NEW: SHAP Waterfall ────────────────────────────────────────────────────────
def _plot_shap_waterfall(results, best_name, X_test, y_test, feat_names):
    """
    SHAP Waterfall plot for the single highest-probability prediction (Image 1).
    Shows cumulative feature contributions from E[f(x)] up to f(x).
    Red bars push prediction up; blue bars push it down.
    """
    try:
        model   = results[best_name]["model"]
        y_proba = results[best_name]["y_proba"]
        # Use the sample with the highest predicted probability
        top_idx  = int(np.argmax(y_proba))
        X_single = X_test[[top_idx]]

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            sv_raw    = explainer.shap_values(X_single)
            sv_1d     = (sv_raw[1] if isinstance(sv_raw, list) else sv_raw)[0]
            base_val  = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else float(explainer.expected_value)
            )
        else:
            explainer = shap.LinearExplainer(model, X_test)
            sv_1d     = explainer.shap_values(X_single)[0]
            base_val  = float(explainer.expected_value)

        # Pick top-N features by absolute SHAP
        n_show  = min(12, len(feat_names))
        order   = np.argsort(np.abs(sv_1d))[-n_show:]

        names_show = [feat_names[i] for i in order]
        shap_show  = sv_1d[order]
        feat_show  = X_single[0, order]

        # Build waterfall positions
        running = base_val
        lefts, widths, bar_colors = [], [], []
        for s in shap_show:
            if s >= 0:
                lefts.append(running)
                bar_colors.append(PALETTE["danger"])   # red  → pushes up
            else:
                lefts.append(running + s)
                bar_colors.append(PALETTE["accent"])   # blue → pushes down
            widths.append(abs(s))
            running += s

        final_pred = running

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(
            names_show, widths, left=lefts,
            color=bar_colors, edgecolor="white", height=0.55,
        )
        # Value labels inside bars
        for bar, s in zip(bars, shap_show):
            sign = "+" if s >= 0 else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{s:.2f}",
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white",
            )

        # Reference lines
        ax.axvline(base_val,   color="gray",              lw=1,   ls="--", alpha=0.7,
                   label=f"E[f(X)] = {base_val:.3f}")
        ax.axvline(final_pred, color=PALETTE["secondary"], lw=1.5, ls=":",
                   label=f"f(x) = {final_pred:.3f}")

        ax.set_title(
            f"SHAP Waterfall — {best_name}\n"
            f"f(x) = {final_pred:.3f}   |   E[f(X)] = {base_val:.3f}",
            fontsize=12, fontweight="bold", pad=12,
        )
        ax.set_xlabel("Model output value", fontsize=10)
        ax.legend(fontsize=9, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate y-tick labels with feature values
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(
            [f"{n}  =  {feat_show[i]:.2g}" for i, n in enumerate(names_show)],
            fontsize=9,
        )
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✓ shap_waterfall.png")
    except Exception as e:
        print(f"  ⚠  shap_waterfall skipped: {e}")


def _plot_domain_insights():
    """Domain-specific insight plots from the raw IRCTC data."""
    try:
        df_raw = pd.read_csv(DATA_PATHS["confirmation"])
        df_raw.columns = [c.strip() for c in df_raw.columns]
        df_raw["confirmed_int"] = (
            df_raw["Confirmation Status"].str.strip() == "Confirmed"
        ).astype(int)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("IRCTC Domain Insights", fontsize=13, fontweight="bold")

        # 1) Confirmation rate by travel class
        class_rate = (
            df_raw.groupby("Class of Travel")["confirmed_int"]
            .mean().sort_values()
        )
        bar_colors = [
            PALETTE["success"] if v >= 0.6 else PALETTE["danger"]
            for v in class_rate.values
        ]
        axes[0].barh(class_rate.index, class_rate.values, color=bar_colors)
        axes[0].set_title("Confirmation Rate by Travel Class", fontweight="bold")
        axes[0].set_xlabel("Confirmation Rate")
        for i, v in enumerate(class_rate.values):
            axes[0].text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # 2) WL position bucket vs confirmation
        df_raw["wl_num"] = pd.to_numeric(
            df_raw["Waitlist Position"].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce",
        )
        df_raw["wl_bucket"] = pd.cut(
            df_raw["wl_num"],
            bins=[0, 10, 20, 30, 50, 200],
            labels=["WL 1–10", "WL 11–20", "WL 21–30", "WL 31–50", "WL 50+"],
        )
        wl_rate = df_raw.groupby("wl_bucket", observed=True)["confirmed_int"].mean()
        wl_colors = [
            PALETTE["success"], PALETTE["accent"],
            PALETTE["primary"], PALETTE["danger"], "#6B2D8B",
        ]
        axes[1].bar(wl_rate.index.astype(str), wl_rate.values, color=wl_colors)
        axes[1].set_title("Confirmation Rate by Waitlist Position", fontweight="bold")
        axes[1].set_xlabel("Waitlist Bucket")
        axes[1].set_ylabel("Confirmation Rate")
        for i, v in enumerate(wl_rate.values):
            axes[1].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=9)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/domain_insights.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✓ domain_insights.png")

    except Exception as e:
        print(f"  ⚠  domain_insights skipped: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 — INFERENCE / SINGLE-TICKET PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────
def predict_ticket(
    *,
    model,
    scaler: RobustScaler,
    encoders: dict,
    feat_names: list,
    # ticket details
    train_no:               int   = 12301,
    train_type:             str   = "Rajdhani",
    source_station:         str   = "NDLS",
    destination:            str   = "HWH",
    travel_class:           str   = "3A",
    quota:                  str   = "General",
    season:                 str   = "Festival",
    days_before_travel:     int   = 18,
    waitlist_number:        int   = 14,
    total_seats:            int   = 72,
    historical_confirm_rate:float = 0.82,
    cancellation_trend:     float = 12.0,
) -> dict:
    """
    Predict confirmation probability for a single IRCTC ticket.

    Returns dict with keys: probability, label, verdict, advice.

    Example
    -------
    >>> result = predict_ticket(model=best_model, scaler=scaler,
    ...     encoders=encoders, feat_names=feat_names,
    ...     train_no=12301, waitlist_number=14, season='Festival')
    """
    ticket = {
        "train_no":                train_no,
        "train_type":              train_type,
        "travel_class":            travel_class,
        "quota":                   quota,
        "source_station":          source_station,
        "destination":             destination,
        "season":                  season,
        "days_before_travel":      days_before_travel,
        "waitlist_number":         waitlist_number,
        "total_seats":             total_seats,
        "historical_confirm_rate": historical_confirm_rate,
        "cancellation_trend":      cancellation_trend,
    }
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

    # Derived features
    df["wl_seat_ratio"]    = (df["waitlist_number"] / df["total_seats"].replace(0, np.nan)).fillna(0)
    df["booking_urgency"]  = pd.cut(df["days_before_travel"].fillna(0),
                                    bins=[-1,3,7,14,30,365], labels=[4,3,2,1,0]).astype(float)
    df["confirm_signal"]   = (df["historical_confirm_rate"] / (df["waitlist_number"].replace(0,0.5)**0.5)).fillna(0)
    df["cancel_pressure"]  = (df["cancellation_trend"] / df["total_seats"].replace(0, np.nan)).fillna(0)
    df["is_premium_class"] = df["travel_class"].isin([0, 1]).astype(int)

    # Align to training feature order
    missing = set(feat_names) - set(df.columns)
    for m in missing:
        df[m] = 0.0
    X = df[feat_names].fillna(0).values.astype(float)
    X = scaler.transform(X)

    prob  = model.predict_proba(X)[0, 1]
    label = int(prob >= 0.5)

    # Decision thresholds
    if prob >= 0.75:
        verdict = "✅  LIKELY CONFIRMED"
        advice  = "Good chance of getting a seat! Keep monitoring the chart closer to departure."
    elif prob >= 0.50:
        verdict = "⚠️   POSSIBLE CONFIRMATION"
        advice  = "Borderline. Consider booking an alternate train or RAC seat as backup."
    else:
        verdict = "❌  UNLIKELY TO CONFIRM"
        advice  = "Recommend booking an alternative. Try Tatkal quota or a nearby date."

    bar_len = 25
    filled  = int(prob * bar_len)
    prog    = "█" * filled + "░" * (bar_len - filled)

    banner = f"""
╔══════════════════════════════════════════════════════════════════╗
║                     🚂  WaitSure  v1.0                           ║
║             IRCTC Waitlist Confirmation Predictor                ║
╚══════════════════════════════════════════════════════════════════╝
  Train     : {train_no}  ({train_type})
  Route     : {source_station} → {destination}
  Class     : {travel_class}    |    Quota    : {quota}
  Journey   : {days_before_travel} days away  |    Season   : {season}
  WL Pos.   : WL/{waitlist_number}            |    Seats    : {total_seats}

  ─────────────────────────────────────────────────────────────────
  📊  PREDICTION
  ─────────────────────────────────────────────────────────────────
  Probability  :  {prob:.1%}   [{prog}]
  Verdict      :  {verdict}

  ─────────────────────────────────────────────────────────────────
  💡  ADVICE
  ─────────────────────────────────────────────────────────────────
  {advice}
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    return {"probability": prob, "label": label, "verdict": verdict,
            "advice": advice, "banner": banner}


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 — REPORT GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def save_report(results: dict, feat_names: list, y_test, best_name: str):
    """Save CSV summary of all model metrics + classification report."""
    rows = []
    for name, res in results.items():
        rows.append({
            "Model":      name,
            "Accuracy":   round(res["accuracy"],    4),
            "F1 Score":   round(res["f1"],          4),
            "AUC-ROC":    round(res["auc"],         4),
            "CV AUC":     round(res["cv_auc_mean"], 4),
            "CV Std":     round(res["cv_auc_std"],  4),
            "MAE":        round(res["mae"],          4),
            "RMSE":       round(res["rmse"],         4),
            "R² Score":   round(res["r2"],           4),
            "Best Model": "★" if name == best_name else "",
        })
    df_rep = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)
    df_rep.to_csv(f"{OUTPUT_DIR}/model_report.csv", index=False)

    # Feature importance CSV
    rf = results.get("Random Forest", {}).get("model")
    if rf and hasattr(rf, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature":    feat_names,
            "Importance": rf.feature_importances_,
        }).sort_values("Importance", ascending=False)
        fi.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)

    # Detailed classification report (best model)
    cr = classification_report(
        y_test,
        results[best_name]["y_pred"],
        target_names=["Not Confirmed", "Confirmed"],
    )
    with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
        f.write(f"Best Model: {best_name}\n\n{cr}")

    print(f"\n📄  Reports saved to {OUTPUT_DIR}/")
    print(df_rep.to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 — MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  🚂  WaitSure — Full ML Pipeline  🚂")
    print("=" * 70)

    # ① Load & merge data
    df = load_and_merge(DATA_PATHS)

    # ② Feature engineering
    print("\n⚙️   Feature Engineering …")
    X, y, encoders, feat_names, scaler = engineer_features(df, fit=True)
    print(f"  Feature matrix : {X.shape}")
    print(f"  Features       : {feat_names}")

    # ③ SMOTE resampling
    sm = SMOTE(random_state=SEED)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"\n  After SMOTE: {X_res.shape}  →  {dict(zip(*np.unique(y_res, return_counts=True)))}")

    # ④ Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=SEED
    )
    print(f"  Train : {X_train.shape}   Test : {X_test.shape}")

    # ⑤ Train & evaluate
    results, best_name = train_and_evaluate(
        X_train, X_test, y_train, y_test, feat_names
    )

    # ⑥ Unsupervised (on original, unsmoted X)
    cluster_labels, X_pca, km, pca = run_unsupervised(X, y, feat_names)

    # ⑦ Visualise — pass df for correlation heatmap
    generate_all_plots(results, best_name, X_test, y_test, feat_names, df_raw=df)

    # ⑧ Classification report (best model)
    print(f"\n📋  Classification Report — {best_name}")
    print(classification_report(
        y_test, results[best_name]["y_pred"],
        target_names=["Not Confirmed", "Confirmed"],
    ))

    # ⑨ Decision tree rules
    dt = results.get("Decision Tree", {}).get("model")
    if dt:
        rules = export_text(dt, feature_names=feat_names, max_depth=4)
        with open(f"{OUTPUT_DIR}/decision_tree_rules.txt", "w") as f:
            f.write(rules)
        print(f"  ✓ decision_tree_rules.txt")

    # ⑩ Save report & models
    save_report(results, feat_names, y_test, best_name)
    best_model = results[best_name]["model"]
    joblib.dump(best_model, f"{MODEL_DIR}/waitsure_best_model.pkl")
    joblib.dump(scaler,     f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(encoders,   f"{MODEL_DIR}/encoders.pkl")
    with open(f"{MODEL_DIR}/feat_names.json", "w") as f:
        json.dump(feat_names, f)
    print(f"  ✓ Models saved to {MODEL_DIR}/")

    # ⑪ Demo prediction — the WaitSure headline output
    print("\n🎯  Demo Prediction")
    predict_ticket(
        model=best_model,
        scaler=scaler,
        encoders=encoders,
        feat_names=feat_names,
        train_no=12301,
        train_type="Rajdhani",
        source_station="NDLS",
        destination="HWH",
        travel_class="3A",
        quota="General",
        season="Festival",
        days_before_travel=18,
        waitlist_number=14,
        total_seats=72,
        historical_confirm_rate=0.82,
        cancellation_trend=12.0,
    )

    print("\n✅  WaitSure pipeline complete!  All outputs → ./outputs/\n")
    return results, best_model, scaler, encoders, feat_names


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()