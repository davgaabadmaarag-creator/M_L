import argparse
import os
import re
import sys
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Teen_Mental_Health_Dataset.csv"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)
MPL_CONFIG_DIR = RESULT_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

DROP_COLUMNS = ["depression_label"]
LEVEL_COLUMNS = ["stress_level", "anxiety_level", "addiction_level"]
RISK_ORDER = ["Low", "Medium", "High"]
STRESS_ORDER = ["Low stress", "Medium stress", "High stress"]
BASELINE_MODEL_NAME = "Dummy Baseline"
SIGNAL_THRESHOLD = 0.10
PROFILE_FEATURES = [
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
]


def clean_col(name: str) -> str:
    """Convert column names to simple snake_case names."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def create_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """Create Low/Medium/High risk labels from stress, anxiety, addiction."""
    missing = [col for col in LEVEL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")

    df = df.copy()
    df["mental_health_risk_score"] = df[LEVEL_COLUMNS].mean(axis=1)
    df["mental_health_risk_level"] = pd.cut(
        df["mental_health_risk_score"],
        bins=[0, 4, 7, 10],
        labels=RISK_ORDER,
        include_lowest=True,
        right=False,
    )

    df.loc[df["mental_health_risk_score"] == 10, "mental_health_risk_level"] = "High"
    df["mental_health_risk_level"] = df["mental_health_risk_level"].astype(str)
    return df


def add_stress_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add an interpretable stress group for practical analysis."""
    df = df.copy()
    df["stress_group"] = pd.cut(
        df["stress_level"],
        bins=[0, 4, 7, 10],
        labels=STRESS_ORDER,
        include_lowest=True,
        right=False,
    )
    df.loc[df["stress_level"] == 10, "stress_group"] = "High stress"
    df["stress_group"] = df["stress_group"].astype(str)
    return df


def make_one_hot_encoder() -> OneHotEncoder:
    """Support both old and new scikit-learn OneHotEncoder APIs."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} was not found. Put the dataset in the same folder as this script."
        )

    df = pd.read_csv(DATA_FILE)
    df.columns = [clean_col(c) for c in df.columns]
    df = df.drop_duplicates()

    existing_drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=existing_drop_cols)
    df = create_risk_level(df)
    df = add_stress_group(df)
    return df, existing_drop_cols


def save_basic_visualizations(df: pd.DataFrame) -> None:
    """Save the basic EDA charts used in the report."""
    for col in LEVEL_COLUMNS:
        if col in df.columns:
            plt.figure(figsize=(7, 5))
            df[col].value_counts().sort_index().plot(kind="bar")
            plt.title(f"Distribution of {col}")
            plt.xlabel("Level (1-10)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(RESULT_DIR / f"{col}_distribution.png", dpi=200)
            plt.close()

    plt.figure(figsize=(7, 5))
    df["mental_health_risk_level"].value_counts().reindex(RISK_ORDER).plot(kind="bar")
    plt.title("Mental Health Risk Level Distribution")
    plt.xlabel("Risk Level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "mental_health_risk_distribution.png", dpi=200)
    plt.savefig(RESULT_DIR / "target_distribution.png", dpi=200)
    plt.close()

    if "platform_usage" in df.columns:
        platform_avg = df.groupby("platform_usage")["mental_health_risk_score"].mean()
        platform_avg = platform_avg.sort_values(ascending=False)
        platform_avg.to_csv(RESULT_DIR / "average_risk_by_platform.csv", encoding="utf-8-sig")

        plt.figure(figsize=(7, 5))
        platform_avg.plot(kind="bar")
        plt.title("Average Risk Score by Platform")
        plt.xlabel("Platform")
        plt.ylabel("Average Risk Score")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "average_risk_by_platform.png", dpi=200)
        plt.close()

    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        corr.to_csv(RESULT_DIR / "correlation_matrix.csv", encoding="utf-8-sig")

        plt.figure(figsize=(10, 8))
        plt.imshow(corr, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "correlation_matrix.png", dpi=200)
        plt.close()


def available_profile_features(df: pd.DataFrame) -> list[str]:
    return [col for col in PROFILE_FEATURES if col in df.columns]


def save_practical_analysis(df: pd.DataFrame) -> None:
    """Create practical summaries such as high-stress behavior profiles."""
    profile_cols = available_profile_features(df)
    high_stress = df[df["stress_level"] >= 7]
    not_high_stress = df[df["stress_level"] < 7]

    risk_profile = df.groupby("mental_health_risk_level")[profile_cols].mean()
    risk_profile = risk_profile.reindex(RISK_ORDER)
    risk_profile.to_csv(RESULT_DIR / "risk_level_profile.csv", encoding="utf-8-sig")

    stress_profile = df.groupby("stress_group")[profile_cols].mean()
    stress_profile = stress_profile.reindex(STRESS_ORDER)
    stress_profile.to_csv(RESULT_DIR / "stress_group_profile.csv", encoding="utf-8-sig")

    if "platform_usage" in df.columns:
        platform_summary = df.groupby("platform_usage").agg(
            count=("platform_usage", "size"),
            avg_stress_level=("stress_level", "mean"),
            avg_risk_score=("mental_health_risk_score", "mean"),
            avg_social_media_hours=("daily_social_media_hours", "mean"),
            avg_sleep_hours=("sleep_hours", "mean"),
            avg_screen_time_before_sleep=("screen_time_before_sleep", "mean"),
        )
        high_stress_rate = df.assign(is_high_stress=df["stress_level"] >= 7)
        high_stress_rate = high_stress_rate.groupby("platform_usage")["is_high_stress"].mean() * 100
        platform_summary["high_stress_rate_pct"] = high_stress_rate
        platform_summary = platform_summary.sort_values("high_stress_rate_pct", ascending=False)
        platform_summary.to_csv(RESULT_DIR / "platform_stress_summary.csv", encoding="utf-8-sig")

        plt.figure(figsize=(8, 5))
        platform_summary["high_stress_rate_pct"].plot(kind="bar")
        plt.title("High Stress Rate by Platform")
        plt.xlabel("Platform")
        plt.ylabel("High stress rate (%)")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "high_stress_by_platform.png", dpi=200)
        plt.close()

    trend = df.groupby("stress_level")["daily_social_media_hours"].mean()
    trend.to_csv(RESULT_DIR / "social_media_by_stress_level.csv", encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    trend.plot(kind="line", marker="o")
    plt.title("Average Social Media Hours by Stress Level")
    plt.xlabel("Stress level")
    plt.ylabel("Average daily social media hours")
    plt.xticks(sorted(df["stress_level"].dropna().unique()))
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "social_media_by_stress_level.png", dpi=200)
    plt.close()

    plot_cols = [
        col
        for col in [
            "daily_social_media_hours",
            "sleep_hours",
            "screen_time_before_sleep",
            "physical_activity",
        ]
        if col in risk_profile.columns
    ]
    if plot_cols:
        plt.figure(figsize=(9, 5))
        risk_profile[plot_cols].plot(kind="bar")
        plt.title("Lifestyle Profile by Risk Level")
        plt.xlabel("Risk level")
        plt.ylabel("Average value")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "risk_level_lifestyle_profile.png", dpi=200)
        plt.close()

    lines = [
        "Practical Insights for Teen Mental Health ML Project",
        "=" * 60,
        f"Total rows after cleaning: {len(df)}",
        f"High stress rows (stress_level >= 7): {len(high_stress)} "
        f"({len(high_stress) / len(df) * 100:.1f}%)",
        "",
        "1. High stress group profile",
        "-" * 30,
    ]

    for col in profile_cols:
        high_mean = high_stress[col].mean()
        other_mean = not_high_stress[col].mean()
        diff = high_mean - other_mean
        lines.append(
            f"{col}: high stress avg={high_mean:.2f}, other avg={other_mean:.2f}, diff={diff:+.2f}"
        )

    if "platform_usage" in df.columns:
        lines.extend(["", "2. Most common platforms among high-stress rows", "-" * 48])
        platform_share = high_stress["platform_usage"].value_counts(normalize=True).mul(100)
        for platform, pct in platform_share.items():
            lines.append(f"{platform}: {pct:.1f}%")

        lines.extend(["", "3. Platform-level high stress rate", "-" * 36])
        for platform, row in platform_summary.iterrows():
            lines.append(
                f"{platform}: high_stress_rate={row['high_stress_rate_pct']:.1f}%, "
                f"avg_social_hours={row['avg_social_media_hours']:.2f}, "
                f"avg_sleep={row['avg_sleep_hours']:.2f}"
            )

    lines.extend(
        [
            "",
            "4. How to use these findings",
            "-" * 30,
            "Use this section to explain practical patterns, for example:",
            "- People in the high-stress group can be compared with other groups by social media hours.",
            "- Platform summaries show where high stress rates are relatively higher in this dataset.",
            "- Sleep hours, screen time before sleep, and physical activity help explain lifestyle differences.",
            "",
            "Important: These results are educational and data-driven. They are not a clinical diagnosis.",
        ]
    )

    (RESULT_DIR / "practical_insights.txt").write_text("\n".join(lines), encoding="utf-8")


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    target_col = "mental_health_risk_level"
    feature_drop_cols = [target_col, "mental_health_risk_score", "stress_group"] + LEVEL_COLUMNS
    X = df.drop(columns=[col for col in feature_drop_cols if col in df.columns])
    y = df[target_col]

    valid_idx = y.notna() & (y.astype(str).str.lower() != "nan")
    X = X.loc[valid_idx].copy()
    y = y.loc[valid_idx].copy()

    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_models(numeric_features: list[str], categorical_features: list[str]) -> dict[str, Pipeline]:
    return {
        BASELINE_MODEL_NAME: Pipeline(
            steps=[
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "Naive Bayes": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_features, categorical_features)),
                ("model", GaussianNB()),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_features, categorical_features)),
                ("model", DecisionTreeClassifier(max_depth=4, random_state=42)),
            ]
        ),
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_features, categorical_features)),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        ),
    }


def save_decision_tree_feature_importance(model: Pipeline) -> None:
    """Save the most useful explanatory output from the Decision Tree model."""
    try:
        preprocessor = model.named_steps["preprocess"]
        tree = model.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return

    clean_names = [name.split("__", 1)[-1] for name in feature_names]
    importance_df = pd.DataFrame(
        {
            "feature": clean_names,
            "importance": tree.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(RESULT_DIR / "decision_tree_feature_importance.csv", index=False, encoding="utf-8-sig")

    top_features = importance_df.head(10).sort_values("importance")
    if not top_features.empty:
        plt.figure(figsize=(8, 5))
        plt.barh(top_features["feature"], top_features["importance"])
        plt.title("Top Decision Tree Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "decision_tree_feature_importance.png", dpi=200)
        plt.close()


def train_and_evaluate_models(
    df: pd.DataFrame,
    existing_drop_cols: list[str],
) -> tuple[pd.DataFrame, str, Pipeline, pd.DataFrame, pd.Series]:
    X, y, numeric_features, categorical_features = split_features_target(df)
    models = build_models(numeric_features, categorical_features)

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    rows = []
    fitted_models = {}
    best_model_name = None
    best_selection_score = (-1.0, -1.0, -1.0)
    baseline_macro_f1 = None

    with open(RESULT_DIR / "model_results.txt", "w", encoding="utf-8") as f:
        f.write("Teen Mental Health ML Project Results\n")
        f.write("Depression label was removed and not used.\n")
        f.write("=" * 55 + "\n")
        f.write(f"Dataset shape after processing: {df.shape}\n")
        f.write(f"Removed columns: {existing_drop_cols}\n")
        f.write("Target column: mental_health_risk_level\n")
        f.write("Target creation: mean(stress_level, anxiety_level, addiction_level) => Low/Medium/High\n")
        f.write(f"Input numeric features: {numeric_features}\n")
        f.write(f"Input categorical features: {categorical_features}\n\n")
        f.write("Selection metric: Macro F1, with Balanced Accuracy and Accuracy as tie-breakers.\n")
        f.write("Dummy Baseline predicts the most frequent class and is used only for comparison.\n\n")
        f.write("Target distribution:\n")
        f.write(str(y.value_counts()))
        f.write("\n\n")

        for name, model in models.items():
            model.fit(X_train, y_train)
            fitted_models[name] = model
            if name == "Decision Tree":
                save_decision_tree_feature_importance(model)

            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            balanced_acc = balanced_accuracy_score(y_test, pred)
            macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_test, pred, average="weighted", zero_division=0)
            is_baseline = name == BASELINE_MODEL_NAME
            rows.append(
                {
                    "Model": name,
                    "Accuracy": round(acc, 4),
                    "Balanced Accuracy": round(balanced_acc, 4),
                    "Macro F1": round(macro_f1, 4),
                    "Weighted F1": round(weighted_f1, 4),
                    "Baseline": is_baseline,
                }
            )

            if is_baseline:
                baseline_macro_f1 = macro_f1

            f.write(f"\n{name}\n")
            f.write("-" * len(name) + "\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Balanced accuracy: {balanced_acc:.4f}\n")
            f.write(f"Macro F1: {macro_f1:.4f}\n")
            f.write(f"Weighted F1: {weighted_f1:.4f}\n")
            f.write("Classification report:\n")
            f.write(classification_report(y_test, pred, labels=RISK_ORDER, zero_division=0))
            f.write("\nConfusion matrix:\n")
            f.write(str(confusion_matrix(y_test, pred, labels=RISK_ORDER)))
            f.write("\n")

            selection_score = (macro_f1, balanced_acc, acc)
            if not is_baseline and selection_score > best_selection_score:
                best_selection_score = selection_score
                best_model_name = name

        f.write(
            f"\nBest non-baseline model: {best_model_name}, "
            f"Macro F1: {best_selection_score[0]:.4f}, "
            f"Balanced Accuracy: {best_selection_score[1]:.4f}, "
            f"Accuracy: {best_selection_score[2]:.4f}\n"
        )
        if baseline_macro_f1 is not None:
            f.write(f"Dummy Baseline Macro F1: {baseline_macro_f1:.4f}\n")
            if best_selection_score[0] <= baseline_macro_f1 + 0.01:
                f.write(
                    "Warning: the best model is not meaningfully better than the baseline. "
                    "Use predictions cautiously.\n"
                )

    result_df = pd.DataFrame(rows).sort_values(
        ["Baseline", "Macro F1", "Balanced Accuracy", "Accuracy"],
        ascending=[True, False, False, False],
    )
    result_df.to_csv(RESULT_DIR / "accuracy_comparison.csv", index=False, encoding="utf-8-sig")
    result_df.to_csv(RESULT_DIR / "model_metric_comparison.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    plt.bar(result_df["Model"], result_df["Accuracy"])
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "accuracy_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(result_df["Model"], result_df["Macro F1"])
    plt.title("Model Macro F1 Comparison")
    plt.xlabel("Model")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "model_macro_f1_comparison.png", dpi=200)
    plt.close()

    return result_df, best_model_name, fitted_models[best_model_name], X, y


def describe_value_against_dataset(
    label: str,
    value: float,
    overall_mean: float,
    high_stress_mean: float,
    higher_is_riskier: bool,
) -> str:
    if pd.isna(value):
        return f"- {label}: no value was entered."

    overall_gap = value - overall_mean
    if abs(overall_gap) < SIGNAL_THRESHOLD:
        comparison = f"similar to dataset average ({overall_mean:.2f})"
    else:
        direction = "higher" if overall_gap > 0 else "lower"
        comparison = f"{direction} than dataset average ({overall_mean:.2f})"

    high_gap = value - high_stress_mean

    if higher_is_riskier:
        if high_gap > SIGNAL_THRESHOLD:
            signal = "risk-increasing signal"
        elif high_gap < -SIGNAL_THRESHOLD:
            signal = "below the high-stress average"
        else:
            signal = "similar to the high-stress average"
    else:
        if high_gap < -SIGNAL_THRESHOLD:
            signal = "risk-increasing signal"
        elif high_gap > SIGNAL_THRESHOLD:
            signal = "above the high-stress average"
        else:
            signal = "similar to the high-stress average"

    return (
        f"- {label}: {value:.2f}, {comparison}; "
        f"gap vs high-stress avg={high_gap:+.2f}; {signal}."
    )


def analyze_single_profile(
    model: Pipeline,
    case_df: pd.DataFrame,
    training_df: pd.DataFrame,
    model_name: str,
    model_metrics: pd.Series | None = None,
    baseline_macro_f1: float | None = None,
) -> str:
    prediction = model.predict(case_df)[0]
    lines = [
        "Single Profile Analysis",
        "=" * 30,
        f"Model used: {model_name}",
        f"Predicted mental health risk level: {prediction}",
    ]

    if model_metrics is not None:
        macro_f1 = float(model_metrics["Macro F1"])
        balanced_acc = float(model_metrics["Balanced Accuracy"])
        lines.append(f"Model Macro F1 on test data: {macro_f1:.4f}")
        lines.append(f"Model Balanced Accuracy on test data: {balanced_acc:.4f}")
        if baseline_macro_f1 is not None and macro_f1 <= baseline_macro_f1 + 0.01:
            lines.append(
                "Caution: this model is close to the baseline, so this prediction should be interpreted carefully."
            )
        elif macro_f1 < 0.40:
            lines.append(
                "Caution: this model has weak class-balanced performance, so this prediction is only a rough estimate."
            )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(case_df)[0]
        proba_pairs = sorted(
            zip(model.classes_, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        lines.append("Prediction probabilities:")
        for label, prob in proba_pairs:
            lines.append(f"- {label}: {prob * 100:.1f}%")

    high_stress = training_df[training_df["stress_level"] >= 7]
    lines.extend(["", "Lifestyle comparison with the dataset:"])

    comparisons = [
        ("daily_social_media_hours", "Daily social media hours", True),
        ("sleep_hours", "Sleep hours", False),
        ("screen_time_before_sleep", "Screen time before sleep", True),
        ("physical_activity", "Physical activity", False),
        ("academic_performance", "Academic performance", False),
    ]

    for col, label, higher_is_riskier in comparisons:
        if col in case_df.columns and col in training_df.columns:
            value = pd.to_numeric(case_df.iloc[0][col], errors="coerce")
            lines.append(
                describe_value_against_dataset(
                    label=label,
                    value=value,
                    overall_mean=training_df[col].mean(),
                    high_stress_mean=high_stress[col].mean(),
                    higher_is_riskier=higher_is_riskier,
                )
            )

    lines.extend(
        [
            "",
            "Interpretation guide:",
            "- Higher social media hours and more screen time before sleep can be discussed as possible risk signals.",
            "- Lower sleep hours and lower physical activity can be discussed as lifestyle risk signals.",
            "- This output is an educational ML estimate, not a medical diagnosis.",
        ]
    )
    return "\n".join(lines)


def prompt_float(
    name: str,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    range_text = ""
    if min_value is not None and max_value is not None:
        range_text = f", range {min_value:.2f}-{max_value:.2f}"

    while True:
        raw = input(f"{name} [{default:.2f}{range_text}]: ").strip()
        if not raw:
            return float(default)
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue

        if min_value is not None and value < min_value:
            print(f"Please enter a value >= {min_value:.2f}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Please enter a value <= {max_value:.2f}.")
            continue
        return value


def prompt_category(name: str, choices: list[str], default: str) -> str:
    choices_text = ", ".join(choices)
    normalized_choices = {choice.lower(): choice for choice in choices}

    while True:
        raw = input(f"{name} ({choices_text}) [{default}]: ").strip()
        if not raw:
            return default
        if raw.lower() in normalized_choices:
            return normalized_choices[raw.lower()]
        print("Please choose one of the listed values.")


def collect_profile_from_console(X: pd.DataFrame) -> pd.DataFrame:
    print("\nEnter one teen profile. Press Enter to use the dataset median/mode defaults.")
    values = {}

    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    for col in numeric_features:
        values[col] = prompt_float(col, X[col].median(), X[col].min(), X[col].max())

    for col in categorical_features:
        choices = sorted(X[col].dropna().astype(str).unique().tolist())
        default = X[col].mode(dropna=True).iloc[0] if not X[col].mode(dropna=True).empty else choices[0]
        values[col] = prompt_category(col, choices, str(default))

    return pd.DataFrame([values], columns=X.columns)


def print_dataset_overview(df: pd.DataFrame) -> None:
    print("First 5 rows:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nDepression label used:", "depression_label" in df.columns)
    print("\nTarget distribution:")
    print(df["mental_health_risk_level"].value_counts())
    print("\nStress group distribution:")
    print(df["stress_group"].value_counts())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teen mental health risk analysis and ML comparison."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask for one person's lifestyle data and predict the risk level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df, existing_drop_cols = load_dataset()
    print_dataset_overview(df)

    missing_table = df.isnull().sum().sort_values(ascending=False)
    missing_table.to_csv(RESULT_DIR / "missing_values.csv", encoding="utf-8-sig")

    save_basic_visualizations(df)
    save_practical_analysis(df)

    result_df, best_model_name, best_model, X, _ = train_and_evaluate_models(df, existing_drop_cols)
    print("\nModel metrics:")
    print(result_df)
    print(f"\nBest model: {best_model_name}")

    if args.interactive:
        case_df = collect_profile_from_console(X)
        best_metrics = result_df[result_df["Model"] == best_model_name].iloc[0]
        baseline_rows = result_df[result_df["Model"] == BASELINE_MODEL_NAME]
        baseline_macro_f1 = None if baseline_rows.empty else float(baseline_rows.iloc[0]["Macro F1"])
        profile_report = analyze_single_profile(
            best_model,
            case_df,
            df,
            best_model_name,
            model_metrics=best_metrics,
            baseline_macro_f1=baseline_macro_f1,
        )
        print("\n" + profile_report)
        (RESULT_DIR / "user_prediction.txt").write_text(profile_report, encoding="utf-8")
        case_df.to_csv(RESULT_DIR / "user_input_profile.csv", index=False, encoding="utf-8-sig")

    print("\nDone. Results were saved in the results/ folder.")


if __name__ == "__main__":
    main()
