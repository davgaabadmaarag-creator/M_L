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
RISK_LABELS_MN = {
    "Low": "Бага",
    "Medium": "Дунд",
    "High": "Өндөр",
}
FEATURE_LABELS_MN = {
    "age": "нас",
    "gender": "хүйс",
    "daily_social_media_hours": "өдөрт сошиал медиа ашигласан цаг",
    "platform_usage": "ашигладаг платформ",
    "sleep_hours": "өдөрт унтдаг цаг",
    "screen_time_before_sleep": "унтахын өмнөх дэлгэцийн цаг",
    "academic_performance": "сурлагын үзүүлэлт",
    "physical_activity": "биеийн хөдөлгөөн",
    "social_interaction_level": "нийгмийн харилцааны түвшин",
    "stress_level": "стрессийн түвшин",
    "anxiety_level": "түгшүүрийн түвшин",
    "addiction_level": "донтох хандлагын түвшин",
    "mental_health_risk_score": "сэтгэцийн эрүүл мэндийн эрсдэлийн оноо",
    "mental_health_risk_level": "сэтгэцийн эрүүл мэндийн эрсдэлийн ангилал",
    "stress_group": "стрессийн бүлэг",
}
PROFILE_FEATURES = [
    # Практик тайлбар дээр target-аас шууд үүссэн багана оруулахгүй.
    # Ингэснээр "эрсдэл өндөр учраас эрсдэл өндөр" гэсэн давхар тайлбар гарахаас сэргийлнэ.
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
]


def clean_col(name: str) -> str:
    """Баганын нэрийг Python-д ашиглахад хялбар snake_case хэлбэрт оруулна."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def feature_label(col: str) -> str:
    """Баганын нэрийг тайлан дээр уншихад ойлгомжтой Монгол нэр болгоно."""
    return FEATURE_LABELS_MN.get(col, col)


def risk_label(label: str) -> str:
    """Low/Medium/High ангиллыг Монгол тайлбартай харуулна."""
    return f"{RISK_LABELS_MN.get(label, label)} ({label})"


def create_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """Stress, anxiety, addiction онооноос Low/Medium/High target үүсгэнэ."""
    missing = [col for col in LEVEL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Target үүсгэхэд хэрэгтэй багана олдсонгүй: {missing}")

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
    """Практик тайлбарт ашиглах стрессийн бүлэг нэмнэ."""
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
    """scikit-learn-ийн хуучин/шинэ хувилбарын OneHotEncoder-ийг дэмжинэ."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} олдсонгүй. Dataset-ээ stat.py файлтай нэг хавтаст байрлуулна уу."
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
    """Тайланд ашиглах үндсэн EDA графикуудыг хадгална."""
    for col in LEVEL_COLUMNS:
        if col in df.columns:
            plt.figure(figsize=(7, 5))
            df[col].value_counts().sort_index().plot(kind="bar")
            plt.title(f"{feature_label(col)}-ийн тархалт")
            plt.xlabel("Түвшин (1-10)")
            plt.ylabel("Тоо")
            plt.tight_layout()
            plt.savefig(RESULT_DIR / f"{col}_distribution.png", dpi=200)
            plt.close()

    plt.figure(figsize=(7, 5))
    df["mental_health_risk_level"].value_counts().reindex(RISK_ORDER).plot(kind="bar")
    plt.title("Сэтгэцийн эрүүл мэндийн эрсдэлийн ангиллын тархалт")
    plt.xlabel("Эрсдэлийн ангилал")
    plt.ylabel("Тоо")
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
        plt.title("Платформ бүрийн дундаж эрсдэлийн оноо")
        plt.xlabel("Платформ")
        plt.ylabel("Дундаж эрсдэлийн оноо")
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
        plt.title("Корреляцийн матриц")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "correlation_matrix.png", dpi=200)
        plt.close()


def available_profile_features(df: pd.DataFrame) -> list[str]:
    return [col for col in PROFILE_FEATURES if col in df.columns]


def save_practical_analysis(df: pd.DataFrame) -> None:
    """Стресс өндөр хүмүүсийн хэрэглээний хэв маяг зэрэг практик тайлан үүсгэнэ."""
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
        plt.title("Платформ бүрийн өндөр стрессийн хувь")
        plt.xlabel("Платформ")
        plt.ylabel("Өндөр стрессийн хувь (%)")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "high_stress_by_platform.png", dpi=200)
        plt.close()

    trend = df.groupby("stress_level")["daily_social_media_hours"].mean()
    trend.to_csv(RESULT_DIR / "social_media_by_stress_level.csv", encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    trend.plot(kind="line", marker="o")
    plt.title("Стрессийн түвшин ба сошиал медиа хэрэглээ")
    plt.xlabel("Стрессийн түвшин")
    plt.ylabel("Өдөрт ашигласан дундаж цаг")
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
        plt.title("Эрсдэлийн ангилал бүрийн амьдралын хэв маяг")
        plt.xlabel("Эрсдэлийн ангилал")
        plt.ylabel("Дундаж утга")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "risk_level_lifestyle_profile.png", dpi=200)
        plt.close()

    lines = [
        "Практик шинжилгээний тайлан",
        "=" * 60,
        f"Цэвэрлэсний дараах нийт мөрийн тоо: {len(df)}",
        f"Стресс өндөр мөрүүд (stress_level >= 7): {len(high_stress)} "
        f"({len(high_stress) / len(df) * 100:.1f}%)",
        "",
        "1. Стресс өндөр бүлгийн дундаж үзүүлэлт",
        "-" * 30,
    ]

    for col in profile_cols:
        high_mean = high_stress[col].mean()
        other_mean = not_high_stress[col].mean()
        diff = high_mean - other_mean
        lines.append(
            f"{feature_label(col)}: стресс өндөр дундаж={high_mean:.2f}, "
            f"бусад бүлгийн дундаж={other_mean:.2f}, зөрүү={diff:+.2f}"
        )

    if "platform_usage" in df.columns:
        lines.extend(["", "2. Стресс өндөр хүмүүсийн хамгийн түгээмэл платформ", "-" * 48])
        platform_share = high_stress["platform_usage"].value_counts(normalize=True).mul(100)
        for platform, pct in platform_share.items():
            lines.append(f"{platform}: {pct:.1f}%")

        lines.extend(["", "3. Платформ бүрийн өндөр стрессийн хувь", "-" * 36])
        for platform, row in platform_summary.iterrows():
            lines.append(
                f"{platform}: өндөр стрессийн хувь={row['high_stress_rate_pct']:.1f}%, "
                f"сошиал хэрэглээний дундаж цаг={row['avg_social_media_hours']:.2f}, "
                f"унтах дундаж цаг={row['avg_sleep_hours']:.2f}"
            )

    lines.extend(
        [
            "",
            "4. Энэ үр дүнг хэрхэн тайлбарлах вэ?",
            "-" * 30,
            "Энэ хэсгийг хамгаалалт/тайланд дараах байдлаар тайлбарлаж болно:",
            "- Стресс өндөр бүлгийг бусад бүлэгтэй сошиал медиа хэрэглээний цагаар харьцуулж болно.",
            "- Платформын хүснэгт нь энэ dataset дээр аль платформд өндөр стрессийн хувь арай их байгааг харуулна.",
            "- Унтах цаг, унтахын өмнөх дэлгэцийн цаг, биеийн хөдөлгөөн нь амьдралын хэв маягийн ялгааг тайлбарлахад тусална.",
            "",
            "Анхаарах зүйл: Энэ нь сургалтын зориулалттай өгөгдлийн шинжилгээ бөгөөд эмнэлзүйн онош биш.",
        ]
    )

    (RESULT_DIR / "practical_insights.txt").write_text("\n".join(lines), encoding="utf-8")


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    target_col = "mental_health_risk_level"
    # Target leakage-ээс сэргийлж target үүсгэхэд орсон stress/anxiety/addiction багануудыг input-оос хасна.
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
    """Decision Tree загварын аль feature чухал байсныг хадгална."""
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
        plt.title("Decision Tree загварын хамгийн чухал feature-үүд")
        plt.xlabel("Чухлын хэмжээ")
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
        f.write("Өсвөр үеийнхний сэтгэцийн эрүүл мэндийн ML төслийн үр дүн\n")
        f.write("depression_label баганыг ашиглаагүй, бүрэн хассан.\n")
        f.write("=" * 55 + "\n")
        f.write(f"Боловсруулалтын дараах dataset-ийн хэмжээ: {df.shape}\n")
        f.write(f"Хассан баганууд: {existing_drop_cols}\n")
        f.write("Target багана: mental_health_risk_level\n")
        f.write("Target үүсгэх арга: mean(stress_level, anxiety_level, addiction_level) => Low/Medium/High\n")
        f.write(f"Model-д орсон тоон feature-үүд: {[feature_label(c) for c in numeric_features]}\n")
        f.write(f"Model-д орсон категори feature-үүд: {[feature_label(c) for c in categorical_features]}\n\n")
        f.write("Загвар сонгох гол metric: Macro F1. Тэнцсэн үед Balanced Accuracy, Accuracy-г харна.\n")
        f.write("Dummy Baseline нь хамгийн олон давтагдсан ангиллыг таамагладаг бөгөөд зөвхөн харьцуулалтын суурь юм.\n\n")
        f.write("Target-ийн тархалт:\n")
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
            f.write("Ангиллын дэлгэрэнгүй тайлан:\n")
            f.write(classification_report(y_test, pred, labels=RISK_ORDER, zero_division=0))
            f.write("\nConfusion matrix буюу андуурлын матриц:\n")
            f.write(str(confusion_matrix(y_test, pred, labels=RISK_ORDER)))
            f.write("\n")

            # Accuracy дангаараа буруу ойлголт өгч болох тул Macro F1-ийг эхэлж харна.
            selection_score = (macro_f1, balanced_acc, acc)
            if not is_baseline and selection_score > best_selection_score:
                best_selection_score = selection_score
                best_model_name = name

        f.write(
            f"\nХамгийн сайн non-baseline загвар: {best_model_name}, "
            f"Macro F1: {best_selection_score[0]:.4f}, "
            f"Balanced Accuracy: {best_selection_score[1]:.4f}, "
            f"Accuracy: {best_selection_score[2]:.4f}\n"
        )
        if baseline_macro_f1 is not None:
            f.write(f"Dummy Baseline Macro F1: {baseline_macro_f1:.4f}\n")
            if best_selection_score[0] <= baseline_macro_f1 + 0.01:
                f.write(
                    "Анхааруулга: хамгийн сайн загвар baseline-аас мэдэгдэхүйц дээр биш байна. "
                    "Иймээс prediction-ийг болгоомжтой тайлбарлана.\n"
                )

    result_df = pd.DataFrame(rows).sort_values(
        ["Baseline", "Macro F1", "Balanced Accuracy", "Accuracy"],
        ascending=[True, False, False, False],
    )
    result_df.to_csv(RESULT_DIR / "accuracy_comparison.csv", index=False, encoding="utf-8-sig")
    result_df.to_csv(RESULT_DIR / "model_metric_comparison.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    plt.bar(result_df["Model"], result_df["Accuracy"])
    plt.title("Загваруудын Accuracy харьцуулалт")
    plt.xlabel("Загвар")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "accuracy_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(result_df["Model"], result_df["Macro F1"])
    plt.title("Загваруудын Macro F1 харьцуулалт")
    plt.xlabel("Загвар")
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
        return f"- {label}: утга оруулаагүй байна."

    overall_gap = value - overall_mean
    if abs(overall_gap) < SIGNAL_THRESHOLD:
        comparison = f"dataset-ийн дундажтай ойролцоо ({overall_mean:.2f})"
    else:
        direction = "өндөр" if overall_gap > 0 else "бага"
        comparison = f"dataset-ийн дунджаас {direction} ({overall_mean:.2f})"

    high_gap = value - high_stress_mean

    if higher_is_riskier:
        if high_gap > SIGNAL_THRESHOLD:
            signal = "эрсдэлийг нэмэгдүүлж болзошгүй дохио"
        elif high_gap < -SIGNAL_THRESHOLD:
            signal = "стресс өндөр бүлгийн дунджаас бага"
        else:
            signal = "стресс өндөр бүлгийн дундажтай ойролцоо"
    else:
        if high_gap < -SIGNAL_THRESHOLD:
            signal = "эрсдэлийг нэмэгдүүлж болзошгүй дохио"
        elif high_gap > SIGNAL_THRESHOLD:
            signal = "стресс өндөр бүлгийн дунджаас өндөр"
        else:
            signal = "стресс өндөр бүлгийн дундажтай ойролцоо"

    return (
        f"- {label}: {value:.2f}, {comparison}; "
        f"стресс өндөр бүлгийн дундажтай зөрүү={high_gap:+.2f}; {signal}."
    )


def score_single_profile(case_df: pd.DataFrame) -> tuple[int, str]:
    """Zip app шиг 0-100 оноо тооцож, Low/Medium/High ангилал буцаана."""
    row = case_df.iloc[0]

    def value(col: str, default: float) -> float:
        parsed = pd.to_numeric(row.get(col, default), errors="coerce")
        return float(default) if pd.isna(parsed) else float(parsed)

    daily = value("daily_social_media_hours", 4.5)
    sleep = value("sleep_hours", 6.5)
    before_sleep = value("screen_time_before_sleep", 1.7)
    academic = value("academic_performance", 3.0)
    physical = value("physical_activity", 1.0)
    social = str(row.get("social_interaction_level", "medium")).lower()

    score = 0.0
    score += min(30, max(0, daily - 1) / 7 * 30)
    score += min(20, max(0, before_sleep - 0.5) / 2.5 * 20)
    score += min(20, max(0, 9 - sleep) / 5 * 20)
    score += min(15, max(0, 2 - physical) / 2 * 15)
    score += min(10, max(0, 4 - academic) / 2 * 10)
    if social == "low":
        score += 5
    elif social == "medium":
        score += 2

    score_int = int(max(0, min(100, round(score))))
    if score_int <= 34:
        return score_int, "Low"
    if score_int <= 66:
        return score_int, "Medium"
    return score_int, "High"


def more_severe_label(first: str, second: str) -> str:
    severity = {"Low": 0, "Medium": 1, "High": 2}
    return first if severity.get(first, 0) >= severity.get(second, 0) else second


def profile_explanation(label: str) -> str:
    if label == "Low":
        return (
            "Одоогийн оруулсан үзүүлэлтүүд харьцангуй хэвийн түвшинд байна. "
            "Гэхдээ унтахын өмнөх дэлгэцийн хэрэглээ, сошиал медиа ашиглах цагаа тогтмол хянаж хэвших нь сайн."
        )
    if label == "Medium":
        return (
            "Дэлгэц болон сошиал медиа хэрэглээ өдөр тутмын амьдралд нөлөөлж эхэлж байж магадгүй. "
            "Унтахын өмнөх дэлгэцийн хэрэглээг багасгаж, өдөрт тогтмол хязгаар тавихыг зөвлөе."
        )
    return (
        "Эрсдэл өндөр байх магадлалтай гэж таамаглагдлаа. "
        "Сошиал медиа ашиглах цагаа багасгах төлөвлөгөө гаргаж, шаардлагатай бол багш, эцэг эх, зөвлөхтэй ярилцах нь зүйтэй."
    )


def make_single_profile_recommendations(case_df: pd.DataFrame, final_label: str) -> list[str]:
    row = case_df.iloc[0]

    def value(col: str, default: float) -> float:
        parsed = pd.to_numeric(row.get(col, default), errors="coerce")
        return float(default) if pd.isna(parsed) else float(parsed)

    recs = []
    if value("daily_social_media_hours", 0) >= 5:
        recs.append("Өдөрт сошиал медиа ашиглах цагаа 30 минутаар аажмаар багасгаж эхэл.")
    if value("screen_time_before_sleep", 0) >= 1:
        recs.append("Унтахаас 30-60 минутын өмнө утас/компьютероо хол тавь.")
    if value("sleep_hours", 9) < 7:
        recs.append("Унтах цагийг 7-9 цагт ойртуулах төлөвлөгөө гарга.")
    if value("physical_activity", 2) < 1:
        recs.append("Өдөр бүр хамгийн багадаа 20-30 минут алхах эсвэл дасгал хийхийг зорь.")
    if value("academic_performance", 4) < 2.8:
        recs.append("Хичээл/даалгаврын цагийг богино блок болгон төлөвлөж, дэлгэцийн завсарлага тогтоож үз.")
    if str(row.get("social_interaction_level", "medium")).lower() == "low":
        recs.append("Өдөр бүр нэг бодит харилцаа нэмэхийг зорь: найз, гэр бүл, багштай богино яриа хийх гэх мэт.")
    if final_label == "High":
        recs.append("Апп бүр дээр daily limit тавьж, notification-оо багасга.")
    if not recs:
        recs.append("Одоогийн дадлаа хадгалж, долоо хоногт нэг удаа screen time-аа шалгаж бай.")
    return recs


def format_zip_style_profile_report(
    model: Pipeline,
    case_df: pd.DataFrame,
    training_df: pd.DataFrame,
    model_name: str,
    model_metrics: pd.Series | None = None,
    baseline_macro_f1: float | None = None,
) -> str:
    ml_prediction = str(model.predict(case_df)[0])
    score, score_label = score_single_profile(case_df)
    final_label = more_severe_label(ml_prediction, score_label)
    recs = make_single_profile_recommendations(case_df, final_label)

    lines = [
        "=== Сэтгэцийн эрүүл мэндийн эрсдэлийн өөрийгөө шалгах тест ===",
        "",
        f"Сургагдсан загвар: {model_name}",
    ]

    if model_metrics is not None:
        macro_f1 = float(model_metrics["Macro F1"])
        balanced_acc = float(model_metrics["Balanced Accuracy"])
        lines.append(f"Model Macro F1: {macro_f1:.2f} | Balanced Accuracy: {balanced_acc:.2f}")

    lines.extend(
        [
            "Анхааруулга: Энэ нь онош биш, зөвхөн сургалтын/өөрийгөө ажиглах зорилготой.",
            "",
            "--- Үр дүн ---",
            f"Эцсийн эрсдэлийн ангилал: {risk_label(final_label)}",
            f"ML загварын таамаглал: {risk_label(ml_prediction)}",
            f"Өөрийгөө шалгах оноо: {score}/100 ({risk_label(score_label)})",
        ]
    )

    if baseline_macro_f1 is not None and model_metrics is not None:
        macro_f1 = float(model_metrics["Macro F1"])
        if macro_f1 <= baseline_macro_f1 + 0.01:
            lines.append("Санамж: энэ model baseline-тай ойролцоо тул таамаглалыг болгоомжтой тайлбарлана.")

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(case_df)[0]
            proba_pairs = sorted(zip(model.classes_, probabilities), key=lambda item: item[1], reverse=True)
            lines.append("")
            lines.append("Ангилал тус бүрийн магадлал:")
            for label, prob in proba_pairs:
                lines.append(f"- {risk_label(str(label))}: {prob * 100:.1f}%")
        except Exception:
            pass

    lines.extend(["", "Тайлбар:", profile_explanation(final_label), "", "Зөвлөмж:"])
    lines.extend([f"- {rec}" for rec in recs])

    lines.extend(["", "Оруулсан мэдээлэл:"])
    for col in case_df.columns:
        lines.append(f"- {feature_label(col)}: {case_df.iloc[0][col]}")

    lines.extend(
        [
            "",
            "Үр дүнг тайлбарлахдаа:",
            "- Сошиал медиа хэрэглээ болон унтахын өмнөх дэлгэцийн цаг өндөр байх нь эрсдэлийн дохио байж болно.",
            "- Унтах цаг, хөдөлгөөн, бодит харилцаа бага байвал амьдралын хэв маягийн эрсдэл нэмэгдэж болно.",
            "- Энэ үр дүн нь сургалтын зориулалттай ML үнэлгээ бөгөөд эмнэлзүйн онош биш.",
        ]
    )
    return "\n".join(lines)


def analyze_single_profile(
    model: Pipeline,
    case_df: pd.DataFrame,
    training_df: pd.DataFrame,
    model_name: str,
    model_metrics: pd.Series | None = None,
    baseline_macro_f1: float | None = None,
) -> str:
    return format_zip_style_profile_report(
        model=model,
        case_df=case_df,
        training_df=training_df,
        model_name=model_name,
        model_metrics=model_metrics,
        baseline_macro_f1=baseline_macro_f1,
    )

    prediction = model.predict(case_df)[0]
    lines = [
        "Нэг хүний мэдээлэл дээр хийсэн шинжилгээ",
        "=" * 30,
        f"Ашигласан загвар: {model_name}",
        f"Таамагласан эрсдэлийн ангилал: {risk_label(prediction)}",
    ]

    if model_metrics is not None:
        macro_f1 = float(model_metrics["Macro F1"])
        balanced_acc = float(model_metrics["Balanced Accuracy"])
        lines.append(f"Test өгөгдөл дээрх Macro F1: {macro_f1:.4f}")
        lines.append(f"Test өгөгдөл дээрх Balanced Accuracy: {balanced_acc:.4f}")
        if baseline_macro_f1 is not None and macro_f1 <= baseline_macro_f1 + 0.01:
            lines.append(
                "Анхааруулга: энэ загвар baseline-тай ойролцоо тул prediction-ийг болгоомжтой тайлбарлана."
            )
        elif macro_f1 < 0.40:
            lines.append(
                "Анхааруулга: class бүрийг тэнцвэртэй таних чадвар сул тул энэ prediction нь зөвхөн ойролцоолсон үнэлгээ."
            )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(case_df)[0]
        proba_pairs = sorted(
            zip(model.classes_, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        lines.append("Ангилал тус бүрийн магадлал:")
        for label, prob in proba_pairs:
            lines.append(f"- {risk_label(label)}: {prob * 100:.1f}%")

    high_stress = training_df[training_df["stress_level"] >= 7]
    lines.extend(["", "Оруулсан мэдээллийг dataset-ийн дундажтай харьцуулсан нь:"])

    comparisons = [
        ("daily_social_media_hours", "Өдөрт сошиал медиа ашигласан цаг", True),
        ("sleep_hours", "Унтах цаг", False),
        ("screen_time_before_sleep", "Унтахын өмнөх дэлгэцийн цаг", True),
        ("physical_activity", "Биеийн хөдөлгөөн", False),
        ("academic_performance", "Сурлагын үзүүлэлт", False),
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
            "Тайлбарлах заавар:",
            "- Сошиал медиа хэрэглээ болон унтахын өмнөх дэлгэцийн цаг өндөр байвал эрсдэлийн дохио байж болно.",
            "- Унтах цаг болон биеийн хөдөлгөөн бага байвал амьдралын хэв маягийн эрсдэлийн дохио байж болно.",
            "- Энэ үр дүн нь сургалтын зориулалттай ML үнэлгээ бөгөөд эмнэлзүйн онош биш.",
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
            print("Тоон утга оруулна уу.")
            continue

        if min_value is not None and value < min_value:
            print(f"{min_value:.2f}-аас их буюу тэнцүү утга оруулна уу.")
            continue
        if max_value is not None and value > max_value:
            print(f"{max_value:.2f}-аас бага буюу тэнцүү утга оруулна уу.")
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
        print("Жагсаалтад байгаа утгуудаас сонгоно уу.")


def collect_profile_from_console(X: pd.DataFrame) -> pd.DataFrame:
    print("\nНэг хүний мэдээлэл оруулна уу. Enter дарвал dataset-ийн median/mode default утгыг ашиглана.")
    values = {}

    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    for col in numeric_features:
        values[col] = prompt_float(feature_label(col), X[col].median(), X[col].min(), X[col].max())

    for col in categorical_features:
        choices = sorted(X[col].dropna().astype(str).unique().tolist())
        default = X[col].mode(dropna=True).iloc[0] if not X[col].mode(dropna=True).empty else choices[0]
        values[col] = prompt_category(feature_label(col), choices, str(default))

    return pd.DataFrame([values], columns=X.columns)


def print_dataset_overview(df: pd.DataFrame) -> None:
    print("Эхний 5 мөр:")
    print(df.head())
    print("\nDataset-ийн хэмжээ:", df.shape)
    print("\nБаганын нэрс:", list(df.columns))
    print("\ndepression_label ашигласан эсэх:", "depression_label" in df.columns)
    print("\nTarget-ийн тархалт:")
    print(df["mental_health_risk_level"].value_counts())
    print("\nСтрессийн бүлгийн тархалт:")
    print(df["stress_group"].value_counts())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Өсвөр үеийнхний сэтгэцийн эрүүл мэндийн эрсдэлийн ML шинжилгээ."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Нэг хүний амьдралын хэв маягийн мэдээллийг асууж, эрсдэлийн ангиллыг таамаглана.",
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
    print("\nЗагваруудын metric:")
    print(result_df)
    print(f"\nХамгийн сайн загвар: {best_model_name}")

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

    print("\nДууслаа. Үр дүн results/ хавтаст хадгалагдлаа.")


if __name__ == "__main__":
    main()
