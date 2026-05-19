"""
Screen Addiction / Screen Dependency Self-Test App
--------------------------------------------------
Сэдэв: Оюутан өөрийн дэлгэцийн хамааралтай эсэх эрсдэлийг туршиж үзэх жижиг Python app.

Ажиллуулах:
    pip install -r requirements.txt
    python screen_addiction_app.py

CLI горим:
    python screen_addiction_app.py --cli

Анхааруулга: Энэ нь эмнэлзүйн онош биш. Зөвхөн сургалтын/өөрийгөө ажиглах зорилготой.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

DATA_FILE = Path(__file__).with_name("Teen_Mental_Health_Dataset.csv")
RESULT_DIR = Path(__file__).with_name("results")
RESULT_DIR.mkdir(exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

FEATURES = [
    "age",
    "gender",
    "daily_social_media_hours",
    "platform_usage",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "social_interaction_level",
    "stress_level",
    "anxiety_level",
]

NUMERIC_FEATURES = [
    "age",
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "stress_level",
    "anxiety_level",
]

CATEGORICAL_FEATURES = ["gender", "platform_usage", "social_interaction_level"]

MN_LABELS = {
    "Low": "Бага эрсдэл",
    "Medium": "Дунд эрсдэл",
    "High": "Өндөр эрсдэл",
}

EXPLANATIONS = {
    "Low": (
        "Одоогийн хэрэглээ харьцангуй хэвийн түвшинд байна. Гэхдээ унтахын өмнөх дэлгэц, "
        "сошиал медиа ашиглах цагийг тогтмол хянаж хэвших нь сайн."
    ),
    "Medium": (
        "Дэлгэцийн хэрэглээ өдөр тутмын амьдралд нөлөөлж эхэлж байж магадгүй. "
        "Унтахын өмнө 30-60 минут дэлгэц ашиглахгүй байх, өдөрт тодорхой цагийн хязгаар тавихыг зөвлөе."
    ),
    "High": (
        "Дэлгэцийн хамаарал өндөр байх эрсдэлтэй гэж таамаглагдлаа. "
        "Сошиал медиа ашиглах цагаа багасгах төлөвлөгөө гаргаж, шаардлагатай бол багш, эцэг эх, зөвлөхтэй ярилцах нь зүйтэй."
    ),
}


class DenseTransformer:
    """GaussianNB нь sparse matrix дээр ажиллахгүй тул dense болгож хувиргана."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else X


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def addiction_to_class(value: float) -> str:
    """Dataset-ийн addiction_level 1-10 оноог 3 ангилалд хөрвүүлнэ."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "Medium"
    if v <= 3:
        return "Low"
    if v <= 7:
        return "Medium"
    return "High"


def make_one_hot_encoder():
    """scikit-learn-ийн хуучин/шинэ хувилбаруудтай нийцүүлэх."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    accuracy: float
    macro_f1: float
    rows: int


def train_best_model(csv_path: Path = DATA_FILE) -> TrainedModel:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset олдсонгүй: {csv_path}\n"
            "Teen_Mental_Health_Dataset.csv файлыг screen_addiction_app.py-тэй нэг хавтаст байрлуулна уу."
        )

    df = clean_columns(pd.read_csv(csv_path)).drop_duplicates()
    missing = [c for c in FEATURES + ["addiction_level"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset-д дараах баганууд дутуу байна: {missing}")

    X = df[FEATURES].copy()
    y = df["addiction_level"].apply(addiction_to_class)

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    models = {
        "Naive Bayes": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("dense", DenseTransformer()),
                ("model", GaussianNB()),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ]
        ),
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
    }

    best: TrainedModel | None = None
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        mf1 = f1_score(y_test, pred, average="macro", zero_division=0)
        candidate = TrainedModel(name=name, pipeline=pipe, accuracy=acc, macro_f1=mf1, rows=len(df))
        if best is None or candidate.macro_f1 > best.macro_f1:
            best = candidate

    assert best is not None
    # Эцсийн app дээр бүх өгөгдлөөр дахин сургаж ашиглана.
    best.pipeline.fit(X, y)
    return best


def heuristic_score(user: Dict[str, object]) -> int:
    """Загварын таамаглалыг тайлбарлахад туслах 0-100 онооны энгийн дүрэм."""
    score = 0
    daily = float(user["daily_social_media_hours"])
    before_sleep = float(user["screen_time_before_sleep"])
    sleep = float(user["sleep_hours"])
    physical = float(user["physical_activity"])
    stress = float(user["stress_level"])
    anxiety = float(user["anxiety_level"])
    academic = float(user["academic_performance"])

    score += min(30, daily * 4)
    score += min(20, before_sleep * 6)
    score += max(0, (7 - sleep) * 4)
    score += max(0, (2 - physical) * 5)
    score += max(0, (stress - 5) * 3)
    score += max(0, (anxiety - 5) * 3)
    score += max(0, (3 - academic) * 4)

    return int(max(0, min(100, round(score))))


def label_from_score(score: int) -> str:
    """0-100 оноог энгийн эрсдэлийн ангилал руу хөрвүүлнэ."""
    if score <= 34:
        return "Low"
    if score <= 66:
        return "Medium"
    return "High"


def predict_user(model: TrainedModel, user: Dict[str, object]) -> Tuple[str, str, int, np.ndarray | None]:
    """ML таамаглал + өөрийгөө шалгах оноонд суурилсан эцсийн ангилал буцаана."""
    row = pd.DataFrame([user], columns=FEATURES)
    ml_pred = str(model.pipeline.predict(row)[0])
    score = heuristic_score(user)
    final_label = label_from_score(score)
    proba = None
    if hasattr(model.pipeline[-1], "predict_proba"):
        try:
            proba = model.pipeline.predict_proba(row)[0]
        except Exception:
            proba = None
    return final_label, ml_pred, score, proba


def make_recommendations(user: Dict[str, object], label: str) -> List[str]:
    recs: List[str] = []
    if float(user["daily_social_media_hours"]) >= 5:
        recs.append("Өдөрт сошиал медиа ашиглах цагаа 30 минутаар аажмаар багасгаж эхэл.")
    if float(user["screen_time_before_sleep"]) >= 1:
        recs.append("Унтахаас 30-60 минутын өмнө утас/компьютероо хол тавь.")
    if float(user["sleep_hours"]) < 7:
        recs.append("Унтах цагийг 7-9 цагт ойртуулах төлөвлөгөө гарга.")
    if float(user["physical_activity"]) < 1:
        recs.append("Өдөр бүр хамгийн багадаа 20-30 минут алхах эсвэл дасгал хийхийг зорь.")
    if float(user["stress_level"]) >= 7 or float(user["anxiety_level"]) >= 7:
        recs.append("Стресс/түгшүүр өндөр байгаа тул итгэдэг хүн эсвэл зөвлөхтэй ярилцах нь сайн.")
    if not recs:
        recs.append("Одоогийн дадлаа хадгалж, долоо хоногт нэг удаа screen time-аа шалгаж бай.")
    if label == "High":
        recs.append("Апп бүр дээр daily limit тавьж, notification-оо багасга.")
    return recs


def save_user_result(
    user: Dict[str, object],
    label: str,
    ml_label: str,
    score: int,
    recommendations: List[str],
    source: str,
) -> None:
    output = {
        **user,
        "final_risk_label": label,
        "ml_predicted_label": ml_label,
        "self_test_score": score,
        "source": source,
    }
    pd.DataFrame([output]).to_csv(
        RESULT_DIR / "screen_addiction_user_input.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "Screen Addiction Self-Test Result",
        "=" * 40,
        f"Source: {source}",
        f"Final risk label: {MN_LABELS.get(label, label)} ({label})",
        f"ML predicted label: {MN_LABELS.get(ml_label, ml_label)} ({ml_label})",
        f"Self-test score: {score}/100",
        "",
        "User input:",
    ]
    for key, value in user.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "Recommendations:"])
    lines.extend([f"- {rec}" for rec in recommendations])
    lines.extend(
        [
            "",
            "Warning: this is not a medical diagnosis. It is for learning/self-observation only.",
        ]
    )
    (RESULT_DIR / "screen_addiction_prediction.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_cli() -> None:
    model = train_best_model()
    print("\n=== Дэлгэцийн хамаарлын өөрийгөө шалгах тест ===")
    print(f"Сургагдсан загвар: {model.name} | Accuracy: {model.accuracy:.2f} | Macro F1: {model.macro_f1:.2f}")
    print("Анхааруулга: Энэ нь онош биш, зөвхөн сургалтын зорилготой.\n")

    def ask_float(prompt: str, lo: float, hi: float) -> float:
        while True:
            try:
                value = float(input(f"{prompt} ({lo}-{hi}): "))
                if lo <= value <= hi:
                    return value
            except ValueError:
                pass
            print("Зөв тоо оруулна уу.")

    def ask_choice(prompt: str, choices: List[str]) -> str:
        print(prompt)
        for i, c in enumerate(choices, start=1):
            print(f"  {i}. {c}")
        while True:
            try:
                idx = int(input("Сонголт: "))
                if 1 <= idx <= len(choices):
                    return choices[idx - 1]
            except ValueError:
                pass
            print("Зөв дугаар оруулна уу.")

    user = {
        "age": ask_float("Нас", 10, 25),
        "gender": ask_choice("Хүйс", ["male", "female", "other"]),
        "daily_social_media_hours": ask_float("Өдөрт сошиал медиа ашигладаг цаг", 0, 16),
        "platform_usage": ask_choice("Их ашигладаг платформ", ["Instagram", "TikTok", "Both", "Other"]),
        "sleep_hours": ask_float("Өдөрт унтдаг цаг", 0, 14),
        "screen_time_before_sleep": ask_float("Унтахын өмнө дэлгэц ашигладаг цаг", 0, 6),
        "academic_performance": ask_float("Сурлагын үнэлгээ /өөрийн үнэлгээ/", 0, 5),
        "physical_activity": ask_float("Өдөрт хөдөлгөөн/дасгал хийдэг цаг", 0, 6),
        "social_interaction_level": ask_choice("Бодит амьдрал дахь харилцааны түвшин", ["low", "medium", "high"]),
        "stress_level": ask_float("Стрессийн түвшин", 1, 10),
        "anxiety_level": ask_float("Түгшүүрийн түвшин", 1, 10),
    }

    label, ml_label, score, _ = predict_user(model, user)
    print("\n--- Үр дүн ---")
    print(f"Эцсийн эрсдэлийн ангилал: {MN_LABELS.get(label, label)}")
    print(f"ML загварын туслах таамаглал: {MN_LABELS.get(ml_label, ml_label)}")
    print(f"Өөрийгөө шалгах оноо: {score}/100")
    print(EXPLANATIONS.get(label, ""))
    print("\nЗөвлөмж:")
    recs = make_recommendations(user, label)
    for rec in recs:
        print(f"- {rec}")
    save_user_result(user, label, ml_label, score, recs, source="cli")
    print(f"\nSaved result: {RESULT_DIR / 'screen_addiction_prediction.txt'}")


def run_gui() -> None:
    import tkinter as tk
    from tkinter import ttk, messagebox

    model = train_best_model()

    root = tk.Tk()
    root.title("Дэлгэцийн хамаарлын өөрийгөө шалгах тест")
    root.geometry("760x760")

    main = ttk.Frame(root, padding=18)
    main.pack(fill="both", expand=True)

    title = ttk.Label(main, text="Дэлгэцийн хамаарлын өөрийгөө шалгах тест", font=("Arial", 18, "bold"))
    title.pack(anchor="w")

    subtitle = ttk.Label(
        main,
        text=(
            "Энэ app нь таны оруулсан мэдээлэл дээр үндэслэн Low/Medium/High эрсдэлийг таамаглана. "
            "Эмнэлзүйн онош биш."
        ),
        wraplength=700,
    )
    subtitle.pack(anchor="w", pady=(5, 12))

    info = ttk.Label(
        main,
        text=f"Dataset: {model.rows} мөр | Ашигласан загвар: {model.name} | Test accuracy: {model.accuracy:.2f}",
    )
    info.pack(anchor="w", pady=(0, 12))

    form = ttk.Frame(main)
    form.pack(fill="x")

    vars_: Dict[str, tk.Variable] = {
        "age": tk.DoubleVar(value=18),
        "gender": tk.StringVar(value="male"),
        "daily_social_media_hours": tk.DoubleVar(value=4),
        "platform_usage": tk.StringVar(value="Both"),
        "sleep_hours": tk.DoubleVar(value=7),
        "screen_time_before_sleep": tk.DoubleVar(value=1),
        "academic_performance": tk.DoubleVar(value=3.5),
        "physical_activity": tk.DoubleVar(value=1),
        "social_interaction_level": tk.StringVar(value="medium"),
        "stress_level": tk.DoubleVar(value=5),
        "anxiety_level": tk.DoubleVar(value=5),
    }

    def add_scale(row: int, key: str, label: str, from_: float, to: float, step_text: str = ""):
        ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=6)
        scale = ttk.Scale(form, from_=from_, to=to, variable=vars_[key], orient="horizontal")
        scale.grid(row=row, column=1, sticky="ew", padx=10)
        value_label = ttk.Label(form, width=10)
        value_label.grid(row=row, column=2, sticky="w")

        def update_label(*_):
            value_label.config(text=f"{float(vars_[key].get()):.1f} {step_text}")

        vars_[key].trace_add("write", update_label)
        update_label()

    def add_combo(row: int, key: str, label: str, values: List[str]):
        ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=6)
        combo = ttk.Combobox(form, textvariable=vars_[key], values=values, state="readonly")
        combo.grid(row=row, column=1, sticky="ew", padx=10)

    form.columnconfigure(1, weight=1)
    add_scale(0, "age", "Нас", 10, 25)
    add_combo(1, "gender", "Хүйс", ["male", "female", "other"])
    add_scale(2, "daily_social_media_hours", "Өдөрт сошиал медиа ашигладаг цаг", 0, 16, "цаг")
    add_combo(3, "platform_usage", "Их ашигладаг платформ", ["Instagram", "TikTok", "Both", "Other"])
    add_scale(4, "sleep_hours", "Өдөрт унтдаг цаг", 0, 14, "цаг")
    add_scale(5, "screen_time_before_sleep", "Унтахын өмнө дэлгэц ашигладаг цаг", 0, 6, "цаг")
    add_scale(6, "academic_performance", "Сурлагын үнэлгээ /0-5/", 0, 5)
    add_scale(7, "physical_activity", "Өдөрт хөдөлгөөн/дасгал хийдэг цаг", 0, 6, "цаг")
    add_combo(8, "social_interaction_level", "Бодит харилцааны түвшин", ["low", "medium", "high"])
    add_scale(9, "stress_level", "Стрессийн түвшин /1-10/", 1, 10)
    add_scale(10, "anxiety_level", "Түгшүүрийн түвшин /1-10/", 1, 10)

    result_box = tk.Text(main, height=12, wrap="word", font=("Arial", 11))
    result_box.pack(fill="both", expand=True, pady=(16, 0))

    def calculate():
        try:
            user = {k: v.get() for k, v in vars_.items()}
            label, ml_label, score, _ = predict_user(model, user)
            recs = make_recommendations(user, label)
            text = (
                f"ҮР ДҮН\n"
                f"Эцсийн эрсдэлийн ангилал: {MN_LABELS.get(label, label)}\n"
                f"ML загварын туслах таамаглал: {MN_LABELS.get(ml_label, ml_label)}\n"
                f"Өөрийгөө шалгах оноо: {score}/100\n\n"
                f"Тайлбар: {EXPLANATIONS.get(label, '')}\n\n"
                f"Зөвлөмж:\n" + "\n".join([f"• {r}" for r in recs]) +
                "\n\nАнхааруулга: Энэ нь эмнэлзүйн онош биш, зөвхөн сургалтын/өөрийгөө ажиглах зорилготой."
            )
            result_box.delete("1.0", "end")
            result_box.insert("1.0", text)
            save_user_result(user, label, ml_label, score, recs, source="gui")
        except Exception as exc:
            messagebox.showerror("Алдаа", str(exc))

    ttk.Button(main, text="Үр дүн харах", command=calculate).pack(anchor="e", pady=12)
    calculate()
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen addiction self-test app")
    parser.add_argument("--cli", action="store_true", help="GUI биш terminal горимоор ажиллуулах")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        # Display байхгүй Linux орчинд GUI алдаа гарвал CLI руу автоматаар шилжинэ.
        if os.name != "nt" and not os.environ.get("DISPLAY"):
            run_cli()
        else:
            run_gui()


if __name__ == "__main__":
    main()
