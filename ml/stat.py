import os
import re
import sys
import warnings
from pathlib import Path
 
RESULT_DIR = Path("results")
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
 
warnings.filterwarnings("ignore")
 
DATA_FILE = "Teen_Mental_Health_Dataset.csv"
DROP_COLUMNS = ["depression_label"]
LEVEL_COLUMNS = ["stress_level", "anxiety_level", "addiction_level"]
 
 
def clean_col(name: str) -> str:
    """Баганын нэрийг Python-д ашиглахад хялбар snake_case хэлбэрт оруулна."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")
 
 
def create_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """Stress, anxiety, addiction 1-10 оноонд үндэслэн шинэ target үүсгэнэ.
 
    mental_health_risk_score = stress/anxiety/addiction-ийн дундаж оноо
    1.0 - 3.99  => Low
    4.0 - 6.99  => Medium
    7.0 - 10.0  => High
    """
    missing = [col for col in LEVEL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Target үүсгэхэд хэрэгтэй багана олдсонгүй: {missing}")
 
    df = df.copy()
    df["mental_health_risk_score"] = df[LEVEL_COLUMNS].mean(axis=1)
    df["mental_health_risk_level"] = pd.cut(
        df["mental_health_risk_score"],
        bins=[0, 4, 7, 10],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
        right=False,
    )
 
    # score яг 10 гарсан мөрийг High болгох
    df.loc[df["mental_health_risk_score"] == 10, "mental_health_risk_level"] = "High"
    df["mental_health_risk_level"] = df["mental_health_risk_level"].astype(str)
    return df
 
 
def save_basic_visualizations(df: pd.DataFrame) -> None:
    """Өгөгдлийн үндсэн зураглалуудыг хадгална."""
    # 1) 1-10 level багануудын тархалт
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
 
    # 2) Шинээр үүсгэсэн target-ийн тархалт
    plt.figure(figsize=(7, 5))
    df["mental_health_risk_level"].value_counts().reindex(["Low", "Medium", "High"]).plot(kind="bar")
    plt.title("Mental Health Risk Level Distribution")
    plt.xlabel("Risk Level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "mental_health_risk_distribution.png", dpi=200)
    plt.close()
 
    # 3) Platform бүрийн дундаж risk score
    if "platform_usage" in df.columns:
        platform_avg = df.groupby("platform_usage")["mental_health_risk_score"].mean().sort_values(ascending=False)
        platform_avg.to_csv(RESULT_DIR / "average_risk_by_platform.csv", encoding="utf-8-sig")
 
        plt.figure(figsize=(7, 5))
        platform_avg.plot(kind="bar")
        plt.title("Average Risk Score by Platform")
        plt.xlabel("Platform")
        plt.ylabel("Average Risk Score")
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "average_risk_by_platform.png", dpi=200)
        plt.close()
 
    # 4) Correlation matrix
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
 
 
def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} олдсонгүй. Dataset-ээ энэ кодтой нэг хавтаст байрлуулна уу."
        )
 
    # 1. Өгөгдөл унших
    df = pd.read_csv(DATA_FILE)
    df.columns = [clean_col(c) for c in df.columns]
    df = df.drop_duplicates()
 
    # 2. depression_label-ийг бүрэн хасах
    existing_drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=existing_drop_cols)
 
    # 3. Stress/anxiety/addiction дээр үндэслэн target үүсгэх
    df = create_risk_level(df)
 
    print("Эхний 5 мөр:")
    print(df.head())
    print("\nӨгөгдлийн хэмжээ:", df.shape)
    print("\nБаганын нэрс:", list(df.columns))
    print("\nDepression label ашигласан эсэх:", "depression_label" in df.columns)
    print("\nTarget тархалт:")
    print(df["mental_health_risk_level"].value_counts())
 
    # 4. Missing values хадгалах
    missing_table = df.isnull().sum().sort_values(ascending=False)
    missing_table.to_csv(RESULT_DIR / "missing_values.csv", encoding="utf-8-sig")
 
    # 5. Visualization хадгалах
    save_basic_visualizations(df)
 
    # 6. ML target ба feature сонгох
    target_col = "mental_health_risk_level"
 
    # Leakage-ээс сэргийлж target үүсгэсэн багануудыг input-оос хасна.
    # Өөрөөр хэлбэл model нь social media/sleep/activity зэрэг lifestyle feature-үүдээр
    # Low/Medium/High risk level-ийг таамаглана.
    feature_drop_cols = [target_col, "mental_health_risk_score"] + LEVEL_COLUMNS
    X = df.drop(columns=[col for col in feature_drop_cols if col in df.columns])
    y = df[target_col]
 
    valid_idx = y.notna() & (y.astype(str).str.lower() != "nan")
    X = X.loc[valid_idx].copy()
    y = y.loc[valid_idx].copy()
 
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
 
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
 
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
 
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
 
    class DenseTransformer:
        def fit(self, X, y=None):
            return self
 
        def transform(self, X):
            return X.toarray() if hasattr(X, "toarray") else X
 
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
 
    models = {
        "Naive Bayes": Pipeline(steps=[
            ("preprocess", preprocessor),
            ("dense", DenseTransformer()),
            ("model", GaussianNB()),
        ]),
        "Decision Tree": Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=4, random_state=42)),
        ]),
        "Logistic Regression": Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }
 
    rows = []
    best_model_name = None
    best_accuracy = -1
 
    with open(RESULT_DIR / "model_results.txt", "w", encoding="utf-8") as f:
        f.write("Teen Mental Health ML Project Results\n")
        f.write("Depression label was removed and not used.\n")
        f.write("=" * 55 + "\n")
        f.write(f"Dataset shape after processing: {df.shape}\n")
        f.write(f"Removed columns: {existing_drop_cols}\n")
        f.write(f"Target column: {target_col}\n")
        f.write("Target creation: mean(stress_level, anxiety_level, addiction_level) => Low/Medium/High\n")
        f.write(f"Input numeric features: {numeric_features}\n")
        f.write(f"Input categorical features: {categorical_features}\n\n")
        f.write("Target distribution:\n")
        f.write(str(y.value_counts()))
        f.write("\n\n")
 
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            rows.append({"Model": name, "Accuracy": round(acc, 4)})
 
            f.write(f"\n{name}\n")
            f.write("-" * len(name) + "\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write("Classification report:\n")
            f.write(classification_report(y_test, pred, zero_division=0))
            f.write("\nConfusion matrix:\n")
            f.write(str(confusion_matrix(y_test, pred, labels=["Low", "Medium", "High"])))
            f.write("\n")
 
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
 
        f.write(f"\nBest model: {best_model_name}, Accuracy: {best_accuracy:.4f}\n")
 
    result_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    result_df.to_csv(RESULT_DIR / "accuracy_comparison.csv", index=False, encoding="utf-8-sig")
    print("\nЗагваруудын accuracy:")
    print(result_df)
 
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
 
    print("\nДууслаа. Үр дүн results/ хавтаст хадгалагдлаа.")
 
 
if __name__ == "__main__":
    main()