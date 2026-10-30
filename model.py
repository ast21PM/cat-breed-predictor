"""
model.py — обучение модели прогнозирования породы кошки и сохранение весов.

Запуск:
    python model.py

На экран выводится точность модели, а в папку data/ сохраняется файл
весовых коэффициентов model_weights.mw, который использует приложение app.py.
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR / "data"
WEIGHTS_PATH = DATA_DIR / "model_weights.mw"

# Признаки, которые подаются в модель (должны совпадать с формой в app.py)
NUMERIC_COLS = [
    "Age_in_years", "Weight", "Body_length",
    "Sleep_time_hours", "Owner_play_time_minutes",
]
BINARY_COLS = ["Gender", "Neutered_or_spayed", "Allowed_outdoor"]
CATEGORICAL_COLS = ["Fur_colour_dominant", "Fur_pattern", "Eye_colour", "Preferred_food"]
FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS
TARGET_COL = "Breed"


def load_dataset():
    """Читает очищенный датасет и приводит признаки к числовому виду."""
    df = pd.read_csv(DATA_DIR / "cat_breeds_clean.csv", sep=";")

    # Бинарные признаки -> 0/1 в том же представлении, что и форма в app.py
    df["Gender"] = df["Gender"].map({"female": 1, "male": 0})
    df["Neutered_or_spayed"] = df["Neutered_or_spayed"].astype(str).str.upper().map(
        {"TRUE": 1, "FALSE": 0}
    )
    df["Allowed_outdoor"] = df["Allowed_outdoor"].astype(str).str.upper().map(
        {"TRUE": 1, "FALSE": 0}
    )

    # Категориальные признаки оставляем в исходном (английском) виде —
    # app.py перед предсказанием приводит выбор пользователя к этим же значениям.
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("unknown")

    # Заполняем возможные пропуски в бинарных/числовых признаках
    for col in BINARY_COLS:
        df[col] = df[col].fillna(0).astype(int)
    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def build_pipeline():
    """Пайплайн: one-hot кодирование категорий + случайный лес."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)
        ],
        remainder="passthrough",
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=42, class_weight="balanced", n_estimators=200
        )),
    ])


def main():
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    labels = sorted(y.unique())
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Сохраняем модель вместе с метриками, чтобы app.py не переобучал её
    bundle = {
        "pipeline": pipeline,
        "accuracy": accuracy,
        "conf_matrix": conf_matrix,
        "class_report": class_report,
        "labels": labels,
        "feature_cols": FEATURE_COLS,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Model accuracy is {accuracy}")
    print(f"Model was saved to data/model_weights.mw")


if __name__ == "__main__":
    main()
