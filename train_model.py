"""Train and compare simple models for the prototype dataset.

Run from the project folder with:
    python train_model.py

The saved classifier learns the dataset's Stress_Level labels. It is not a
validated measure of a student's actual wellness or medical condition.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "student_lifestyle_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "wellness_model.joblib"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"

TARGET_COLUMN = "Stress_Level"

# Student_ID is only an identifier. Study and sleep are left out because the
# dataset page says they are used to derive Stress_Level; including them could
# let a model simply reproduce the label-generation rule.
FEATURE_COLUMNS = [
    "Extracurricular_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "GPA",
]


def make_models() -> dict[str, Pipeline]:
    """Create a small set of beginner-friendly models to compare."""
    return {
        # A baseline that always predicts the most common training class.
        "Dummy baseline": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        # Scaling helps Logistic Regression compare numeric features fairly.
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=5, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    """Train on a stratified split, compare metrics, and save the best model."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place student_lifestyle_dataset.csv in data/raw/ first."
        )

    data = pd.read_csv(DATA_PATH)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(f"Required dataset columns are missing: {missing_columns}")

    # Remove rows with missing target labels. Feature gaps are filled inside
    # each model pipeline, which learns imputation values from training data.
    data = data.dropna(subset=[TARGET_COLUMN]).copy()
    X = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = data[TARGET_COLUMN].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    comparison_rows: list[dict[str, float | str]] = []
    models = make_models()
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Model features: {', '.join(FEATURE_COLUMNS)}")
    print("\nMODEL COMPARISON (5-fold cross-validation on training data)")

    # Compare models using only the training portion. The separate test set
    # stays untouched until after we have selected a model.
    for model_name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cross_validation,
            scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
            n_jobs=1,
        )
        metrics = {
            "model": model_name,
            "cv_macro_f1_mean": scores["test_macro_f1"].mean(),
            "cv_macro_f1_std": scores["test_macro_f1"].std(),
            "cv_balanced_accuracy_mean": scores["test_balanced_accuracy"].mean(),
        }
        comparison_rows.append(metrics)
        print(
            f"{model_name}: macro_f1={metrics['cv_macro_f1_mean']:.3f} "
            f"(+/- {metrics['cv_macro_f1_std']:.3f}), "
            f"balanced_accuracy={metrics['cv_balanced_accuracy_mean']:.3f}"
        )

    # Macro F1 gives each class equal weight; balanced accuracy averages recall
    # across classes. Use CV macro F1 to select a model without peeking at test.
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["cv_macro_f1_mean", "cv_balanced_accuracy_mean"], ascending=False
    )
    best_name = str(comparison.iloc[0]["model"])
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    best_predictions = best_model.predict(X_test)
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, best_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, best_predictions)),
        "macro_f1": float(f1_score(y_test, best_predictions, average="macro")),
    }

    # Permutation importance estimates how much the selected model's test
    # macro-F1 changes when one feature is shuffled. It describes model
    # behavior; it does not show that a feature causes a wellness outcome.
    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=10,
        random_state=42,
        n_jobs=1,
    )
    importance_table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False)
    importance_table.to_csv(IMPORTANCE_PATH, index=False)
    joblib.dump(best_model, MODEL_PATH)

    # Save the feature order, class labels, and evaluation details for the app.
    metadata = {
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "classes": [str(label) for label in best_model.classes_],
        "cross_validation": {
            key: float(value)
            for key, value in comparison.iloc[0].to_dict().items()
            if key != "model"
        },
        "test_metrics": test_metrics,
        "split": {
            "test_size": 0.20,
            "random_state": 42,
            "stratified": True,
            "cv_folds": 5,
        },
        "interpretation_note": (
            "Learns labels from the selected dataset only; not a validated or "
            "clinical wellness-risk estimate."
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSELECTED MODEL: {best_name}")
    print("\nHELD-OUT TEST METRICS (used once after model selection)")
    print(
        f"accuracy={test_metrics['accuracy']:.3f}, "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.3f}, "
        f"macro_f1={test_metrics['macro_f1']:.3f}"
    )
    print("\nCLASSIFICATION REPORT (selected model, held-out test set)")
    print(classification_report(y_test, best_predictions, zero_division=0))
    print("CONFUSION MATRIX (rows=true class, columns=predicted class)")
    labels = best_model.classes_
    print(pd.DataFrame(confusion_matrix(y_test, best_predictions, labels=labels), index=labels, columns=labels))
    print(f"\nSaved model: {MODEL_PATH.relative_to(PROJECT_DIR)}")
    print(f"Saved comparison: {COMPARISON_PATH.relative_to(PROJECT_DIR)}")
    print(f"Saved feature importance: {IMPORTANCE_PATH.relative_to(PROJECT_DIR)}")
    print(f"Saved metadata: {METADATA_PATH.relative_to(PROJECT_DIR)}")
    print("\nReminder: these scores measure reproduction of this dataset's labels only.")


if __name__ == "__main__":
    main()
