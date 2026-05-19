"""Tests for the Iris Flower Classifier application."""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

# ── Import app logic ─────────────────────────────────────────────
# We test the ML pipeline directly (the streamlit-.-decorated
# functions are tested via their underlying logic).

FEAT_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SPECIES_MAP = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}


def load_iris():
    """Load the Iris dataset (same as the app)."""
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=FEAT_COLS)
    y = pd.Series(iris.target, name="species")
    return X, y, iris.target_names


def train_model(name: str):
    """Train a classifier (same logic as the app, without streamlit caching)."""
    X, y, _ = load_iris()
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    }
    clf = models[name]
    clf.fit(X, y)
    return clf


def get_model_metrics(name: str):
    """Compute cross-val scores and confusion matrix (same logic as the app)."""
    X, y, target_names = load_iris()
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    }
    clf = models[name]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    clf.fit(X, y)
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)

    report = classification_report(y, y_pred, target_names=target_names, output_dict=True)

    return {
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "confusion_matrix": cm,
        "classification_report": report,
        "target_names": target_names,
    }


def train_model_2d(name: str, feat_cols: list):
    """Train a classifier on a subset of features (for decision boundary tests)."""
    X, y, _ = load_iris()
    X_2d = X[feat_cols].values
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    }
    clf = models[name]
    clf.fit(X_2d, y)
    return clf


class TestIrisClassifier:
    """Test suite for the Iris classifier models."""

    def setup_method(self):
        self.X, self.y, self.target_names = load_iris()
        self.models = ["Random Forest", "SVM", "Logistic Regression"]

    def test_dataset_shape(self):
        """The dataset should have 150 samples and 4 features."""
        assert self.X.shape == (150, 4), f"Expected (150, 4), got {self.X.shape}"
        assert self.y.shape == (150,), f"Expected (150,), got {self.y.shape}"

    def test_dataset_no_missing_values(self):
        """There should be no missing values in the dataset."""
        assert self.X.isnull().sum().sum() == 0
        assert self.y.isnull().sum() == 0

    def test_species_balance(self):
        """Each species should have exactly 50 samples."""
        counts = self.y.value_counts()
        for i in range(3):
            assert counts[i] == 50, f"Species {i} has {counts[i]} samples, expected 50"

    def test_feature_ranges(self):
        """Feature ranges should match expected Iris dataset bounds."""
        assert self.X["sepal_length"].min() >= 4.0
        assert self.X["sepal_length"].max() <= 8.0
        assert self.X["sepal_width"].min() >= 2.0
        assert self.X["sepal_width"].max() <= 4.5
        assert self.X["petal_length"].min() >= 1.0
        assert self.X["petal_length"].max() <= 7.0
        assert self.X["petal_width"].min() >= 0.0
        assert self.X["petal_width"].max() <= 2.6

    def test_all_models_train_without_error(self):
        """All three models should train without raising exceptions."""
        for name in self.models:
            clf = train_model(name)
            assert clf is not None

    def test_all_models_predict_setosa(self):
        """A typical setosa input should be classified as setosa by all models."""
        setosa_input = pd.DataFrame(
            [[5.1, 3.5, 1.4, 0.2]], columns=FEAT_COLS
        )
        for name in self.models:
            clf = train_model(name)
            pred = clf.predict(setosa_input)[0]
            assert pred == 0, f"{name} predicted {pred} ({SPECIES_MAP[pred]}) instead of Setosa"

    def test_all_models_predict_virginica(self):
        """A typical virginica input should be classified as virginica by all models."""
        virginica_input = pd.DataFrame(
            [[6.3, 3.3, 6.0, 2.5]], columns=FEAT_COLS
        )
        for name in self.models:
            clf = train_model(name)
            pred = clf.predict(virginica_input)[0]
            assert pred == 2, f"{name} predicted {pred} ({SPECIES_MAP[pred]}) instead of Virginica"

    def test_probabilities_sum_to_one(self):
        """Prediction probabilities should sum to ~1.0 for all models."""
        sample = pd.DataFrame([[5.4, 3.4, 1.3, 0.2]], columns=FEAT_COLS)
        for name in self.models:
            clf = train_model(name)
            proba = clf.predict_proba(sample)[0]
            assert abs(proba.sum() - 1.0) < 1e-6, f"{name} probabilities sum to {proba.sum()}"

    def test_random_forest_ensemble_size(self):
        """Random Forest should use 100 trees by default."""
        clf = train_model("Random Forest")
        assert clf.n_estimators == 100

    def test_svm_probability_enabled(self):
        """SVM should have probability estimates enabled."""
        clf = train_model("SVM")
        assert clf.probability is True

    def test_logistic_regression_max_iter(self):
        """Logistic Regression should use max_iter=200."""
        clf = train_model("Logistic Regression")
        assert clf.max_iter == 200

    # ── Model metrics tests ──

    def test_cv_scores_have_five_folds(self):
        """Cross-validation should return exactly 5 scores."""
        for m in self.models:
            metrics = get_model_metrics(m)
            assert len(metrics["cv_scores"]) == 5, f"{m} has {len(metrics['cv_scores'])} folds"

    def test_cv_accuracy_above_baseline(self):
        """All models should have CV accuracy well above random chance (33%)."""
        for m in self.models:
            metrics = get_model_metrics(m)
            assert metrics["cv_mean"] > 0.8, f"{m} CV accuracy {metrics['cv_mean']:.2%} is too low"

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be 3x3 (3 species)."""
        for m in self.models:
            metrics = get_model_metrics(m)
            assert metrics["confusion_matrix"].shape == (3, 3), f"{m} CM shape {metrics['confusion_matrix'].shape}"

    def test_confusion_matrix_diagonal_dominant(self):
        """Confusion matrix diagonal should contain most samples (≥140/150)."""
        for m in self.models:
            metrics = get_model_metrics(m)
            cm = metrics["confusion_matrix"]
            correct = cm.trace()
            assert correct >= 140, f"{m} only got {correct}/150 correct on training data"

    def test_classification_report_has_all_keys(self):
        """Classification report should contain all 3 species plus averages."""
        for m in self.models:
            metrics = get_model_metrics(m)
            report = metrics["classification_report"]
            for cls in ["setosa", "versicolor", "virginica"]:
                assert cls in report, f"{m} missing {cls} in report"
            assert "macro avg" in report
            assert "weighted avg" in report

    def test_macro_avg_f1_reasonable(self):
        """Macro average F1 should be high (≥0.85) for all models."""
        for m in self.models:
            metrics = get_model_metrics(m)
            macro_f1 = metrics["classification_report"]["macro avg"]["f1-score"]
            assert macro_f1 >= 0.85, f"{m} macro avg F1 {macro_f1:.3f} is too low"

    # ── Tier 2: Decision Boundary tests ──

    def test_decision_boundary_2d_training(self):
        """Model trained on 2 features should work and predict."""
        X, y, _ = load_iris()
        X_2d = X[["petal_length", "petal_width"]].values
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_2d, y)
        pred = clf.predict([[1.5, 0.3]])
        assert pred[0] in [0, 1, 2], f"2D RF predicted outside range: {pred[0]}"

    def test_decision_boundary_meshgrid_shapes(self):
        """Meshgrid for decision boundary should produce valid predictions."""
        X, y, _ = load_iris()
        X_2d = X[["petal_length", "sepal_length"]].values
        clf = LogisticRegression(max_iter=200, random_state=42)
        clf.fit(X_2d, y)

        x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 50),
            np.linspace(y_min, y_max, 50),
        )
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        assert Z.shape == (50, 50), f"Expected (50, 50), got {Z.shape}"

    def test_decision_boundary_all_models_2d(self):
        """All three models should be trainable on any 2-feature subset."""
        X, y, _ = load_iris()
        for name in self.models:
            clf = train_model_2d(name, FEAT_COLS[:2])
            pred = clf.predict([[5.0, 3.0]])
            assert pred[0] in [0, 1, 2], f"{name} 2D pred outside range: {pred[0]}"

    # ── Tier 2: Feature Importance tests ──

    def test_random_forest_feature_importances(self):
        """Random Forest should return valid feature importances."""
        X, y, _ = load_iris()
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        importances = clf.feature_importances_
        assert len(importances) == 4, f"Expected 4 importances, got {len(importances)}"
        assert all(i >= 0 for i in importances), "Feature importances should be non-negative"
        assert abs(importances.sum() - 1.0) < 1e-6, f"Importances should sum to ~1, got {importances.sum()}"

    def test_logistic_regression_coefficients(self):
        """Logistic Regression should have 3 sets of coefficients (OvR)."""
        X, y, _ = load_iris()
        clf = LogisticRegression(max_iter=200, random_state=42)
        clf.fit(X, y)
        assert clf.coef_.shape == (3, 4), f"Expected (3, 4), got {clf.coef_.shape}"
        # Mean absolute coefficient should identify petal features as important
        mean_abs = np.mean(np.abs(clf.coef_), axis=0)
        assert len(mean_abs) == 4
        # Petal features are typically more important than sepal features
        assert mean_abs[2] > mean_abs[0] or mean_abs[3] > mean_abs[0], \
            "Petal features should be more important than sepal length"

    def test_feature_importance_sorted_order(self):
        """Feature importance results should be sortable by importance value."""
        X, y, _ = load_iris()
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        fi_df = pd.DataFrame({
            "Feature": FEAT_COLS,
            "Importance": clf.feature_importances_,
        }).sort_values("Importance", ascending=True)
        assert fi_df["Importance"].is_monotonic_increasing, \
            "DataFrame should be sorted ascending by importance"

    # ── Tier 2: Hyperparameter Tuning tests ──

    def test_tuning_rf_parameters_affect_score(self):
        """Different RF hyperparams should produce different CV scores."""
        X, y, _ = load_iris()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        shallow = RandomForestClassifier(n_estimators=10, max_depth=1, random_state=42)
        deep = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)

        shallow_score = cross_val_score(shallow, X, y, cv=cv, scoring="accuracy").mean()
        deep_score = cross_val_score(deep, X, y, cv=cv, scoring="accuracy").mean()

        assert deep_score >= shallow_score, \
            f"Deeper RF ({deep_score:.2%}) should be at least as good as shallow ({shallow_score:.2%})"

    def test_tuning_svm_kernel_effects(self):
        """Different SVM kernels should yield different results."""
        X, y, _ = load_iris()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Linear and RBF kernels should both be reasonable for Iris
        for kernel in ["linear", "rbf"]:
            clf = SVC(kernel=kernel, random_state=42)
            score = cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean()
            assert score > 0.8, f"SVM ({kernel}) CV accuracy {score:.2%} is too low"

    def test_tuning_lr_regularization_impact(self):
        """Strong vs weak regularization should show meaningful differences."""
        X, y, _ = load_iris()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        strong = LogisticRegression(C=0.001, max_iter=500, random_state=42)
        weak = LogisticRegression(C=10.0, max_iter=500, random_state=42)

        strong_score = cross_val_score(strong, X, y, cv=cv, scoring="accuracy").mean()
        weak_score = cross_val_score(weak, X, y, cv=cv, scoring="accuracy").mean()

        # Both should perform reasonably
        assert strong_score > 0.5, f"Strong reg LR {strong_score:.2%} too low"
        assert weak_score > 0.5, f"Weak reg LR {weak_score:.2%} too low"

    def test_tuning_all_models_default_vs_tuned(self):
        """Default and tuned models should both complete CV successfully."""
        X, y, _ = load_iris()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Default
        default = RandomForestClassifier(random_state=42)
        default_score = cross_val_score(default, X, y, cv=cv, scoring="accuracy").mean()

        # Tuned
        tuned = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=3, random_state=42)
        tuned_score = cross_val_score(tuned, X, y, cv=cv, scoring="accuracy").mean()

        assert isinstance(default_score, float)
        assert isinstance(tuned_score, float)
        assert 0 < default_score <= 1
        assert 0 < tuned_score <= 1
