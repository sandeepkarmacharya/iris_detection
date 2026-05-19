"""Tests for the Iris Flower Classifier application."""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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
