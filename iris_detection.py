"""
Iris Flower Species Classifier

A Streamlit app that predicts Iris flower species (setosa, versicolor, virginica)
based on sepal and petal measurements using scikit-learn classifiers.

Features:
  - Interactive sidebar sliders for feature input
  - Multiple ML models: Random Forest, SVM, Logistic Regression
  - Model comparison mode
  - Interactive visualizations (scatter plots, PCA projection, histograms)
  - Training data explorer with statistics
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered")

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown(
    """
<style>
.prediction-card {
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    margin: 1rem 0;
}
.prediction-setosa { background: linear-gradient(135deg, #e8d5f5, #c77dff); }
.prediction-versicolor { background: linear-gradient(135deg, #d4edda, #52b788); }
.prediction-virginica { background: linear-gradient(135deg, #f8d7da, #e63946); color: white; }
.flower-emoji { font-size: 3rem; }
.flower-name { font-size: 2rem; font-weight: 700; }
.confidence-text { font-size: 1.1rem; opacity: 0.85; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ────────────────────────────────────────────────────────
SPECIES_MAP = {
    0: {"name": "Setosa", "emoji": "🌸", "color": "#9b5de5"},
    1: {"name": "Versicolor", "emoji": "🌷", "color": "#52b788"},
    2: {"name": "Virginica", "emoji": "🌹", "color": "#e63946"},
}
FEAT_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FEAT_LABELS = ["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"]


# ── Cached resources ─────────────────────────────────────────────────
@st.cache_resource
def load_iris():
    """Load and cache the Iris dataset."""
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=FEAT_COLS)
    y = pd.Series(iris.target, name="species")
    return X, y, iris.target_names


@st.cache_resource
def get_model(name):
    """Train and cache a classifier by name."""
    X, y, _ = load_iris()
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    }
    clf = models[name]
    clf.fit(X, y)
    return clf


@st.cache_resource
def get_model_metrics(name):
    """Compute cross-val scores and confusion matrix for a model."""
    X, y, target_names = load_iris()
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    }
    clf = models[name]

    # 5-fold cross-validation accuracy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    # Full-fit confusion matrix
    clf.fit(X, y)
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)

    # Classification report as dict
    report = classification_report(y, y_pred, target_names=target_names, output_dict=True)

    return {
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "confusion_matrix": cm,
        "classification_report": report,
        "target_names": target_names,
    }


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌸 Iris Classifier")
    st.markdown("Adjust the sliders to set your flower's measurements.")

    sl_sepal_l = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
    sl_sepal_w = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4, 0.1)
    sl_petal_l = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3, 0.1)
    sl_petal_w = st.slider("Petal Width (cm)", 0.0, 2.6, 0.2, 0.1)

    st.markdown("---")
    model_choice = st.selectbox(
        "ML Model",
        ["Random Forest", "SVM", "Logistic Regression"],
        help="Which classifier to use for prediction.",
    )
    compare_mode = st.checkbox(
        "Compare all models",
        value=False,
        help="Show predictions from all three classifiers side by side.",
    )

# ── Input DataFrame ──────────────────────────────────────────────────
input_df = pd.DataFrame(
    [[sl_sepal_l, sl_sepal_w, sl_petal_l, sl_petal_w]],
    columns=FEAT_COLS,
)


# ── Header ───────────────────────────────────────────────────────────
st.title("🌸 Iris Flower Species Classifier")
st.markdown(
    "Predict the species of an Iris flower from its sepal and petal "
    "measurements. Adjust the sliders in the sidebar and see the results "
    "update instantly."
)
st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────
tab_pred, tab_viz, tab_perf, tab_data = st.tabs([
    "📊 Prediction",
    "📈 Visualizations",
    "🎯 Model Performance",
    "📖 About the Data",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ══════════════════════════════════════════════════════════════════════
with tab_pred:
    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.subheader("Your Input")
        display_df = input_df.rename(columns={
            "sepal_length": "Sepal L",
            "sepal_width": "Sepal W",
            "petal_length": "Petal L",
            "petal_width": "Petal W",
        })
        st.dataframe(display_df, hide_index=True, use_container_width=True)

        with st.expander("Dataset feature ranges"):
            X, _, _ = load_iris()
            rng = X.describe().loc[["min", "max"]]
            rng.columns = ["Sepal L", "Sepal W", "Petal L", "Petal W"]
            st.dataframe(rng, use_container_width=True)

    if not compare_mode:
        # ── Single model prediction ──
        clf = get_model(model_choice)
        pred = clf.predict(input_df)[0]
        proba = clf.predict_proba(input_df)[0]
        species = SPECIES_MAP[pred]

        css = f"prediction-{species['name'].lower()}"
        st.markdown(
            f"""<div class="prediction-card {css}">
                <div class="flower-emoji">{species['emoji']}</div>
                <div class="flower-name">Iris {species['name']}</div>
                <div class="confidence-text">Confidence: {proba[pred]:.1%}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Probability bar chart
        st.subheader("Prediction Probabilities")
        proba_df = pd.DataFrame({
            "Species": [SPECIES_MAP[i]["name"] for i in range(3)],
            "Probability": proba,
        })
        fig, ax = plt.subplots(figsize=(8, 2.5))
        colors = [SPECIES_MAP[i]["color"] for i in range(3)]
        bars = ax.barh(proba_df["Species"], proba_df["Probability"], color=colors, height=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        for bar, pr in zip(bars, proba):
            ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{pr:.1%}",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)

    else:
        # ── Compare all models ──
        st.subheader("🤖 Model Comparison")
        st.markdown("See how all three classifiers vote on your flower.")

        models = ["Random Forest", "SVM", "Logistic Regression"]
        cards = st.columns(3)

        for i, (m, col) in enumerate(zip(models, cards)):
            clf = get_model(m)
            pred = clf.predict(input_df)[0]
            proba = clf.predict_proba(input_df)[0]
            sp = SPECIES_MAP[pred]

            with col:
                st.markdown(
                    f"""<div style="text-align:center; padding:1rem; border-radius:10px;
                         border:1px solid #ddd; background:#fafafa;">
                        <div style="font-size:2rem;">{sp['emoji']}</div>
                        <div style="font-weight:600; margin:0.3rem 0;">{m}</div>
                        <div style="font-size:1.3rem; font-weight:700;">Iris {sp['name']}</div>
                        <div style="color:#666;">{proba[pred]:.1%} confidence</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.divider()

        st.subheader("Detailed Probabilities")
        rows = []
        for m in models:
            clf = get_model(m)
            proba = clf.predict_proba(input_df)[0]
            rows.append({
                "Model": m,
                **{SPECIES_MAP[i]["name"]: f"{proba[i]:.1%}" for i in range(3)},
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── Download results ──
    st.divider()
    pred_model = model_choice if not compare_mode else "All Models"
    download_rows = []
    models_to_run = [model_choice] if not compare_mode else ["Random Forest", "SVM", "Logistic Regression"]
    for m in models_to_run:
        clf = get_model(m)
        pred = clf.predict(input_df)[0]
        proba = clf.predict_proba(input_df)[0]
        download_rows.append({
            "Model": m,
            "Sepal Length": sl_sepal_l,
            "Sepal Width": sl_sepal_w,
            "Petal Length": sl_petal_l,
            "Petal Width": sl_petal_w,
            "Prediction": f"Iris {SPECIES_MAP[pred]['name']}",
            "Confidence": f"{proba[pred]:.1%}",
            "Prob_Setosa": f"{proba[0]:.3f}",
            "Prob_Versicolor": f"{proba[1]:.3f}",
            "Prob_Virginica": f"{proba[2]:.3f}",
        })
    download_df = pd.DataFrame(download_rows)
    csv_bytes = download_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Prediction Results (CSV)",
        data=csv_bytes,
        file_name="iris_prediction_results.csv",
        mime="text/csv",
        help="Download your input measurements, prediction, and probabilities as a CSV file.",
    )

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Visualizations
# ══════════════════════════════════════════════════════════════════════
with tab_viz:
    st.subheader("Explore the Iris Dataset")

    X, y, _ = load_iris()
    viz_df = X.copy()
    viz_df["species"] = y.map({i: SPECIES_MAP[i]["name"] for i in range(3)})

    viz_mode = st.radio(
        "Visualization type",
        ["Scatter Plot", "PCA Projection", "Feature Distributions"],
        horizontal=True,
    )

    if viz_mode == "Scatter Plot":
        x_opt = st.selectbox(
            "X-axis", FEAT_COLS, index=0,
            format_func=lambda c: FEAT_LABELS[FEAT_COLS.index(c)],
        )
        y_opt = st.selectbox(
            "Y-axis", FEAT_COLS, index=2,
            format_func=lambda c: FEAT_LABELS[FEAT_COLS.index(c)],
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        for i in range(3):
            subset = viz_df[viz_df["species"] == SPECIES_MAP[i]["name"]]
            ax.scatter(
                subset[x_opt], subset[y_opt],
                c=SPECIES_MAP[i]["color"], label=SPECIES_MAP[i]["name"],
                s=60, edgecolors="white", linewidth=0.8, alpha=0.8,
            )
        # Mark user input
        ax.scatter(
            input_df[x_opt].values[0], input_df[y_opt].values[0],
            c="black", s=200, marker="*",
            edgecolors="white", linewidth=1.5, zorder=5,
            label="Your Input",
        )
        ax.set_xlabel(FEAT_LABELS[FEAT_COLS.index(x_opt)], fontsize=12)
        ax.set_ylabel(FEAT_LABELS[FEAT_COLS.index(y_opt)], fontsize=12)
        ax.legend(fontsize=11)
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)

    elif viz_mode == "PCA Projection":
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        pca_df = pd.DataFrame({
            "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
            "species": viz_df["species"],
        })

        fig, ax = plt.subplots(figsize=(8, 5))
        for i in range(3):
            subset = pca_df[pca_df["species"] == SPECIES_MAP[i]["name"]]
            ax.scatter(
                subset["PC1"], subset["PC2"],
                c=SPECIES_MAP[i]["color"], label=SPECIES_MAP[i]["name"],
                s=60, edgecolors="white", linewidth=0.8, alpha=0.8,
            )
        user_pca = pca.transform(input_df)
        ax.scatter(
            user_pca[0, 0], user_pca[0, 1],
            c="black", s=200, marker="*",
            edgecolors="white", linewidth=1.5, zorder=5,
            label="Your Input",
        )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.set_title("PCA: 2D Projection of Iris Dataset", fontsize=13)
        ax.legend(fontsize=11)
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)

    elif viz_mode == "Feature Distributions":
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for idx, (feat, ax) in enumerate(zip(FEAT_COLS, axes.flat)):
            for i in range(3):
                subset = viz_df[viz_df["species"] == SPECIES_MAP[i]["name"]]
                ax.hist(
                    subset[feat], bins=12, alpha=0.6,
                    color=SPECIES_MAP[i]["color"],
                    label=SPECIES_MAP[i]["name"],
                )
            ax.set_title(FEAT_LABELS[idx], fontsize=11)
            ax.set_xlabel("cm")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
        plt.suptitle("Feature Distributions by Species", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════
with tab_perf:
    st.subheader("🎯 Model Performance")
    st.markdown(
        "Cross-validation accuracy, confusion matrices, and per-class metrics "
        "for each classifier on the full Iris dataset."
    )

    perf_model = st.selectbox(
        "Select model to evaluate",
        ["Random Forest", "SVM", "Logistic Regression"],
        key="perf_model_select",
    )

    metrics = get_model_metrics(perf_model)

    # Cross-validation scores
    st.subheader("Cross-Validation Accuracy (5-Fold)")
    cv_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(5)],
        "Accuracy": metrics["cv_scores"],
    })
    col_cv_left, col_cv_right = st.columns([1, 2])
    with col_cv_left:
        st.metric(
            "Mean CV Accuracy",
            f"{metrics['cv_mean']:.2%}",
            delta=f"±{metrics['cv_std']:.2%}",
        )
    with col_cv_right:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        colors_bar = ["#52b788" if s >= metrics["cv_mean"] else "#e63946" for s in metrics["cv_scores"]]
        bars = ax.bar(cv_df["Fold"], cv_df["Accuracy"], color=colors_bar, width=0.5)
        ax.axhline(metrics["cv_mean"], color="#333", linestyle="--", linewidth=1, label=f"Mean: {metrics['cv_mean']:.2%}")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=9)
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics["confusion_matrix"],
        display_labels=metrics["target_names"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"{perf_model} — Confusion Matrix", fontsize=12)
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Classification report
    st.subheader("Per-Class Metrics")
    report = metrics["classification_report"]
    report_rows = []
    for cls_name in metrics["target_names"]:
        cls = report[cls_name]
        report_rows.append({
            "Class": cls_name,
            "Precision": f"{cls['precision']:.3f}",
            "Recall": f"{cls['recall']:.3f}",
            "F1-Score": f"{cls['f1-score']:.3f}",
            "Support": int(cls["support"]),
        })
    # Add macro avg
    macro = report["macro avg"]
    report_rows.append({
        "Class": "Macro Avg",
        "Precision": f"{macro['precision']:.3f}",
        "Recall": f"{macro['recall']:.3f}",
        "F1-Score": f"{macro['f1-score']:.3f}",
        "Support": int(macro["support"]),
    })
    # Add weighted avg
    weighted = report["weighted avg"]
    report_rows.append({
        "Class": "Weighted Avg",
        "Precision": f"{weighted['precision']:.3f}",
        "Recall": f"{weighted['recall']:.3f}",
        "F1-Score": f"{weighted['f1-score']:.3f}",
        "Support": int(weighted["support"]),
    })
    st.dataframe(
        pd.DataFrame(report_rows),
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    # Model comparison summary
    st.subheader("📊 All Models — Side by Side")
    all_metrics_rows = []
    for m in ["Random Forest", "SVM", "Logistic Regression"]:
        m_metrics = get_model_metrics(m)
        all_metrics_rows.append({
            "Model": m,
            "CV Accuracy (mean)": f"{m_metrics['cv_mean']:.2%}",
            "CV Accuracy (std)": f"±{m_metrics['cv_std']:.2%}",
        })
    st.dataframe(
        pd.DataFrame(all_metrics_rows),
        hide_index=True,
        use_container_width=True,
    )

    # Bar chart comparing all models
    fig, ax = plt.subplots(figsize=(8, 3))
    model_names = [r["Model"] for r in all_metrics_rows]
    means = [m_metrics['cv_mean'] for m in model_names]
    stds = [m_metrics['cv_std'] for m in model_names]
    model_colors = [SPECIES_MAP[i]["color"] for i in range(3)]
    bars = ax.bar(model_names, means, yerr=stds, capsize=5, color=model_colors, width=0.4)
    ax.set_ylabel("CV Accuracy")
    ax.set_ylim(0, 1.1)
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean:.1%}",
            ha="center", fontsize=11, fontweight="bold",
        )
    sns.despine()
    st.pyplot(fig)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — About the Data
# ══════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("About the Iris Dataset")
    st.markdown(
        """
        The **Iris flower dataset** (also called *Fisher's Iris*) is a classic
        multivariate dataset introduced by the British statistician and biologist
        **Ronald Fisher** in 1936.

        **Dataset contents:**
        - **150 samples** — 50 from each of three Iris species
        - **4 features:** sepal length, sepal width, petal length, petal width (all in cm)
        - **3 target classes:** Iris setosa, Iris versicolor, Iris virginica

        **Why it's famous:**
        - Perfect for demonstrating classification algorithms
        - One of the most widely used datasets in ML education
        - One class (setosa) is linearly separable; the other two require more sophisticated models
        """
    )

    X, y, _ = load_iris()
    summary = X.copy()
    summary["species"] = y.map({i: SPECIES_MAP[i]["name"] for i in range(3)})

    st.subheader("Dataset Preview")
    st.dataframe(summary.head(10), use_container_width=True, hide_index=True)

    st.subheader("Summary Statistics")
    stats = X.describe().round(3)
    stats.columns = FEAT_LABELS
    st.dataframe(stats, use_container_width=True)

    st.subheader("Class Distribution")
    counts = summary["species"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = [SPECIES_MAP[i]["color"] for i in range(3)]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5)
    ax.set_ylabel("Count")
    ax.set_ylim(0, 70)
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(val), ha="center", fontsize=12,
        )
    sns.despine()
    st.pyplot(fig)
    plt.close(fig)

# ── Footer ───────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Built with Streamlit & scikit-learn 🚀</div>",
    unsafe_allow_html=True,
)
