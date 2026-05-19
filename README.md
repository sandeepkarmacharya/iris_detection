# Iris Flower Species Classifier

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?logo=streamlit&logoColor=white)](https://karma0san-iris-detection.streamlit.app/)
[![CI](https://github.com/sandeepkarmacharya/iris_detection/actions/workflows/ci.yml/badge.svg)](https://github.com/sandeepkarmacharya/iris_detection/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://hub.docker.com)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive **Iris flower species predictor** built with [Streamlit](https://streamlit.io) and [scikit-learn](https://scikit-learn.org). Adjust sepal/petal measurements via sliders and instantly see which Iris species your flower matches -- along with prediction confidence and visual context.

> **Note:** This repo is an **Iris flower classifier**, not an eye/iris detection app. Despite the historical repo name, the project is a classic ML demonstration using Fisher's Iris dataset.

---

## Features

| **Instant Predictions** -- Adjust sliders to see predicted species + confidence
| **Multiple Models** -- Random Forest, SVM, Logistic Regression, **XGBoost**
| **Model Comparison** -- Compare all 4 classifiers side-by-side
| **Interactive Visualizations** -- Scatter plots, PCA projection, feature distribution histograms
| **Decision Boundaries** -- 2D visualization of how models separate species
| **Hyperparameter Tuning** -- Interactive sliders to tweak model params and compare accuracy
| **Feature Importance** -- See which measurements drive predictions
| **SHAP Explanations** -- Waterfall plots showing feature contributions to each prediction
| **Model Performance** -- CV accuracy, confusion matrices, per-class precision/recall/F1
| **CSV Export** -- Download prediction results for record-keeping
| **Cached Training** -- Models train once and are reused via `@st.cache_resource`
| **Docker Support** -- Run anywhere with Docker or docker-compose
| **CI Pipeline** -- Automated testing on every push (Python 3.10-3.12)

---

## Live Demo

Try the app live on Streamlit Community Cloud:

**[karma0san-iris-detection.streamlit.app](https://karma0san-iris-detection.streamlit.app/)**

---

## How It Works

1. Use the **sidebar sliders** to set your flower's sepal length/width and petal length/width
2. Choose a **classifier** (Random Forest, SVM, Logistic Regression, or XGBoost)
3. The app instantly predicts the species and shows:
   - A **styled prediction card** with confidence percentage
   - **Probability bar chart** for all 3 species
   - **SHAP waterfall plot** explaining which features drove the prediction (expandable)
   - Your input point plotted on the **training data visualizations**

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/sandeepkarmacharya/iris_detection.git
cd iris_detection

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run iris_detection.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Docker

### Build and run with Docker

```bash
# Build the image
docker build -t iris-classifier .

# Run the container
docker run -d -p 8501:8501 --name iris-classifier iris-classifier
```

### Or use docker-compose

```bash
docker compose up -d
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> The container includes automatic health checks and restarts on failure.

---

## Project Structure

```
iris_detection/
├── iris_detection.py       # Main Streamlit application
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Docker Compose configuration
├── .dockerignore           # Files excluded from Docker builds
├── tests/
│   └── test_app.py         # Unit tests (pytest)
├── .github/
│   └── workflows/
│       └── ci.yml          # CI pipeline (GitHub Actions)
├── .devcontainer/
│   └── devcontainer.json   # VS Code / GitHub Codespaces config
├── .gitignore              # Ignored files
├── LICENSE                 # MIT license
└── README.md               # This file
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## About the Dataset

The [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) (Fisher's Iris) is a classic multivariate dataset with:

- **150 samples** -- 50 from each of 3 Iris species
- **4 features:** sepal length, sepal width, petal length, petal width
- **3 classes:** Iris setosa, Iris versicolor, Iris virginica

Introduced by Ronald Fisher in 1936, it's one of the most famous datasets in machine learning -- often called the "Hello World" of pattern recognition.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web app framework |
| [scikit-learn](https://scikit-learn.org) | ML models and dataset |
| [XGBoost](https://xgboost.readthedocs.io) | Gradient boosted trees classifier |
| [SHAP](https://shap.readthedocs.io) | Model explainability (waterfall plots) |
| [Matplotlib](https://matplotlib.org) | Plotting and charts |
| [Seaborn](https://seaborn.pydata.org) | Statistical visualizations |
| [pandas](https://pandas.pydata.org) | Data handling |
| [Docker](https://docker.com) | Containerization |

---

## License

This project is licensed under the **MIT License** -- see the [LICENSE](LICENSE) file for details.
