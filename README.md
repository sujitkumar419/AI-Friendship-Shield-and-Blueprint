# 🤝 AI Friendship Shield & Blueprint

An end-to-end Machine Learning web application that leverages a custom **K-Nearest Neighbors (KNN)** classification pipeline (89.10% test accuracy) to analyze human trust traits, eliminate behavioral regrets, and generate personalized friendship blueprints.

---

## 🚀 Live Demo
🔗 **Access the Web App Here:** [Insert Your Streamlit App Link Here]

---

## 💡 The Core Problem
Many individuals face emotional regret and breach of trust because they evaluate new connections based purely on superficial metrics (e.g., sweet talk or frequency of interactions). They fail to screen for underlying operational triggers or environmental network traits. 

**AI Friendship Shield** solves this by treating connection evaluation as a geometric pattern recognition problem, mapping personal weaknesses against potential friend attributes to find stable psychological baselines.

---

## 📊 Dataset & Features
The model trains on a balanced synthetic dataset of **5,000 corporate and academic routines** across 14 encoded columns. It evaluates the following critical inputs:
*   **Helps In Need Score:** Metric assessing baseline availability during crises.
*   **Secret Keeping Rate (%):** Confidentiality data leakage score.
*   **Backbiting Frequency:** Incident rate of negative proxy communication.
*   **Calls Only For Help:** Binary flag for opportunistic interaction loops.
*   **Closest Friends Toxic Count:** A physical representation of the KNN neighborhood principle—evaluating the subject based on their immediate neighbors (padosi).
*   **Respect Matrix:** General social respect score toward hospitality staff and parents.

---

## 🧠 Machine Learning Architecture
*   **Algorithm:** K-Nearest Neighbors (KNN Classifier)
*   **Hyperparameters:** `n_neighbors=5`, `weights='uniform'`, Distance Metric = `Minkowski (Euclidean, p=2)`
*   **Data Pipeline:** Continuous feature variance handling via `StandardScaler`. Categorical tracking (`User_Profession`, `User_Weakness`) processed through complete One-Hot Encoding.
*   **Model Accuracy:** **89.10%** 
*   **Class 1 (Trustworthy Friend) Precision:** **91.00%** (Reducing user regret rate down to 9%).

---

## 🛡️ Premium UI Dashboard Features
1. **Dual-Language Engine:** Clean, real-time toggle between **Hindi** and **English** using synchronous text binding mapping (zero reloads required).
2. **Neon Glow Suggestion Matrix:** Custom styled CSS containers that render real-time Emergency Action Plans and Target Trait Blueprints tailored to the user's specific vulnerability.
3. **2D Friendship Trust Space:** An interactive scatter plot engine powered by `Plotly Express` that charts the audited subject explicitly against 5,000 baseline population cluster parameters.

---

## 🛠️ Installation & Local Execution

1. Clone this repository to your local PyCharm setup:
   ```bash
   git clone https://github.com[Your-Username]/AI-Friendship-Shield-and-Blueprint.git
   ```
2. Navigate into the directory and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Boot up the Streamlit server locally:
   ```bash
   streamlit run app.py
   ```

---

## 📁 Repository Structure
```text
├── app.py                            # Complete multi-language frontend dashboard script
├── Friendship_Blueprint_KNN.ipynb    # Comprehensive Jupyter Notebook (Data EDA & Model Training)
├── knn_similar_prediction.ipynb       # Target validation and baseline similarity mapping notebook
├── requirements.txt                  # Production cloud server dependency sheets
├── knn_friend_blueprint_model.pkl    # Serialized 89.10% accurate KNN weights
├── knn_friend_blueprint_scaler.pkl   # Serialized StandardScaler pipeline asset
├── knn_model_features.pkl            # Structural model column layout mapping array
└── README.md                         # Comprehensive portfolio documentation
```
