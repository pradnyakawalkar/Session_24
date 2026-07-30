import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="AIML Multi Model Prediction App",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# Custom CSS
# -------------------------------

st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

h1{
    color:#0066cc;
    text-align:center;
}

h2{
    color:#0066cc;
}

.stButton>button{
    width:100%;
    background:#0066cc;
    color:white;
    border-radius:10px;
    height:45px;
    font-size:18px;
}

.stButton>button:hover{
    background:#004999;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Classification Models
# -------------------------------

log_model = joblib.load("models/logistic_regression.pkl")
dt_model = joblib.load("models/decision_tree_classifier.pkl")
svm_model = joblib.load("models/svm_classifier.pkl")
knn_model = joblib.load("models/knn_classifier.pkl")
nb_model = joblib.load("models/naive_bayes.pkl")
classification_columns = joblib.load(
    "models/classification_columns.pkl"
)

regression_columns = joblib.load(
    "models/regression_columns.pkl"
)

classification_scaler = joblib.load("models/classification_scaler.pkl")

# -------------------------------
# Load Regression Models
# -------------------------------

lr_model = joblib.load("models/linear_regression.pkl")
dt_reg = joblib.load("models/decision_tree_regressor.pkl")
svr_model = joblib.load("models/svr_regressor.pkl")
knn_reg = joblib.load("models/knn_regressor.pkl")

regression_scaler = joblib.load("models/regression_scaler.pkl")

# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.title("🤖 AIML Assignment")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🚢 Titanic Classification",
        "💎 Diamond Regression"
    ]
)

# -------------------------------
# Home Page
# -------------------------------

if page == "🏠 Home":

    st.title("🤖 AIML Multi Model Prediction App")

    st.markdown("---")

    st.header("Welcome")

    st.write("""
This project is developed using Machine Learning.

### Classification Algorithms
- Logistic Regression
- Decision Tree
- Support Vector Machine
- KNN
- Naive Bayes

### Regression Algorithms
- Linear Regression
- Decision Tree Regressor
- Support Vector Regressor
- KNN Regressor

Use the sidebar to navigate between prediction pages.
""")

    st.success("Project Developed for AIML Session 24 Assignment")
 # ==========================================
# TITANIC CLASSIFICATION
# ==========================================

elif page == "🚢 Titanic Classification":

    st.title("🚢 Titanic Survival Prediction")

    algorithm = st.sidebar.selectbox(
        "Select Classification Algorithm",
        (
            "Logistic Regression",
            "Decision Tree",
            "Support Vector Machine",
            "KNN",
            "Naive Bayes"
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3]
        )

        sex = st.selectbox(
            "Gender",
            ["male", "female"]
        )

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=100,
            value=25
        )

        sibsp = st.number_input(
            "Siblings / Spouse",
            min_value=0,
            max_value=10,
            value=0
        )

    with col2:

        parch = st.number_input(
            "Parents / Children",
            min_value=0,
            max_value=10,
            value=0
        )

        fare = st.number_input(
            "Fare",
            min_value=0.0,
            value=20.0
        )

        embarked = st.selectbox(
            "Embarked",
            ["C", "Q", "S"]
        )

    # -------------------------
    # Encoding
    # -------------------------

    sex = 1 if sex == "male" else 0

    if embarked == "C":
        embarked = 0
    elif embarked == "Q":
        embarked = 1
    else:
        embarked = 2

    input_data = pd.DataFrame(
        [[
            pclass,
            sex,
            age,
            sibsp,
            parch,
            fare,
            embarked
        ]],
        columns=[
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked"
        ]
    )

    input_scaled = classification_scaler.transform(input_data)

    # -------------------------
    # Model Selection
    # -------------------------

    if algorithm == "Logistic Regression":
        model = log_model

    elif algorithm == "Decision Tree":
        model = dt_model

    elif algorithm == "Support Vector Machine":
        model = svm_model

    elif algorithm == "KNN":
        model = knn_model

    else:
        model = nb_model

# -------------------------
# Prediction
# -------------------------

    if st.button("Predict Survival"):

        prediction = model.predict(input_scaled)

        if prediction[0] == 1:

            st.success("✅ Passenger Will Survive")

        else:

            st.error("❌ Passenger Will Not Survive")
# ==========================================
# DIAMOND PRICE PREDICTION
# ==========================================
elif page == "💎 Diamond Regression":

    st.title("💎 Diamond Price Prediction")

    algorithm = st.sidebar.selectbox(
        "Select Regression Algorithm",
        [
            "Linear Regression",
            "Decision Tree",
            "SVR",
            "KNN Regressor"
        ]
    )

    col1, col2 = st.columns(2)

    with col1:

        carat = st.number_input("Carat", 0.1, 5.0, 0.50)
        depth = st.number_input("Depth", 40.0, 80.0, 61.5)
        table = st.number_input("Table", 40.0, 100.0, 55.0)
        x = st.number_input("Length (x)", 0.0, 20.0, 5.0)

    with col2:

        y = st.number_input("Width (y)", 0.0, 20.0, 5.0)
        z = st.number_input("Height (z)", 0.0, 20.0, 3.0)

        cut = st.selectbox(
            "Cut",
            ["Fair","Good","Very Good","Premium","Ideal"]
        )

        color = st.selectbox(
            "Color",
            ["D","E","F","G","H","I","J"]
        )

        clarity = st.selectbox(
            "Clarity",
            ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
        )

    if st.button("Predict Price"):

        input_df = pd.DataFrame(
            [[
                carat,
                cut,
                color,
                clarity,
                depth,
                table,
                x,
                y,
                z
            ]],
            columns=regression_columns
        )

        input_df = pd.get_dummies(input_df)

        for col in regression_scaler.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[regression_scaler.feature_names_in_]

        input_scaled = regression_scaler.transform(input_df)

        if algorithm == "Linear Regression":
            model = lr_model

        elif algorithm == "Decision Tree":
            model = dt_reg

        elif algorithm == "SVR":
            model = svr_model

        else:
            model = knn_reg

        prediction = model.predict(input_scaled)

        st.success(f"💰 Estimated Price : ${prediction[0]:,.2f}")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;'>

    ### 🤖 AIML Multi Model Prediction App

    **Datasets Used**
    - 🚢 Titanic Survival Prediction
    - 💎 Diamond Price Prediction

    **Algorithms**

    ✔ Logistic Regression

    ✔ Decision Tree

    ✔ Support Vector Machine

    ✔ KNN

    ✔ Naive Bayes

    ✔ Linear Regression

    ✔ Decision Tree Regressor

    ✔ SVR

    ✔ KNN Regressor

    ------------------------------------

    Developed for AIML Assignment

    </div>
    """,
    unsafe_allow_html=True
)