from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os, warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Global model artefacts ────────────────────────────────────────────────────
model        = None
scaler       = None
ohe          = None
ordinal_enc  = None
feature_cols = None   # final column order fed to scaler

def train_model():
    """Train on the dataset and store artefacts globally."""
    global model, scaler, ohe, ordinal_enc, feature_cols

    csv_path = os.path.join(os.path.dirname(__file__), "student_dropout_dataset_v3.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "student_dropout_dataset_v3.csv not found next to app.py. "
            "Place the dataset file in the same folder."
        )

    df = pd.read_csv(csv_path)

    # ── Impute ────────────────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    # ── Drop unused columns (same as notebook) ────────────────────────────────
    df = df.drop(columns=["Age", "Student_ID", "GPA", "Semester_GPA", "Department", "Semester"])

    # ── One-hot encode ────────────────────────────────────────────────────────
    ohe_cols = ["Gender", "Internet_Access", "Part_Time_Job"]
    ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    encoded = pd.DataFrame(
        ohe.fit_transform(df[ohe_cols]),
        columns=ohe.get_feature_names_out(ohe_cols),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=ohe_cols), encoded], axis=1)

    # ── Ordinal encode Scholarship & Parental_Education ───────────────────────
    ordinal_enc = OrdinalEncoder(
        categories=[["No", "Yes"], ["High School", "Bachelor", "Master", "PhD"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    df[["Scholarship", "Parental_Education"]] = ordinal_enc.fit_transform(
        df[["Scholarship", "Parental_Education"]]
    )

    # ── Split & scale ─────────────────────────────────────────────────────────
    X = df.drop("Dropout", axis=1)
    y = df["Dropout"]
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # ── Train final model (Logistic Regression — best recall) ─────────────────
    model = LogisticRegression(
        random_state=42, class_weight="balanced",
        max_iter=100, C=0.01, penalty="l1", solver="liblinear"
    )
    model.fit(X_train_scaled, y_train)
    print("✓ Model trained successfully.")


def build_input_df(form):
    """Convert flat form dict → single-row DataFrame ready for scaler."""
    raw = {
        "Family_Income":          float(form["family_income"]),
        "Study_Hours_per_Day":    float(form["study_hours"]),
        "Attendance_Rate":        float(form["attendance_rate"]),
        "Assignment_Delay_Days":  float(form["assignment_delay"]),
        "Travel_Time_Minutes":    float(form["travel_time"]),
        "Stress_Index":           float(form["stress_index"]),
        "CGPA":                   float(form["cgpa"]),
        # OHE inputs (raw categories)
        "Gender":                 form["gender"],
        "Internet_Access":        form["internet_access"],
        "Part_Time_Job":          form["part_time_job"],
        # Ordinal inputs (raw categories)
        "Scholarship":            form["scholarship"],
        "Parental_Education":     form["parental_education"],
    }
    df = pd.DataFrame([raw])

    # OHE
    ohe_cols = ["Gender", "Internet_Access", "Part_Time_Job"]
    encoded = pd.DataFrame(
        ohe.transform(df[ohe_cols]),
        columns=ohe.get_feature_names_out(ohe_cols),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=ohe_cols), encoded], axis=1)

    # Ordinal
    df[["Scholarship", "Parental_Education"]] = ordinal_enc.transform(
        df[["Scholarship", "Parental_Education"]]
    )

    # Reorder columns to match training
    df = df[feature_cols]
    return df


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        df = build_input_df(request.form)
        scaled = scaler.transform(df)
        pred   = int(model.predict(scaled)[0])
        prob   = float(model.predict_proba(scaled)[0][1])
        return jsonify({
            "prediction": pred,
            "probability": round(prob * 100, 1),
            "label": "At risk of dropout" if pred == 1 else "Likely to stay enrolled",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_model()
    app.run(debug=True)

if __name__ == "__main__":
    train_model()
    app.run(host="0.0.0.0", port=10000)
