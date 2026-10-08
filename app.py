"""Streamlit interface for the student wellness awareness prototype.

The model predicts a class from the selected dataset. It does not diagnose a
condition or provide a validated estimate of an individual student's wellness.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "models" / "wellness_model.joblib"
METADATA_PATH = PROJECT_DIR / "models" / "model_metadata.json"
IMPORTANCE_PATH = PROJECT_DIR / "models" / "feature_importance.csv"

st.set_page_config(
    page_title="Student Wellness Risk Assessment",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .main .block-container {
        max-width: 1120px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
      }
      .project-kicker {
        color: var(--primary-color);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
      }
      .project-subtitle {
        color: inherit;
        opacity: 0.78;
        font-size: 1.08rem;
        margin-top: -0.4rem;
        margin-bottom: 1.4rem;
      }
      .result-label {
        color: inherit;
        opacity: 0.78;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
      }
      .result-value {
        color: inherit;
        font-size: 2.4rem;
        font-weight: 750;
        line-height: 1.1;
      }
      @media (max-width: 768px) {
        .main .block-container {
          padding-left: 1rem;
          padding-right: 1rem;
          padding-top: 1.2rem;
        }
        .result-value {
          font-size: 2rem;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the saved scikit-learn pipeline once per Streamlit session."""
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


st.markdown('<div class="project-kicker">AI / ML Internship Prototype</div>', unsafe_allow_html=True)
st.title("Student Wellness Risk Assessment and Support System")
st.markdown(
    '<div class="project-subtitle">A simple awareness interface based on patterns in a public student lifestyle dataset.</div>',
    unsafe_allow_html=True,
)

st.warning(
    "The output is a prediction of the category represented in the dataset. "
    "It is **not a diagnosis** or a validated assessment of a student's actual "
    "stress or wellness. This prototype is for educational demonstration only."
)

st.markdown(
    "Enter the four values used by the model to see which `Stress_Level` "
    "category it predicts from the dataset. The prediction is a category label, "
    "not a personal risk probability."
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.info(
        "The trained model files are not available yet. From the project folder, "
        "run `python train_model.py`, then refresh this page."
    )
    st.stop()

model, metadata = load_model()

with st.container(border=True):
    st.subheader("Student Inputs")
    st.caption(
        "Ranges match the values observed in the dataset. Student ID, study "
        "hours, and sleep hours are not used as model inputs."
    )

    with st.form("wellness_input_form"):
        left_column, right_column = st.columns(2)

        with left_column:
            extracurricular = st.number_input(
                "Extracurricular hours per day",
                min_value=0.0,
                max_value=4.0,
                value=2.0,
                step=0.1,
                help="Daily time spent on extracurricular activities in the dataset.",
            )
            social = st.number_input(
                "Social hours per day",
                min_value=0.0,
                max_value=6.0,
                value=2.5,
                step=0.1,
                help="Daily social time in the dataset.",
            )

        with right_column:
            physical_activity = st.number_input(
                "Physical activity hours per day",
                min_value=0.0,
                max_value=13.0,
                value=4.0,
                step=0.1,
                help="Daily physical activity time in the dataset.",
            )
            gpa = st.number_input(
                "GPA",
                min_value=2.24,
                max_value=4.0,
                value=3.1,
                step=0.01,
                format="%.2f",
                help="GPA value as represented in the dataset.",
            )

        submitted = st.form_submit_button(
            "Generate category prediction", type="primary", use_container_width=True
        )

if submitted:
    # Keep the input names and order aligned with train_model.py metadata.
    input_values = {
        "Extracurricular_Hours_Per_Day": extracurricular,
        "Social_Hours_Per_Day": social,
        "Physical_Activity_Hours_Per_Day": physical_activity,
        "GPA": gpa,
    }
    model_input = pd.DataFrame(
        [[input_values[column] for column in metadata["feature_columns"]]],
        columns=metadata["feature_columns"],
    )
    predicted_label = str(model.predict(model_input)[0])

    with st.container(border=True):
        st.subheader("Prediction")
        st.markdown('<div class="result-label">Predicted dataset category</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value">{predicted_label}</div>', unsafe_allow_html=True)
        st.caption(
            "This is the model's prediction of the dataset category—not an "
            "individual diagnosis or validated wellness assessment."
        )

    suggestions = {
        "Low": [
            "Keep using routines and activities that feel manageable for you.",
            "Take regular breaks and make time for rest and enjoyable activities.",
        ],
        "Moderate": [
            "Try breaking a large task into a few smaller steps and taking short breaks.",
            "If useful, check in with a trusted friend, family member, mentor, or advisor.",
        ],
        "High": [
            "Consider choosing one small, manageable next step and giving yourself time to rest.",
            "You could talk with someone you trust or ask your campus student-support service what general help is available.",
        ],
    }

    with st.container(border=True):
        st.subheader("General Suggestions")
        st.caption("These suggestions are general and non-medical.")
        suggestion_items = suggestions.get(
            predicted_label,
            [
                "Consider taking a break, reviewing your workload, or talking with someone you trust.",
            ],
        )
        st.markdown("\n".join(f"- {suggestion}" for suggestion in suggestion_items))

st.divider()
st.header("Model Insights")
st.caption(
    "Evaluation values below are unchanged. They describe performance on the "
    "held-out dataset split, not real-world wellness accuracy."
)

test_metrics = metadata.get("test_metrics", {})
if test_metrics:
    metric_columns = st.columns(3)
    metric_columns[0].metric("Test accuracy", f"{test_metrics['accuracy']:.3f}")
    metric_columns[1].metric(
        "Balanced accuracy", f"{test_metrics['balanced_accuracy']:.3f}"
    )
    metric_columns[2].metric("Macro F1", f"{test_metrics['macro_f1']:.3f}")

st.markdown(f"**Selected model:** {metadata['model_name']}")

if IMPORTANCE_PATH.exists():
    importance_data = pd.read_csv(IMPORTANCE_PATH)
    display_names = {
        "Extracurricular_Hours_Per_Day": "Extracurricular hours",
        "Social_Hours_Per_Day": "Social hours",
        "Physical_Activity_Hours_Per_Day": "Physical activity hours",
        "GPA": "GPA",
    }
    importance_data["display_feature"] = importance_data["feature"].map(
        lambda feature: display_names.get(feature, feature)
    )
    importance_data = importance_data.sort_values("importance_mean", ascending=False)

    st.subheader("Global model feature importance")
    st.caption(
        "Global permutation importance: each input is shuffled in the test data, "
        "then the change in model performance is measured. A larger bar means "
        "the model relied more on that input in this evaluation. This is not an "
        "explanation of one prediction and does not show cause and effect."
    )
    largest_importance = max(
        float(importance_data["importance_mean"].abs().max()), 1e-12
    )
    for row in importance_data.itertuples(index=False):
        feature_column, bar_column, value_column = st.columns([2, 5, 1])
        feature_column.markdown(f"**{row.display_feature}**")
        # Progress bars stay readable in both light and dark Streamlit themes.
        # The numeric value is shown as well; the bar is scaled to this chart's
        # largest absolute importance value.
        bar_column.progress(abs(float(row.importance_mean)) / largest_importance)
        value_column.markdown(f"`{row.importance_mean:.3f}`")
    st.caption(
        "Bars are scaled to the largest absolute importance value shown above; "
        "numbers are the original mean macro-F1 changes."
    )

with st.expander("About / Disclaimer"):
    st.markdown(
        "The dataset page describes survey responses, but its provenance has "
        "not been independently verified.\n\n"
        "The dataset page says its `Stress_Level` label is derived from study "
        "and sleep hours. Those fields were excluded from model inputs, but the "
        "target remains a dataset label rather than an independently validated "
        "wellness outcome.\n\n"
        "The output is for a classroom prototype and should not be used to "
        "make decisions about students or to replace support from people.\n"
        "\n\nSuggestions are general and non-medical."
    )

st.caption("Student Wellness Risk Assessment and Support System · Internship prototype")
