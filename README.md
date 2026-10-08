# AI-Based Student Wellness Risk Assessment and Support System

A small educational prototype for exploring how academic and lifestyle factors might be used to provide a **wellness-risk awareness indicator**. It is not a medical or psychological assessment and does not diagnose any condition.

## Live Demo

[Open the Student Wellness Risk Assessment app](https://student-wellness-risk-assessment-4y7ck5rd4tk2oh9omhbq9a.streamlit.app/)



## Project Status

The prototype is implemented and deployed. It uses a trained Random Forest model to predict the stress-level category represented in the dataset from four input features. The Streamlit app provides predictions, general non-medical suggestions, and global model feature importance.

This is a prototype for educational demonstration. Its predictions are not a diagnosis or a validated assessment of a student’s actual stress or wellness.

## Planned workflow

1. Document the dataset source, license, columns, and target-label definition.
2. Explore data quality, missing values, ranges, and class balance (`python explore_data.py`).
3. Preprocess data with a scikit-learn pipeline and make a stratified train/test split.
4. Compare simple baseline and classification models using appropriate metrics.
5. Select and save a model only after evaluation.
6. Build the Streamlit interface, general supportive suggestions, and simple feature explanation.
7. Document limitations and verify the app.

## Setup

From the project folder in PowerShell, create a virtual environment and install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Explore the data and train the prototype model:

```powershell
python explore_data.py
python train_model.py
```

The training script writes the selected model and evaluation summaries into `models/`. Start the Streamlit app with:

```powershell
python -m streamlit run app.py
```

## Dataset used

- **Name:** Daily Lifestyle and Academic Performance of Students (downloaded file: `student_lifestyle_dataset.csv`).
- **Source:** [Kaggle dataset page](https://www.kaggle.com/datasets/steve1215rogg/student-lifestyle-dataset).
- **What the page says about provenance:** It describes 2,000 Google Form survey responses, primarily from India, covering August 2023 to May 2024. This provenance has not been independently verified.
- **License:** Check the license shown on the Kaggle dataset page and follow its terms before redistributing the CSV. The project does not assume permission to redistribute it.
- **Observed file:** 2,000 rows and 8 columns; no missing cells in the initial inspection.
- **Columns:** `Student_ID` (identifier; exclude from model inputs), `Study_Hours_Per_Day`, `Extracurricular_Hours_Per_Day`, `Sleep_Hours_Per_Day`, `Social_Hours_Per_Day`, `Physical_Activity_Hours_Per_Day`, `GPA`, and `Stress_Level` (candidate label).
- **Observed label counts:** High 1,029; Moderate 674; Low 297.
- **Important caveat:** The dataset page says stress level is derived from study and sleep hours. In the downloaded file, the five daily-hour fields sum to 24 hours per row when accounting for decimal rounding. These observations raise provenance and label-construction concerns; they do not establish whether the file is synthetic. Treat it as a limited prototype dataset, not evidence of real-world or clinically meaningful wellness-risk prediction.

Any synthetic data used later must be explicitly labeled synthetic and must not be presented as real student outcomes.

## Initial model comparison

The first training run used an 80/20 stratified train/test split (`random_state=42`). Study and sleep hours were omitted from model inputs because the dataset page says those values are used to derive the target label. The remaining features were extracurricular hours, social hours, physical activity hours, and GPA.

Four scikit-learn models were compared using five-fold cross-validation on the training split. Random Forest had the best mean cross-validation macro F1 (0.594), so it was selected. On the held-out test split it achieved 0.595 accuracy, 0.549 balanced accuracy, and 0.540 macro F1. Its recall for the Low class was 0.46, so that class is still often missed.

These are observed results for this particular dataset and split, not a claim about real student wellness prediction. They only describe how well the selected model reproduces this dataset's `Stress_Level` labels. `models/feature_importance.csv` contains permutation importance estimates; these describe the model's behavior and do not show causation.

## Streamlit prototype

The app accepts extracurricular hours, social hours, physical activity hours, and GPA. It displays the predicted dataset category, general non-medical suggestions, held-out evaluation metrics, and global permutation importance. It clearly states that the model is not a diagnosis or a validated personal wellness-risk estimate.

The prototype is for learning and awareness only. Predictions are not diagnoses, medical advice, or validated individual risk estimates. Supportive suggestions will remain general and non-medical.
