import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load data dan pipeline
df = pd.read_csv('data/HRDataset_v14.csv')
logreg = joblib.load('../model/logreg_model.pkl')
scaler = joblib.load('../model/scaler.pkl')
selector = joblib.load('../model/selector.pkl')
selected_features = joblib.load('../model/selected_features.pkl')
feature_columns = joblib.load('../model/feature_columns.pkl')

# Fitur input dan kategorikal (harus sama dengan training)
fitur_aman = [
    'Age', 'GenderID', 'MaritalStatusID', 'DeptID', 'Salary', 'Tenure',
    'EngagementSurvey', 'SpecialProjectsCount', 'DaysLateLast30',
    'Absences_per_year', 'Late_per_year', 'EngagementSalary'
]
categorical_cols = []
for col in ['Sex', 'Department', 'Position', 'State', 'RaceDesc', 'MaritalDesc']:
    if col in df.columns:
        categorical_cols.append(col)

def preprocess_row(row):
    now = pd.to_datetime('today')
    row['DOB'] = pd.to_datetime(row['DOB'], errors='coerce')
    row['DateofHire'] = pd.to_datetime(row['DateofHire'], errors='coerce')
    row['Age'] = (now - row['DOB']).days // 365 if pd.notnull(row['DOB']) else 0
    row['Tenure'] = (now - row['DateofHire']).days // 365 if pd.notnull(row['DateofHire']) else 0
    row['Absences_per_year'] = row['Absences'] / (row['Tenure'] if row['Tenure'] != 0 else 1)
    row['Late_per_year'] = row['DaysLateLast30'] / (row['Tenure'] if row['Tenure'] != 0 else 1)
    row['EngagementSalary'] = row['EngagementSurvey'] * row['Salary']
    X_row = pd.DataFrame([row[fitur_aman + categorical_cols]])
    X_row = pd.get_dummies(X_row)
    X_row = X_row.reindex(columns=feature_columns, fill_value=0)
    X_row = X_row.fillna(X_row.median())
    X_scaled = scaler.transform(X_row)
    X_selected = selector.transform(X_scaled)
    return X_selected

st.title("Prediksi Top Performer Karyawan")
emp_id = st.text_input("Masukkan EmpID karyawan:")

if emp_id:
    emp_row = df[df['EmpID'].astype(str) == emp_id]
    if not emp_row.empty:
        nama = emp_row['Employee_Name'].values[0]
        st.write(f"**Nama Karyawan:** {nama}")
        X_input = preprocess_row(emp_row.iloc[0])
        pred = logreg.predict(X_input)[0]
        proba = logreg.predict_proba(X_input)[0][1]
        hasil = "Top Performer" if pred == 1 else "Bukan Top Performer"
        st.write(f"**Hasil Prediksi:** {hasil} (Probabilitas: {proba:.2f})")
        st.subheader("Feature Importance untuk Karyawan Ini")
        importances = logreg.coef_[0]
        fi_df = pd.DataFrame({
            'Feature': list(selected_features),
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        st.table(fi_df)
    else:
        st.warning("EmpID tidak ditemukan di data.")
