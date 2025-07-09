import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# Pastikan folder model ada
os.makedirs('../model', exist_ok=True)

# Load data
df = pd.read_csv('../data/HRDataset_v14.csv')

# Feature engineering
now = pd.to_datetime('today')
df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
df['DateofHire'] = pd.to_datetime(df['DateofHire'], errors='coerce')
df['Age'] = (now - df['DOB']).dt.days // 365
df['Tenure'] = (now - df['DateofHire']).dt.days // 365
df['Absences_per_year'] = df['Absences'] / (df['Tenure'].replace(0, 1))
df['Late_per_year'] = df['DaysLateLast30'] / (df['Tenure'].replace(0, 1))
df['EngagementSalary'] = df['EngagementSurvey'] * df['Salary']

# Target: Top 40% performer & kepuasan
perf_q6 = df['PerfScoreID'].quantile(0.6)
satis_q6 = df['EmpSatisfaction'].quantile(0.6)
df['TopPerformer'] = ((df['PerfScoreID'] >= perf_q6) & (df['EmpSatisfaction'] >= satis_q6)).astype(int)

# Fitur input (tanpa PerfScoreID & EmpSatisfaction)
fitur_aman = [
    'Age', 'GenderID', 'MaritalStatusID', 'DeptID', 'Salary', 'Tenure',
    'EngagementSurvey', 'SpecialProjectsCount', 'DaysLateLast30',
    'Absences_per_year', 'Late_per_year', 'EngagementSalary'
]
categorical_cols = []
for col in ['Sex', 'Department', 'Position', 'State', 'RaceDesc', 'MaritalDesc']:
    if col in df.columns:
        categorical_cols.append(col)

df_model = pd.get_dummies(df[fitur_aman + categorical_cols + ['TopPerformer']], columns=categorical_cols, drop_first=True)
X = df_model.drop(columns=['TopPerformer'])
y = df_model['TopPerformer']
X = X.fillna(X.median())

# Simpan urutan kolom fitur
feature_columns = [col for col in df_model.columns if col != 'TopPerformer']
joblib.dump(feature_columns, '../model/feature_columns.pkl')

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection
selector = SelectKBest(f_classif, k=min(15, X.shape[1]))
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X.columns[selector.get_support(indices=True)]
joblib.dump(selected_features, '../model/selected_features.pkl')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Train models
logreg = LogisticRegression(max_iter=2000, class_weight='balanced')
logreg.fit(X_train_bal, y_train_bal)
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_bal, y_train_bal)

# Save models and pipeline
joblib.dump(logreg, '../model/logreg_model.pkl')
joblib.dump(rf, '../model/rf_model.pkl')
joblib.dump(scaler, '../model/scaler.pkl')
joblib.dump(selector, '../model/selector.pkl')
