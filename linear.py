
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
from statsmodels.stats.stattools import durbin_watson

# --- Φόρτωση δεδομένων ---
df = pd.read_csv('Housing.csv')
df = df.dropna()


# ΑΦΑΙΡΕΣΗ OUTLIERS (με βάση την τιμή - target variable)

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Κρατάμε μόνο τις "κανονικές" τιμές
df = df[(df['price'] >= lower) & (df['price'] <= upper)]

print(f"Δεδομένα μετά την αφαίρεση outliers: {len(df)} γραμμές")


# ΠΡΟΕΤΟΙΜΑΣΙΑ ΜΟΝΤΕΛΟΥ


# Διαχωρισμός X και y
X = df.drop('price', axis=1)
y = df['price']

# Εντοπισμός κατηγορικών στηλών
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# OneHot Encoding
encoder = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)],
    remainder='passthrough'
)
X_encoded = encoder.fit_transform(X)

# Διαχωρισμός σε train/test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Fit μοντέλου
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Υπολογισμός residuals
residuals = y_test - y_pred
fitted = y_pred

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Φόρτωση δεδομένων 
df = pd.read_csv('Housing.csv', encoding='utf-8')

print(f"Δεδομένα: {len(df)} γραμμές")

# Αφαίρεση missing values
df = df.dropna()

print(df.nunique())

#  Διαχωρισμός X και y 
X = df.drop('price', axis=1)
y = df['price']

#  Train-test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Εντοπισμός κατηγορικών στηλών 
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

#  OneHotEncoder 
encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

# Fit στο train, transform και στα δύο
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

print(f"Features: {X_train.shape[1]}")

# Linear Regression Model 
model = LinearRegression()
model.fit(X_train, y_train)

#  Προβλέψεις 
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

#  Αξιολόγηση 
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print("\n" + "="*50)
print("ΑΠΟΤΕΛΕΣΜΑΤΑ - LINEAR REGRESSION")
print("="*50)
print(f"Train MAE: {train_mae:,.2f}")
print(f"Train R²:  {train_r2:.4f}")
print(f"Test MAE:  {test_mae:,.2f}")
print(f"Test R²:   {test_r2:.4f}")
print(f"Gap R²:    {train_r2 - test_r2:.4f}")
print("="*50)

#  Πίνακας πραγματικών vs προβλεπόμενων τιμών (Test set)
df_compare = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred_test
})

print(df_compare.head(10))