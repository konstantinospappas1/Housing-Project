import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt 

# Φόρτωση δεδομένων
df = pd.read_csv('Housing.csv', encoding='utf-8')

print(f"Δεδομένα: {len(df)} γραμμές")

# Αφαίρεση missing values
df = df.dropna()

# Διαχωρισμός X και y
X = df.drop('price', axis=1)
y = df['price']

# Train-test split ΠΡΙΝ το encoding
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Εντοπισμός κατηγορικών στηλών
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# OneHotEncoder
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

# XGBoost μοντέλο
model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# Προβλέψεις
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Αξιολόγηση
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print("\n" + "="*50)
print("ΑΠΟΤΕΛΕΣΜΑΤΑ")
print("="*50)
print(f"Train MAE: {train_mae:,.2f}")
print(f"Train R²:  {train_r2:.4f}")
print(f"Test MAE:  {test_mae:,.2f}")
print(f"Test R²:   {test_r2:.4f}")
print(f"Gap R²:    {train_r2 - test_r2:.4f}")
print("="*50)

import pandas as pd

# Για το test set
df_compare = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred_test
})

print(df_compare.head(10))  
