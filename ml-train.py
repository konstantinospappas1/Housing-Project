import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Φόρτωση 
df =pd.read_csv('Housing.csv', encoding='utf-8')

print(f"Δεδομένα: {len(df)} γραμμές")

# Αφαίρεση missing values
df = df.dropna()

# Διαχωρισμός X και y
X = df.drop('price', axis=1)
y = df['price']

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

print(f"Features: {X.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance.head(10).to_string(index=False))