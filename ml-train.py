import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Φόρτωση δεδομένων
df = pd.read_csv('Housing.csv', encoding='utf-8')

print("Πρώτες γραμμές:")
print(df.head())
print("\nΠληροφορίες δεδομένων:")
print(df.info())
print("\nΣτατιστικά:")
print(df.describe())

# Αφαίρεση γραμμών με κενή τιμή
df = df.dropna(subset=['price'])

# Διαχωρισμός X και y
X = df.drop('price', axis=1)
y = df['price']


# Μετατροπή κατηγορικών σε dummy variables
X = pd.get_dummies(X, drop_first=True)

print(f"\nΑριθμός χαρακτηριστικών: {X.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Μοντέλο Random Forest 
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
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
print(f"Train R²: {train_r2:.3f}")
print(f"Test MAE: {test_mae:,.2f}")
print(f"Test R²: {test_r2:.3f}")
print("="*50)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 σημαντικότερα χαρακτηριστικά:")
print(feature_importance.head(10))