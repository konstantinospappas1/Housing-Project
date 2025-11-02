import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import randint, uniform

#  1. Φόρτωση δεδομένων 
df = pd.read_csv('Housing.csv', encoding='utf-8')
df = df.dropna()
print(f"Δεδομένα: {len(df)} γραμμές")

# 2. Διαχωρισμός X και y 
X = df.drop('price', axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. OneHot Encoding 
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
encoder = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)],
    remainder='passthrough'
)

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
print(f"Features μετά το encoding: {X_train.shape[1]}")

#  4. Ορισμός βασικού μοντέλου 
xgb = XGBRegressor(random_state=42, n_jobs=-1)

#  5. Randomized Search 
param_dist = {
    'n_estimators': randint(5, 30),     
    'max_depth': randint(2, 8),            
    'learning_rate': uniform(0.01, 0.4),   
    'subsample': uniform(0.4, 0.6),        
    'colsample_bytree': uniform(0.5, 1), 
    'min_child_weight': randint(1, 10)     
}

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=200,     # δοκιμή  συνδυασμών      
    scoring='r2',
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nΞεκινά το Randomized Search...")
random_search.fit(X_train, y_train)
print("\n Τέλος αναζήτησης")

#  6. Αποτελέσματα 
print(" Καλυτερες Παραμετροι του randomized search ")
print("="*60)
print(random_search.best_params_)
print(f"Μέσο R² από CV: {random_search.best_score_:.4f}")

#  7. Εκπαίδευση με τις καλύτερες παραμέτρους 
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)

#  8. Αξιολόγηση
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)


print(" Αποτελέσματα")
print("="*60)
print(f"Train MAE: {train_mae:,.2f}")
print(f"Test MAE:  {test_mae:,.2f}")
print(f"Train R²:  {train_r2:.4f}")
print(f"Test R²:   {test_r2:.4f}")
print("="*60)

#  9. Παράδειγμα προβλέψεων 
df_compare = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred_test
})
print("\nΠαράδειγμα προβλέψεων:")
print(df_compare.head(10))
