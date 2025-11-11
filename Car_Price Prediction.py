import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load and explore data
print("Loading dataset...")
df = pd.read_csv(r'C:\Users\PRAKRATHI\Desktop\OASIS INFOBYTE Internship\DataScience\3. car data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Enhanced preprocessing
print("\nPreprocessing data...")

# 1. Handle null and missing values
print("\n1. Checking for null and missing values...")
print("Null values:\n", df.isnull().sum())

if df.isnull().sum().any():
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

# 2. Clean categorical data
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].str.strip()
    if (df[col] == '').any():
        mode_val = df[col][df[col] != ''].mode()[0]
        df[col] = df[col].replace('', mode_val)

# 3. Remove duplicates
df = df.drop_duplicates()

# 4. Enhanced Feature Engineering
print("\nEngineering features...")

# Age features
df['Car_Age'] = 2024 - df['Year']
df['Age_Squared'] = df['Car_Age'] ** 2

# Price per km ratio
df['Price_per_KM'] = df['Present_Price'] / (df['Driven_kms'] + 1)

# Mileage categories
df['High_Mileage'] = (df['Driven_kms'] > df['Driven_kms'].quantile(0.75)).astype(int)

# Age categories
df['New_Car'] = (df['Car_Age'] <= 3).astype(int)
df['Old_Car'] = (df['Car_Age'] > 10).astype(int)

# Price categories
df['Luxury_Car'] = (df['Present_Price'] > df['Present_Price'].quantile(0.75)).astype(int)

print(f"Features engineered successfully")

# 5. Handle outliers
def remove_outliers(df, column, n_std=3):
    mean = df[column].mean()
    std = df[column].std()
    df = df[(df[column] >= mean - n_std * std) & (df[column] <= mean + n_std * std)]
    return df

initial_len = len(df)
df = remove_outliers(df, 'Selling_Price')
df = remove_outliers(df, 'Driven_kms')
print(f"\nRemoved {initial_len - len(df)} outliers")

print("\nData validation summary:")
print(f"Final dataset shape: {df.shape}")
print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
print(f"Price range: ₹{df['Selling_Price'].min():.2f} - ₹{df['Selling_Price'].max():.2f} Lakhs")

# Prepare features for modeling
print("\nPreparing features for modeling...")
df_model = df.drop(['Car_Name'], axis=1)

# Encode categorical variables
le_fuel = LabelEncoder()
le_seller = LabelEncoder() 
le_transmission = LabelEncoder()

df_model['Fuel_Type'] = le_fuel.fit_transform(df_model['Fuel_Type'])
df_model['Selling_type'] = le_seller.fit_transform(df_model['Selling_type'])
df_model['Transmission'] = le_transmission.fit_transform(df_model['Transmission'])

print("\nEncoding mappings:")
print(f"Fuel Type: {dict(zip(le_fuel.classes_, range(len(le_fuel.classes_))))}")
print(f"Seller Type: {dict(zip(le_seller.classes_, range(len(le_seller.classes_))))}")
print(f"Transmission: {dict(zip(le_transmission.classes_, range(len(le_transmission.classes_))))}")

# Split data
X = df_model.drop('Selling_Price', axis=1)
y = df_model['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

# Train Random Forest with hyperparameter tuning
print("\n" + "="*50)
print("TRAINING OPTIMIZED RANDOM FOREST MODEL")
print("="*50)

rf_params = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("\nPerforming GridSearch for best hyperparameters...")
rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='r2', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)

model = rf_grid.best_estimator_
print(f"\nBest parameters found: {rf_grid.best_params_}")

# Evaluate model
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²:  {test_r2:.4f}")
print(f"RMSE:     {rmse:.4f} lakhs")
print(f"MAE:      {mae:.4f} lakhs")

overfit = train_r2 - test_r2
if overfit > 0.1:
    print(f"Overfitting detected: {overfit:.4f}")
else:
    print(f"Good generalization: {overfit:.4f}")

# Enhanced visualizations
fig = plt.figure(figsize=(15, 5))

# 1. Actual vs Predicted
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price (Lakhs)', fontsize=11)
plt.ylabel('Predicted Price (Lakhs)', fontsize=11)
plt.title(f'Actual vs Predicted\nR² = {test_r2:.4f}', fontsize=12, fontweight='bold')
plt.grid(alpha=0.3)

# 2. Feature Importance
plt.subplot(1, 3, 2)
feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
feature_imp.tail(10).plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
plt.xlabel('Importance', fontsize=11)

# 3. Residuals plot
plt.subplot(1, 3, 3)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price (Lakhs)', fontsize=11)
plt.ylabel('Residuals', fontsize=11)
plt.title('Residual Plot', fontsize=12, fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Enhanced prediction function with user input
def predict_car_price():
    print("\n" + "="*60)
    print("CAR PRICE PREDICTION - USER INPUT MODE")
    print("="*60)
    
    while True:
        try:
            print("\nEnter car details:")
            print("-" * 60)
            
            year = int(input("Year of manufacture: "))
            present_price = float(input("Current showroom price (lakhs): "))
            kms = int(input("Kilometers driven: "))
            
            print(f"\nFuel type options: {', '.join(le_fuel.classes_)}")
            fuel = input("Fuel type: ").strip().title()
            
            print(f"\nSeller type options: {', '.join(le_seller.classes_)}")
            seller = input("Seller type: ").strip().title()
            
            print(f"\nTransmission options: {', '.join(le_transmission.classes_)}")
            transmission = input("Transmission: ").strip().title()
            
            owner = int(input("Number of previous owners: "))
            
            # Validate inputs
            if year < 1900 or year > 2024:
                print("Error: Year must be between 1900 and 2024")
                continue
            if present_price <= 0 or kms < 0 or owner < 0:
                print("Error: Please enter valid positive numbers")
                continue
            
            # Feature engineering
            car_age = 2024 - year
            age_squared = car_age ** 2
            price_per_km = present_price / (kms + 1)
            high_mileage = 1 if kms > df['Driven_kms'].quantile(0.75) else 0
            new_car = 1 if car_age <= 3 else 0
            old_car = 1 if car_age > 10 else 0
            luxury_car = 1 if present_price > df['Present_Price'].quantile(0.75) else 0
            
            # Encode
            fuel_enc = le_fuel.transform([fuel])[0]
            seller_enc = le_seller.transform([seller])[0]
            transmission_enc = le_transmission.transform([transmission])[0]
            
            # Create feature array
            features = np.array([[
                year, present_price, kms, fuel_enc, seller_enc, 
                transmission_enc, owner, car_age, age_squared,
                price_per_km, high_mileage, new_car, old_car, luxury_car
            ]])
            
            # Predict
            predicted = model.predict(features)[0]
            
            # Display results
            print("\n" + "="*60)
            print("PREDICTION RESULT")
            print("="*60)
            print(f"Model Accuracy (R²): {test_r2:.2%}")
            print(f"Model Error (MAE):   ±₹{mae:.2f} Lakhs")
            print(f"\nPresent Price:       ₹{present_price:.2f} Lakhs")
            print(f"Predicted Price:     ₹{predicted:.2f} Lakhs")
            
            if predicted < present_price:
                depreciation = present_price - predicted
                pct = (depreciation / present_price) * 100
                print(f"Depreciation:        ₹{depreciation:.2f} Lakhs ({pct:.1f}%)")
            else:
                appreciation = predicted - present_price
                print(f"Value Change:        +₹{appreciation:.2f} Lakhs")
            
            print(f"\nConfidence Range: ₹{max(0, predicted-mae):.2f} - ₹{predicted+mae:.2f} Lakhs")
            print("="*60)
            
            # Ask if user wants to predict another car
            another = input("\nPredict another car? (yes/no): ").strip().lower()
            if another not in ['yes', 'y']:
                break
                
        except ValueError:
            print("Input error: Please enter valid numbers/text")
        except Exception as e:
            print(f"Error: {e}")
            print("Please check your inputs and try again")
    
    print("\nThank you for using the Car Price Predictor! 👋")

# Run interactive prediction
predict_car_price()

print("\n Program completed!")