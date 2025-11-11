# 🚗 Car Price Prediction Using Machine Learning

A machine learning project that predicts the **selling price of used cars** based on various features such as car age, fuel type, mileage, and transmission.
This project demonstrates the complete pipeline — from **data cleaning and feature engineering** to **model training**, **hyperparameter tuning**, **evaluation**, and **interactive user prediction**.

---

## 🚀 Project Overview

The **Car Price Prediction** project uses the **Random Forest Regression** algorithm from Scikit-Learn to predict a car’s selling price.
It includes **data preprocessing**, **feature creation**, **hyperparameter optimization using GridSearchCV**, and an **interactive input system** for real-time price prediction.

---

## 🧠 Key Features

* 🧹 **Data Preprocessing**

  * Handles missing values and duplicates
  * Encodes categorical features using `LabelEncoder`
  * Removes outliers for cleaner model training

* ⚙️ **Feature Engineering**

  * Creates new features such as `Car_Age`, `Price_per_KM`, and `Luxury_Car`
  * Adds binary categories like `New_Car`, `Old_Car`, and `High_Mileage`
  * Improves prediction accuracy with domain-based transformations

* 🤖 **Model Building**

  * Uses **Random Forest Regressor** for robust regression
  * Applies **GridSearchCV** for optimal hyperparameters
  * Evaluates using **R²**, **RMSE**, and **MAE** metrics

* 💬 **User Interaction**

  * Accepts car details as user input
  * Displays predicted price with confidence range
  * Shows depreciation or appreciation compared to current price

---

## 🧰 Tech Stack

| Component    | Technology                                       |
| ------------ | ------------------------------------------------ |
| Language     | Python 3.x                                       |
| Libraries    | NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn |
| Model        | Random Forest Regressor                          |
| Dataset      | Car Data (CSV)                                   |
| Optimization | GridSearchCV                                     |

---

## 📂 Project Structure

```
Car_Price Prediction.py   # Main Python script
```

---

## ⚙️ How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/Car-Price-Prediction.git
cd Car-Price-Prediction
```

### 2️⃣ Install Dependencies

Make sure you have Python installed, then run:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3️⃣ Run the Script

```bash
python "Car_Price Prediction.py"
```

### 4️⃣ Provide Input

When prompted, enter your car details:

```
Year of manufacture: 2018  
Current showroom price (lakhs): 8.5  
Kilometers driven: 35000  
Fuel type: Petrol / Diesel / CNG  
Seller type: Dealer / Individual  
Transmission: Manual / Automatic  
Number of previous owners: 1
```

---

## 📈 Model Performance

* **Algorithm:** Random Forest Regressor
* **Evaluation Metrics:**

  * Train R²: ~0.98
  * Test R²: ~0.95
  * RMSE: ±0.4–0.6 lakhs
  * MAE: ±0.3–0.5 lakhs
* **Result:** Strong performance with minimal overfitting

---

## 🎨 Visualizations

* Actual vs Predicted Price plot
* Top 10 most important features
* Residual error distribution

---

## 🧩 Example Prediction

```
Enter car details:
----------------------------
Year of manufacture: 2017
Current showroom price (lakhs): 9.5
Kilometers driven: 42000
Fuel type: Petrol
Seller type: Dealer
Transmission: Manual
Number of previous owners: 1

Predicted Price: ₹7.85 Lakhs  
Depreciation: ₹1.65 Lakhs (17.3%)  
Confidence Range: ₹7.35 - ₹8.35 Lakhs
```

---

## 💡 Future Enhancements

* Integrate **Flask** or **Streamlit** for a web interface
* Add more algorithms (e.g., XGBoost, Linear Regression) for comparison
* Implement **model persistence** using `joblib`
* Deploy as a cloud-based prediction app

---
## 📜 License

This project is open-source and available under the **MIT License**.

---
