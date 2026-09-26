# 💰 Salary Prediction Model

A machine learning web application that predicts salary based on age, years of experience, education level, gender, and job title.

The application is built with **Python, Scikit-learn, Pandas, Plotly, and Streamlit**.

## 📌 Project Overview

This project uses a salary dataset to train a machine learning regression model and provides an interactive Streamlit dashboard.

Users can enter their:

* Age
* Years of Experience
* Gender
* Education Level
* Job Title

and receive an estimated salary prediction.

## 📊 Dashboard

The Streamlit app includes:

* Salary distribution
* Experience vs Salary visualization
* Average salary by education level
* Average salary by gender
* Average salary by job title
* Interactive salary prediction

## 🤖 Machine Learning

### Model

Random Forest Regressor

### Features

* Age
* Years of Experience
* Gender
* Education Level
* Job Title

### Target

Salary

Categorical variables are encoded using `OneHotEncoder`, and the preprocessing and model are saved together as a Scikit-learn pipeline.

## 📈 Model Evaluation

The model was evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

Because the original dataset contained many duplicate records, duplicate rows were removed before training the final model.

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Plotly
* Streamlit
* Joblib

## 📁 Project Structure

```text
salary-prediction-streamlit/
│
├── app.py
├── Salary_Data.csv
├── salary_prediction_model.pkl
├── requirements.txt
├── README.md
```

## ⚠️ Disclaimer

The predictions are estimates based on patterns learned from the provided dataset. They should not be treated as guaranteed real-world salaries.

## 👩‍💻 Author

Mubeen Shehzadi ||
Learning by building
