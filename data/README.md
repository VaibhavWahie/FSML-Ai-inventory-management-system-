# Dataset Description

## Overview
This project uses the **Store Sales Time Series Forecasting dataset** to build a machine learning model for predicting future product demand. The goal is to forecast sales values 7 days ahead to support inventory management decisions such as stock planning and demand optimization.

---

## Source
Kaggle Dataset Link:  
https://www.kaggle.com/competitions/store-sales-time-series-forecasting

---

## Dataset Description
The dataset contains historical sales records from multiple stores and product categories. It captures daily sales along with promotional information, allowing the model to learn temporal patterns, seasonality, and demand trends.

---

## Key Columns

- **date**  
  Represents the date of the sales record. Used for time-series analysis and feature extraction.

- **store_nbr**  
  Unique identifier for each store.

- **family**  
  Product category (e.g., groceries, beverages, etc.). This is a categorical feature.

- **sales**  
  Target variable representing the total sales for a given product category at a specific store and date.

- **onpromotion**  
  Indicates the number of items on promotion. Helps capture the effect of promotions on sales.

---

## Data Preprocessing

The dataset undergoes the following preprocessing steps:

- Conversion of `date` column to datetime format
- Sorting data by store, product family, and date
- Handling missing values:
  - `sales` filled with 0
  - `onpromotion` filled with 0
- Encoding categorical variable `family`

---

## Feature Engineering

Advanced time-series features are created to improve model performance:

### Time-Based Features
- Day of week
- Month
- Week number
- Weekend indicator

### Lag Features
- Previous sales values (lag_1, lag_2, lag_3, lag_7, lag_14, lag_21, lag_28)

### Rolling Statistics
- Rolling mean (7 days, 14 days)
- Rolling standard deviation (7 days)

These features help capture trends, seasonality, and short-term fluctuations.

---

## Target Variable

The model predicts future sales using:

target = sales shifted by 7 days

This allows forecasting demand one week in advance.

---

## Notes

- The original dataset is not uploaded in the repository due to size constraints.
- Only a sample processed dataset is included.
- Feature engineering is carefully designed to avoid data leakage by using shifted values.