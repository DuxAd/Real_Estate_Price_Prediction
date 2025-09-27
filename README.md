# Real_Estate_Price_Prediction
Real Estate Price Prediction

## Overview
This project predicts house prices using the "House Prices - Advanced Regression Techniques" dataset fomr Kaggle (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

### Key Features
- **Data Analysis**: Skewness analysis, Pearson correlations, Chi² tests for feature selection.
- **Preprocessing**: Creation of a Pipeline of fill the missing values, apply scaler and encode categorical features.
- **Modeling**: LinearRegression, RandomForestRegressor, and GradientBoostingRegressor with `GridSearchCV` for hyperparameter tuning.
- **Evaluation**: Metrics include MSE, RMSE (dollars), R² (log scale), and cross-validation scores.

## Results

| Model                | RMSE      | R2      | RMSE(CV)           | 
|----------------------|-----------|---------|--------------------|
| Regression           | 26794.40  | 0.91048 | 31866.13 ± 7131.88 |
| RandomForest         | 30136.86  | 0.88391 | 31027.92 ± 3621.89 |
| GradientBoosting     | 28875.28  | 0.89594 | 27550.86 ± 4931.23 |

## Key Observations
LinearRegression achieves the lowest RMSE (26,794 $) and highest R² (0.910).
GradientBoosting is the most robust in cross-validation (RMSE: 27,550 $).
Top features: OverallQual, GrLivArea, YearBuilt, TotalBsmtSF, GarageCars and ExterQual_TA.
Applying a Log transformation on the SalePrice and numerical features improved model performance.


