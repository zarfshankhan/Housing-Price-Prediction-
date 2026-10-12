# California Housing Price Prediction Using Machine Learning

## 1. Project Introduction

The California Housing Price Prediction project uses Machine Learning to predict median house values based on housing characteristics such as median income, house age, total rooms, total bedrooms, population, households, and geographical location.

This project uses the California Housing Prices dataset and a Random Forest Regressor. It covers data cleaning, exploratory data analysis, visualization, feature selection, model training, performance evaluation, and prediction for a new housing sample.

## 2. Objectives

- Load and explore the California housing dataset.
- Identify missing values, infinite values, and duplicate records.
- Clean and prepare the data for Machine Learning.
- Visualize house-value distributions and feature relationships.
- Select six important numerical features.
- Train a Random Forest regression model.
- Evaluate the model using regression metrics.
- Compare actual and predicted house values.
- Analyze feature importance.
- Predict the value of a new housing sample.

## 3. Dataset Description

**Dataset Name:** California Housing Prices

**Dataset File:** `housing.csv`

**Dataset Source:** [California Housing Prices Dataset on Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

The dataset contains housing information for California districts. The standard Kaggle version contains approximately 20,640 records and 10 columns.

### Dataset Features

| Feature | Description |
|---|---|
| `longitude` | Geographic longitude |
| `latitude` | Geographic latitude |
| `housing_median_age` | Median age of houses |
| `total_rooms` | Total number of rooms |
| `total_bedrooms` | Total number of bedrooms |
| `population` | Total population |
| `households` | Number of households |
| `median_income` | Median household income |
| `ocean_proximity` | Category describing proximity to the ocean |
| `median_house_value` | Target variable for prediction |

**Target variable:** `median_house_value`

The model-training section excludes `ocean_proximity` because the current pipeline uses numerical features only.

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 5. Machine Learning Algorithm

### Random Forest Regressor

Random Forest Regressor is an ensemble learning algorithm that combines multiple decision trees to predict continuous numerical values, such as house prices.

The model uses the following configuration:

- Number of trees: 100
- Random state: 42
- Parallel processing: Enabled with `n_jobs=-1`

### Feature Selection

The project uses `SelectKBest` with `f_regression` to select six numerical features with the highest individual F-scores in relation to the target variable.

### Data Preprocessing

The project performs the following preprocessing operations:

1. Replace infinite values with missing values.
2. Remove exact duplicate records.
3. Separate input features and the target variable.
4. Split the dataset into training and testing sets.
5. Replace missing numerical feature values with their median.
6. Select six features using the feature-selection pipeline.

The dataset is split into 80% training data and 20% testing data.

## 6. Project Workflow

1. Install the required libraries.
2. Import Python libraries.
3. Load the `housing.csv` dataset.
4. Display the first five records.
5. Check dataset shape, column names, and data types.
6. Generate a statistical summary.
7. Check missing values and duplicate records.
8. Clean the dataset.
9. Visualize the distribution of house values.
10. Analyze relationships between income, rooms, population, and house values.
11. Generate a correlation heatmap.
12. Split the dataset into training and testing sets.
13. Train the Random Forest regression model.
14. Display the selected features.
15. Predict house values on the test dataset.
16. Evaluate model performance.
17. Plot actual versus predicted values.
18. Analyze feature importance.
19. Predict the value of a new housing sample.

## 7. Data Visualization

The project includes the following visualizations:

- **House Value Distribution:** Displays the distribution of median house values.
- **Income vs House Value:** Examines the relationship between median income and house values.
- **Average Rooms vs House Value:** Explores the relationship between average rooms per household and house values.
- **Population vs House Value:** Examines population in relation to house values.
- **Correlation Heatmap:** Shows correlations among numerical features.
- **Actual vs Predicted Values:** Compares actual values with model predictions.
- **Feature Importance:** Displays the relative importance of the selected features in the trained Random Forest.

## 8. Model Evaluation

The model is evaluated using four regression metrics.

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error measures the average absolute prediction error. |
| MSE | Mean Squared Error measures the average squared prediction error. |
| RMSE | Root Mean Squared Error measures prediction error in the target's original units. |
| R² Score | Measures how well the model explains variation in the target variable. |

Lower MAE, MSE, and RMSE values generally indicate smaller prediction errors. An R² score closer to 1 generally indicates a better fit.

The notebook prints the actual evaluation results after training. These results should be recorded from the executed notebook.

**Note:** The target values in this dataset are expressed in units of $100,000. The code multiplies MAE and RMSE by 100,000 to express these errors in dollars.

## 9. Installation

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 10. How to Run the Project

1. Clone the GitHub repository:

   ```bash
   git clone https://github.com/zarfshankhan/Housing-Price-Prediction-.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Housing-Price-Prediction-
   ```

3. Place `housing.csv` in the project directory.

4. Open the Jupyter Notebook containing the project code.

5. Run the notebook cells in order, from library installation through model evaluation and new-house prediction.

## 11. Expected Results

After running the notebook, the project produces:

- Dataset information and statistical summaries.
- Missing-value and duplicate-record checks.
- Housing feature visualizations.
- Six selected numerical features.
- A trained Random Forest regression model.
- Actual and predicted house-value comparisons.
- MAE, MSE, RMSE, and R² scores.
- A feature-importance table and chart.
- A sample predicted house value.

## 12. Conclusion

This project demonstrates how Machine Learning can be applied to California housing price prediction. It combines data preprocessing, exploratory data analysis, visualization, feature selection, Random Forest regression, and model evaluation in a single workflow.

The model learns patterns from housing data and uses selected numerical features to estimate median house values. Predictions are estimates based on historical data and should not be treated as guaranteed current market prices or professional property valuations.

## 13. Author

**Zarfshan Attiq Khan**

GitHub: [zarfshankhan](https://github.com/zarfshankhan)

