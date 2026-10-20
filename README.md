# Indian Street Food Body Damage Predictor

click here for demo -->> "https://street-food-body-damage-predictor.streamlit.app/"

A beginner-friendly Machine Learning project that uses **Linear Regression** to predict the dataset's `Annual_Body_Damage_Percentage` from the entered consumption counts of selected Indian street foods.

> **Important:** This is an educational prediction project, not a medically validated measure of body damage or health risk. Its predictions depend on the dataset and should not be used for medical or dietary decisions.

## Project Overview

The notebook loads a dataset of 50,000 rows, explores and cleans the data, trains a Linear Regression model, evaluates its predictions, visualizes results, and lets a user enter food-consumption values to get a predicted target value.

### Food features used

- Momo
- Pizza
- Samosa
- Pani Puri
- French Fries
- Soya Bari
- Burger
- Patties
- Chowmein
- Dosa

**Target column:** `Annual_Body_Damage_Percentage`

The dataset's target is treated as a percentage by the notebook. Its real-world meaning and validity are not established by the notebook itself.

## Features

- Dataset inspection and summary statistics
- Missing-value and duplicate checks
- Data cleaning using `dropna()` and `drop_duplicates()`
- Train/test split (80% training, 20% testing)
- Linear Regression model training
- Predictions compared with actual target values
- Evaluation with MAE, MSE, RMSE, and R²
- Actual-vs-predicted scatter plot
- Feature-coefficient table and chart
- Interactive console input for food counts and prediction

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Model Evaluation

The notebook's recorded evaluation results are:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 1.6006 |
| Mean Squared Error (MSE) | 4.0049 |
| Root Mean Squared Error (RMSE) | 2.0012 |
| R² Score | 0.9599 |

These metrics describe performance on the notebook's test split. A high R² score does **not** establish that the dataset's target is medically meaningful or that the model can reliably predict actual health outcomes.

## Project Structure

```text
project/
├── indian_food_body_damage.ipynb
├── indian_street_food_dataset.csv
└── README.md
```

Keep the CSV file in the same folder as the notebook, because the notebook loads it using:

```python
pd.read_csv("indian_street_food_dataset.csv")
```

## Installation and Setup

1. Install Python (3.9 or newer recommended).
2. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib scikit-learn jupyter
   ```

3. Place `indian_street_food_dataset.csv` in the same folder as the notebook.
4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open the notebook and run the cells from top to bottom.

## How to Use

1. Run the notebook cells in order to load and prepare the dataset.
2. Train the Linear Regression model.
3. Review the evaluation metrics and plots.
4. In the final cell, enter a value for each food item when prompted.
5. The model prints the predicted `Annual_Body_Damage_Percentage`.

The input prompts in the current notebook show ranges such as `0–30`; use values within the intended range of the dataset and do not interpret the output as a medical assessment.

## Machine Learning Workflow

```text
Dataset (CSV)
     ↓
Explore and clean data
     ↓
Select input features (X) and target (y)
     ↓
Split into training and testing sets
     ↓
Train Linear Regression
     ↓
Predict test-set values
     ↓
Evaluate metrics and visualize results
     ↓
Enter food counts and generate a prediction
```

## Limitations

- The project uses a dataset target named `Annual_Body_Damage_Percentage`; the notebook does not establish how that target was measured or whether it represents real biological damage.
- Linear Regression learns patterns from the supplied data and may not reflect real-world health relationships.
- The model uses only the ten listed food-count features; it does not account for portion size, nutrition, lifestyle, medical history, or other factors.
- The notebook expects a local CSV file that must be provided separately.
- The project is a learning/demo prototype, not a healthcare tool.

## Future Improvements

- Document the dataset's source and how the target was created.
- Validate the data against reliable, real-world evidence before making health-related claims.
- Add input validation and a more user-friendly interface.
- Compare Linear Regression with other appropriate models.
- Save the trained model and add automated tests.

## Author

**Abhay Singh**

---

If you find this project useful for learning, feel free to fork it and experiment with the workflow. Please retain the educational and non-medical disclaimer when sharing it.
