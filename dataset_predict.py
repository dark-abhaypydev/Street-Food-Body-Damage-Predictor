import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 50000

print('Generating 50,000 rows of Updated Indian Fast Food Dataset...')

# Momo sabse upar (1st feature)
data = {
    'Momo': np.random.randint(0, 21, size=n_samples),
    'Pizza': np.random.randint(0, 11, size=n_samples),
    'Samosa': np.random.randint(0, 16, size=n_samples),
    'Pani_Puri': np.random.randint(0, 21, size=n_samples),
    'French_Fries': np.random.randint(0, 21, size=n_samples),
    'Soya_Bari': np.random.randint(0, 16, size=n_samples),
    'Burger': np.random.randint(0, 16, size=n_samples),
    'Patties': np.random.randint(0, 16, size=n_samples),
    'Chowmein': np.random.randint(0, 16, size=n_samples),
    'Dosa': np.random.randint(0, 16, size=n_samples),
}

df = pd.DataFrame(data)

# Weights (Momo ko thoda special heavy weight diya hai health impact ke liye)
weights = np.array([3.4, 3.3, 3.5, 2.2, 2.5, 2.7, 2.8, 3.1, 3.2, 2.6])

base_damage = np.dot(df.values, weights)

max_possible = (
    20 * 3.4
    + 10 * 3.3
    + 15 * 3.5
    + 20 * 2.2
    + 20 * 2.5
    + 15 * 2.7
    + 15 * 2.8
    + 15 * 3.1
    + 15 * 3.2
    + 15 * 2.6
)

risk_percentage = (base_damage / max_possible) * 100 + np.random.normal(
    0, 2.0, n_samples
)
df['Annual_Body_Damage_Percentage'] = np.clip(risk_percentage, 0, 100).round(2)

df.to_csv('indian_street_food_dataset.csv', index=False)
print("Success! 'indian_street_food_dataset.csv' generated successfully.")