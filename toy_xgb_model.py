import numpy as np
import pandas as pd
import xgboost as xgb

# Create a toy dataset

# Seed the random number generator
np.random.seed(42)

n_rows = 200

chronological_age = np.random.randint(18, 30, n_rows)
rhr = np.random.randint(50, 90, n_rows)
hrv = np.random.randint(30, 90, n_rows)
sleep_score = np.random.uniform(5, 9, n_rows)

noise = np.random.normal(0, 1, n_rows)
biological_age = chronological_age + (rhr - 65) * 0.1 - (hrv - 60) * 0.05 - (
    sleep_score - 7) * 0.5 + noise

df = pd.DataFrame({
    'chronological_age': chronological_age,
    'rhr': rhr,
    'hrv': hrv,
    'sleep_score': sleep_score,
})

# train the model

model = xgb.XGBRegressor(objective='reg:squarederror',
                         n_estimators=100,
                         random_state=42,
                         max_depth=3)

model.fit(df, biological_age)

model.save_model("toy_xgb_model.json")
print("Toy model trained and saved successfully")
