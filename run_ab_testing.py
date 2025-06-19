# run_ab_testing.py

import pandas as pd
from src.ab_testing_pipeline import run_all_tests

# Load your data
df = pd.read_csv("MachineLearningRating_v3.txt", sep='|', low_memory=False)

# Run all hypothesis tests
run_all_tests(df)
