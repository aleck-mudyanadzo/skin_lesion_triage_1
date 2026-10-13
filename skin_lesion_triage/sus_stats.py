import pandas as pd
df = pd.read_csv("survey_responses_export.csv")
print("n:", len(df))
print("mean:", round(df["sus_score"].mean(), 1))
print("median:", round(df["sus_score"].median(), 1))
print("std:", round(df["sus_score"].std(), 1))
