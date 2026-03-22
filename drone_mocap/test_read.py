import pandas as pd

df = pd.read_csv(r'runs/20260321_212951/derived/angles_sagittal.csv')
print('Ankle NaN count:', df['left_ankle_deg'].isna().sum())
print('Ankle sample values:')
print(df['left_ankle_deg'].head(10))