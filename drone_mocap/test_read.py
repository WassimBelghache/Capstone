import pandas as pd
import numpy as np
from drone_mocap.io.mocap_txt import read_mocap_angles_txt
from drone_mocap.evaluation.compare_mocap import _corr, _interp_to, _shift_series

df_vid = pd.read_csv(r'runs/20260321_212951/derived/angles_sagittal.csv')
df_moc = read_mocap_angles_txt(r'C:\Users\wassi\ENGG 502\Treadmill Angle Data\Raghav_run_L1-LOWER_BILATERAL_angles_simple.txt')

vtime = df_vid['time_s'].to_numpy(float)
mtime = df_moc['time'].to_numpy(float)
v_hip = df_vid['left_hip_deg'].to_numpy(float)

# Check all hip axes before shifting
for axis in ['X', 'Y', 'Z']:
    m = _interp_to(mtime, df_moc[f'L_HIP_angle_{axis}'].to_numpy(float), vtime)
    print(f'Hip vs  L_HIP_{axis} (no shift): r={_corr(v_hip, m):.3f}')
    print(f'Hip vs -L_HIP_{axis} (no shift): r={_corr(v_hip, -m):.3f}')