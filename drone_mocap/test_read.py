import numpy as np
import pandas as pd
from drone_mocap.io.mocap_txt import read_mocap_angles_txt
from drone_mocap.evaluation.compare_mocap import _interp_to, _shift_series, _corr

df_vid = pd.read_csv(r'runs/20260321_233339/derived/angles_sagittal.csv')
df_moc = read_mocap_angles_txt(r'C:\Users\wassi\ENGG 502\Treadmill Angle Data\Raghav_run1-LOWER_BILATERAL_angles_simple.txt')

vtime = df_vid['time_s'].to_numpy(float)
mtime = df_moc['time'].to_numpy(float)

v_knee  = df_vid['right_knee_deg'].to_numpy(float)
v_hip   = df_vid['right_hip_deg'].to_numpy(float)
v_ankle = df_vid['right_ankle_deg'].to_numpy(float)

m_knee  = _interp_to(mtime, df_moc['R_KNEE_angle_Z'].to_numpy(float), vtime)
m_hip   = _interp_to(mtime, df_moc['R_HIP_angle_Z'].to_numpy(float), vtime)
m_ankle = _interp_to(mtime, df_moc['R_ANKLE_angle_Z'].to_numpy(float), vtime)

print("At shift 139 (knee-derived):")
print(f'  Knee:  r={_corr(v_knee,  _shift_series(-m_knee,  139)):.3f}')
print(f'  Hip:   r={_corr(v_hip,   _shift_series(m_hip,    139)):.3f}')
print(f'  Ankle: r={_corr(v_ankle, _shift_series(m_ankle,  139)):.3f}')

print("\nAt shift 264 (ankle-derived):")
print(f'  Knee:  r={_corr(v_knee,  _shift_series(-m_knee,  264)):.3f}')
print(f'  Hip:   r={_corr(v_hip,   _shift_series(m_hip,    264)):.3f}')
print(f'  Ankle: r={_corr(v_ankle, _shift_series(m_ankle,  264)):.3f}')