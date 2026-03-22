import pandas as pd
import numpy as np
from drone_mocap.io.mocap_txt import read_mocap_angles_txt
from drone_mocap.evaluation.compare_mocap import _interp_to, _shift_series

df_vid = pd.read_csv(r'runs/20260321_212951/derived/angles_sagittal.csv')
df_moc = read_mocap_angles_txt(r'C:\Users\wassi\ENGG 502\Treadmill Angle Data\Raghav_run_L1-LOWER_BILATERAL_angles_simple.txt')

vtime = df_vid['time_s'].to_numpy(float)
mtime = df_moc['time'].to_numpy(float)

v_ankle = df_vid['left_ankle_deg'].to_numpy(float)
m_ankle_z = _interp_to(mtime, df_moc['L_ANKLE_angle_Z'].to_numpy(float), vtime)
m_ankle_z_shifted = _shift_series(m_ankle_z, 55)

mask = np.isfinite(v_ankle) & np.isfinite(m_ankle_z_shifted)
print(f'Video ankle mean:  {np.mean(v_ankle[mask]):.2f}')
print(f'MoCap ankle Z mean: {np.mean(m_ankle_z_shifted[mask]):.2f}')
print(f'Mean difference (video - mocap): {np.mean(v_ankle[mask] - m_ankle_z_shifted[mask]):.2f}')