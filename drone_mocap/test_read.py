import pandas as pd
import numpy as np
from drone_mocap.io.mocap_txt import read_mocap_angles_txt
from drone_mocap.evaluation.compare_mocap import _interp_to, _shift_series, _corr
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df_vid = pd.read_csv(r'runs/20260321_220738/derived/angles_sagittal.csv')
df_moc = read_mocap_angles_txt(r'C:\Users\wassi\ENGG 502\Treadmill Angle Data\Raghav_run_L1-LOWER_BILATERAL_angles_simple.txt')

vtime = df_vid['time_s'].to_numpy(float)
mtime = df_moc['time'].to_numpy(float)

v_ankle = df_vid['left_ankle_deg'].to_numpy(float)
m_ankle_z = _interp_to(mtime, df_moc['L_ANKLE_angle_Z'].to_numpy(float), vtime)
m_ankle_z_shifted = _shift_series(m_ankle_z, 55)

fig, axes = plt.subplots(4, 1, figsize=(14, 12))
axes[0].plot(vtime, v_ankle, label='Video (current)', color='blue')
axes[0].plot(vtime, m_ankle_z_shifted, label='MoCap', color='red', linestyle='--')
axes[0].set_title(f'Current: r={_corr(v_ankle, m_ankle_z_shifted):.3f}')
axes[0].legend()
axes[0].grid(True)

for i, kernel in enumerate([13, 17, 21]):
    v_med = median_filter(v_ankle, size=kernel)
    r = _corr(v_med, m_ankle_z_shifted)
    axes[i+1].plot(vtime, v_med, label=f'Median kernel={kernel}', color='blue')
    axes[i+1].plot(vtime, m_ankle_z_shifted, label='MoCap', color='red', linestyle='--')
    axes[i+1].set_title(f'Median filter kernel={kernel}: r={r:.3f}')
    axes[i+1].legend()
    axes[i+1].grid(True)
    print(f'Median kernel={kernel}: r={r:.3f}')

plt.tight_layout()
plt.savefig('ankle_median_test2.png', dpi=150)
print('Saved ankle_median_test2.png')