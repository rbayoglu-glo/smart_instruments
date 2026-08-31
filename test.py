#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

file_path = r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Research_Team_Data\L231147_L5-S1\6DOF\Book4.xlsx"
df = pd.read_excel(file_path)
print(df.head())

#%
# Plot stiffness data: Mx (x-axis) vs Angle (y-axis) for flexion
fig, ax = plt.subplots(figsize=(8, 6))

# ax.plot(df['Mx (Nm)'], df['Angle (deg)'], marker='o', markersize=3, linewidth=1.5)
ax.plot(df.iloc[612:-1, 5], df.iloc[612:-1, 1], marker='o', markersize=3, linewidth=1.5)
ax.set_xlabel('Mx (Nm)', fontsize=12)
ax.set_ylabel('Angle (°)', fontsize=12)
ax.set_title('Flexion Stiffness - L231147 L5-S1', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.axhline(0, color='k', linewidth=0.8)
ax.axvline(0, color='k', linewidth=0.8)

plt.tight_layout()
plt.show()

#%
# Lateral Bending stiffness: My vs Angle
file_path_lb = r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Research_Team_Data\L231147_L5-S1\6DOF\Book5.xlsx"
df_lb = pd.read_excel(file_path_lb)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df_lb.iloc[395:-1, 6], df_lb.iloc[395:-1, 1], marker='o', markersize=3, linewidth=1.5)
ax.set_xlabel('My (Nm)', fontsize=12)
ax.set_ylabel('Angle (°)', fontsize=12)
ax.set_title('Lateral Bending Stiffness - L231147 L5-S1', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.axhline(0, color='k', linewidth=0.8)
ax.axvline(0, color='k', linewidth=0.8)

plt.tight_layout()
plt.show()

#%
# Axial Rotation stiffness: Mz vs Angle
file_path_ar = r"C:\Projects\Smart_Instruments\Lamina_Spreader_Lab\Research_Team_Data\L231147_L5-S1\6DOF\Book6.xlsx"
df_ar = pd.read_excel(file_path_ar)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df_ar.iloc[174:-1, 7], df_ar.iloc[174:-1, 1], marker='o', markersize=3, linewidth=1.5)
ax.set_xlabel('Mz (Nm)', fontsize=12)
ax.set_ylabel('Angle (°)', fontsize=12)
ax.set_title('Axial Rotation Stiffness - L231147 L5-S1', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.axhline(0, color='k', linewidth=0.8)
ax.axvline(0, color='k', linewidth=0.8)

plt.tight_layout()
plt.show()

