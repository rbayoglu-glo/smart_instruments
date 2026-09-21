#%%

import pandas as pd


path = 'C:\\Projects\\Smart_Instruments\\Lamina_Spreader_Lab\\Lab_09-04_2026\\tracking_data\\2026-11-08_[13-50-13]\\ndi\\P9-26610\\NDI_Localizer.txt'

df = pd.read_csv(path, sep=',', skiprows=4)

# print(df.head())
# print(df.shape)
# print(df.columns.tolist())


