#!/usr/bin/env python
# coding: utf-8

import h5py
import pandas as pd

# ### Pixel differences

pixel_diff_file = "Files_Alessio/Skript_Test/5086952_1962_M2_pixel_diff.h5"
store = h5py.File(pixel_diff_file)
print(store.keys())
print(store["pixel_differences"].keys())
store.close()
df = pd.read_hdf(pixel_diff_file, key="pixel_differences")
print(df)

# ### All pixel information

pixel_info_file = "Files_Alessio/Skript_Test/5086952_1962_M2_pixel_info.h5"
store = h5py.File(pixel_info_file)
print(store.keys())
print(store["pixel_information"].keys())
store.close()
df = pd.read_hdf(pixel_info_file, key="pixel_information")
print(df)
