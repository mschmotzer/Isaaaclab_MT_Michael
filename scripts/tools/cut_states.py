import h5py
import shutil
import os
import sys

src_file = "/home/pdz/MasterThesis_MSC/IsaacLab/datasets/Datasets_benchmarking/annotated_RGB_data_new_obs_nostates_new_obs.hdf5"   # <-- put your real file here
base, ext = os.path.splitext(src_file)
output_path = f"{base}_nostatesnoobs{ext}"

# Copy original file to new file first
import shutil
shutil.copy2(src_file, output_path)

print(f"Created new file: {output_path}")
# -------------------------
# 3. Try opening destination
# -------------------------
print("Opening copied file for editing...")
f = h5py.File(output_path, "r+")

# -------------------------
# 4. Remove all states groups
# -------------------------
data_group = f["data"]

for demo in list(data_group.keys()):
    demo_group = data_group[demo]
    if "states" in demo_group:
        print(f"Deleting states in: {demo}")
        del demo_group["states"]
    if "obs" in demo_group:
        print(f"Deleting obs in: {demo}")
        del demo_group["obs"]

f.close()

print("\n✔ Done! Clean file saved as:", output_path)
