import h5py
import shutil
import os
import sys

src_file = "/media/pdz/Elements1/merged.hdf5"   # <-- put your real file here
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
    """if "states" in demo_group:
        print(f"Deleting states in: {demo}")
        del demo_group["states"]"""
    if "obs" in demo_group:
        print(f"Clearing obs in: {demo}")
        obs_group = demo_group["obs"]
        for key in list(obs_group.keys()):
            del obs_group[key]


f.close()

print("\n✔ Done! Clean file saved as:", output_path)