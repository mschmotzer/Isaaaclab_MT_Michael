#!/usr/bin/env python3
import h5py
import numpy as np
import argparse

def combine_datasets(file1, file2, out_file, episodes_per_dataset=25):
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        episodes1 = list(f1['data'].keys())
        episodes2 = list(f2['data'].keys())

        if len(episodes1) < episodes_per_dataset or len(episodes2) < episodes_per_dataset:
            raise ValueError("Not enough episodes in one of the datasets.")

        # Randomly sample
        selected1 = np.random.choice(episodes1, episodes_per_dataset, replace=False)
        selected2 = np.random.choice(episodes2, episodes_per_dataset, replace=False)

        with h5py.File(out_file, 'w') as fout:
            fout.create_group('data')

            # Copy episodes from dataset 1
            for i, ep in enumerate(selected1):
                new_name = f"episode_{i}"
                f1.copy(f"data/{ep}", fout['data'], name=new_name)

            # Copy episodes from dataset 2
            for i, ep in enumerate(selected2, start=episodes_per_dataset):
                new_name = f"episode_{i}"
                f2.copy(f"data/{ep}", fout['data'], name=new_name)

    print(f"✅ Combined dataset saved to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Combine two HDF5 datasets into one with 50 episodes (25 each).")
    parser.add_argument("--dataset1", type=str, help="Path to the first HDF5 dataset")
    parser.add_argument("--dataset2", type=str, help="Path to the second HDF5 dataset")
    parser.add_argument("--output", type=str, help="Path to save the combined dataset")
    parser.add_argument("--episodes", type=int, default=25, help="Number of episodes to sample from each dataset (default=25)")

    args = parser.parse_args()

    combine_datasets(args.dataset1, args.dataset2, args.output, episodes_per_dataset=args.episodes)

if __name__ == "__main__":
    main()
