import argparse
import h5py
import os

parser = argparse.ArgumentParser(description="Merge a set of HDF5 datasets.")
parser.add_argument(
    "--input_files",
    type=str,
    nargs="+",
    default=[],
    help="A list of paths to HDF5 files to merge.",
)
parser.add_argument(
    "--end",
    type=int,
    nargs="*",  # Changed from "+" to "*" to make it optional
    default=[],
    help="A list of end indices for each HDF5 file to merge. If not provided, all episodes are merged.",
)
parser.add_argument("--output_file", type=str, default="merged_dataset.hdf5", help="File path to merged output.")

args_cli = parser.parse_args()


def merge_datasets():
    for filepath in args_cli.input_files:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The dataset file {filepath} does not exist.")

    # Validate end indices
    if args_cli.end and len(args_cli.end) != len(args_cli.input_files):
        raise ValueError(f"Number of end indices ({len(args_cli.end)}) must match number of input files ({len(args_cli.input_files)})")

    with h5py.File(args_cli.output_file, "w") as output:
        episode_idx = 0
        copy_attributes = True

        for file_idx, filepath in enumerate(args_cli.input_files):
            # Get end index for this file (if provided)
            end_idx = args_cli.end[file_idx] if args_cli.end else None
            
            with h5py.File(filepath, "r") as input:
                episodes = list(input["data"].keys())
                
                # Sort episodes to ensure consistent ordering
                episodes.sort(key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
                
                # Apply end index limit if provided
                if end_idx is not None:
                    episodes = episodes[:end_idx]
                    print(f"Merging first {len(episodes)} episodes from {filepath}")
                else:
                    print(f"Merging all {len(episodes)} episodes from {filepath}")
                
                for episode in episodes:
                    input.copy(f"data/{episode}", output, f"data/demo_{episode_idx}")
                    episode_idx += 1

                if copy_attributes:
                    output["data"].attrs["env_args"] = input["data"].attrs["env_args"]
                    copy_attributes = False

    print(f"Merged dataset saved to {args_cli.output_file} with {episode_idx} total episodes")


if __name__ == "__main__":
    merge_datasets()