# plot_trajectories_views.py
# Example:
#   python plot_trajectories_views.py --input_file ./datasets/Dataset_mimicvsjucier_10_demo.hdf5 \
#       --limit_demos 50 --stride 2 --save_path trajectories_views.png --include_cube

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot trajectories (XY, XZ, YZ) and report mean/std per axis.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the HDF5 dataset file")
    parser.add_argument("--limit_demos", type=int, default=None, help="Limit the number of demos to plot (default: all)")
    parser.add_argument("--stride", type=int, default=1, help="Downsample factor for timesteps (default: 1 = no downsampling)")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the figure (default: show interactively only)")
    parser.add_argument("--include_cube", action="store_true", help="Also plot and compute stats for cube positions")
    args = parser.parse_args()

    with h5py.File(args.input_file, "r") as f:
        demos = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[-1]))
        if args.limit_demos is not None:
            demos = demos[:args.limit_demos]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        views = [("X", "Y"), ("X", "Z"), ("Y", "Z")]
        fig.suptitle(args.input_file)

        # Collect all EEF samples for stats
        all_eef_samples = []
        # Optionally collect all cube samples for stats
        all_cube_samples = [] if args.include_cube else None

        for d in demos:
            print("Demos:",d)
            base = f[f"data/{d}/obs"]
            eef = base["eef_pos"][:][::args.stride]           # (T, 3)
            all_eef_samples.append(eef)

            if args.include_cube:
                cube = base["cube_positions"][:][::args.stride]  # (T, 3)
                all_cube_samples.append(cube)

            for ax, (a, b) in zip(axes, views):
                ia, ib = "XYZ".index(a), "XYZ".index(b)
                ax.plot(eef[:, ia], eef[:, ib], alpha=0.55, linewidth=1.2,
                        label="EEF" if d == demos[0] else "")
                if args.include_cube:
                    ax.plot(cube[:, ia], cube[:, ib], alpha=0.55, linewidth=1.0,
                            linestyle="--", label="Cube" if d == demos[0] else "")
                ax.set_xlabel(a)
                ax.set_ylabel(b)
                ax.set_title(f"{a}{b} view")

        # === Compute stats (mean & std) over ALL timesteps of ALL demos ===
        eef_all = np.vstack(all_eef_samples) if len(all_eef_samples) > 0 else np.empty((0, 3))
        eef_mean = eef_all.mean(axis=0) if eef_all.size else np.array([np.nan, np.nan, np.nan])
        eef_std  = eef_all.std(axis=0)  if eef_all.size else np.array([np.nan, np.nan, np.nan])

        print("EEF mean  [X, Y, Z]:", eef_mean)
        print("EEF std   [X, Y, Z]:", eef_std)

        # Drop a mean±std marker for each 2D view (EEF)
        for ax, (a, b) in zip(axes, views):
            ia, ib = "XYZ".index(a), "XYZ".index(b)
            ax.errorbar(eef_mean[ia], eef_mean[ib],
                        xerr=eef_std[ia], yerr=eef_std[ib],
                        fmt="o", capsize=3, label="EEF mean±std")

        # Optionally compute/plot cube stats
        if args.include_cube and len(all_cube_samples) > 0:
            cube_all = np.vstack(all_cube_samples)
            cube_mean = cube_all.mean(axis=0)
            cube_std  = cube_all.std(axis=0)
            print("Cube mean [X, Y, Z]:", cube_mean)
            print("Cube std  [X, Y, Z]:", cube_std)

            for ax, (a, b) in zip(axes, views):
                ia, ib = "XYZ".index(a), "XYZ".index(b)
                ax.errorbar(cube_mean[ia], cube_mean[ib],
                            xerr=cube_std[ia], yerr=cube_std[ib],
                            fmt="s", capsize=3, label="Cube mean±std")

        axes[0].legend(loc="best")
        plt.tight_layout()

        if args.save_path:
            plt.savefig(args.save_path, dpi=200, bbox_inches="tight")
            print(f"Saved figure to {args.save_path}")
        else:
            plt.show()

if __name__ == "__main__":
    main()
