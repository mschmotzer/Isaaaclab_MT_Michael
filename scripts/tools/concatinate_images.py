import h5py
import numpy as np
import argparse


def process_hdf5(input_hdf5_path: str, output_hdf5_path: str):
    with h5py.File(input_hdf5_path, "r") as fin, \
         h5py.File(output_hdf5_path, "w") as fout:

        # --- Create data group ---
        fout.create_group("data")

        # 🔴 CRITICAL: copy robomimic metadata
        for k, v in fin["data"].attrs.items():
            fout["data"].attrs[k] = v

        for demo_key in fin["data"].keys():
            print(f"Processing {demo_key}")

            demo_in = fin[f"data/{demo_key}"]
            demo_out = fout["data"].create_group(demo_key)
                # --- Copy episode attributes ---
            for k, v in demo_in.attrs.items():
                demo_out.attrs[k] = v
            # --- Copy everything except rgb_camera ---
            for key in demo_in.keys():
                if key == "rgb_camera":
                    continue
                fin.copy(demo_in[key], demo_out, name=key)

            # --- Load matrices from rgb_camera ---
            rgb_group = demo_in["rgb_camera"]

            cam_keys = sorted(rgb_group.keys())
            matrices = []
            feat_dim = None

            for cam_key in cam_keys:
                mat = rgb_group[cam_key][()]
                if mat.ndim != 2:
                    raise ValueError(
                        f"{demo_key}/{cam_key} must be 2D, got {mat.shape}"
                    )

                if feat_dim is None:
                    feat_dim = mat.shape[1]
                elif mat.shape[1] != feat_dim:
                    raise ValueError(
                        f"{demo_key}: feature dim mismatch in {cam_key}"
                    )

                matrices.append(mat)

            img_features = np.concatenate(matrices, axis=1)

            # --- Store img_features ---
            obs_group = demo_out.require_group("obs")
            obs_group.create_dataset(
                "img_features",
                data=img_features,
                compression="gzip"
            )

        print(f"\nDone. New file written to:\n{output_hdf5_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()
    process_hdf5(args.input, args.output)
