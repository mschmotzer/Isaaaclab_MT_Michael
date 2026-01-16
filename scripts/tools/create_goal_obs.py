import h5py
import numpy as np
import argparse
import os

def copy_specific_obs_with_goals(obs_group, next_obs_grp, change_indices, all_indices, target_keys):
    """Copy only specific observation keys and fill with goal observations"""
    
    for obs_key in target_keys:
        if obs_key not in obs_group:
            print(f"Warning: '{obs_key}' not found in observations, skipping.")
            continue
            
        obs_item = obs_group[obs_key]
        
        if isinstance(obs_item, h5py.Dataset):
            # It's a dataset, process it
            obs_data = obs_item[:]
            
            # Check if this dataset has a time dimension (first axis should match trajectory length)
            expected_length = all_indices[-1]  # This is the total trajectory length
            
            if len(obs_data.shape) == 0 or obs_data.shape[0] != expected_length:
                # This dataset doesn't have a time dimension or has wrong length
                # Just copy it as-is (it's likely a constant/config parameter)
                next_obs_grp.create_dataset(obs_key, data=obs_data)
                continue
            
            goal_obs_data = np.zeros_like(obs_data)
            
            # Fill goal observations based on change indices
            start_idx = 0
            for i, end_idx in enumerate(all_indices):
                if i < len(change_indices):
                    # Use observation at the change index as goal
                    goal_data = obs_data[change_indices[i]]
                else:
                    # For the final segment, use the last observation
                    goal_data = obs_data[-1]
                
                # Fill the segment with the goal values
                goal_obs_data[start_idx:end_idx] = goal_data
                start_idx = end_idx
            
            # Create dataset in next_obs group
            next_obs_grp.create_dataset(obs_key, data=goal_obs_data)
            
        elif isinstance(obs_item, h5py.Group):
            # It's a subgroup, create corresponding subgroup and recurse
            sub_next_obs_grp = next_obs_grp.create_group(obs_key)
            # For subgroups, we still need to copy all their contents
            copy_obs_structure_with_goals(obs_item, sub_next_obs_grp, change_indices, all_indices)

def copy_obs_structure_with_goals(obs_group, next_obs_grp, change_indices, all_indices):
    """Recursively copy observation structure and fill with goal observations"""
    
    for obs_key in obs_group.keys():
        obs_item = obs_group[obs_key]
        
        if isinstance(obs_item, h5py.Dataset):
            # It's a dataset, process it
            obs_data = obs_item[:]
            
            # Check if this dataset has a time dimension (first axis should match trajectory length)
            expected_length = all_indices[-1]  # This is the total trajectory length
            
            if len(obs_data.shape) == 0 or obs_data.shape[0] != expected_length:
                # This dataset doesn't have a time dimension or has wrong length
                # Just copy it as-is (it's likely a constant/config parameter)
                next_obs_grp.create_dataset(obs_key, data=obs_data)
                continue
            
            goal_obs_data = np.zeros_like(obs_data)
            
            # Fill goal observations based on change indices
            start_idx = 0
            for i, end_idx in enumerate(all_indices):
                if i < len(change_indices):
                    # Use observation at the change index as goal
                    goal_data = obs_data[change_indices[i]]
                else:
                    # For the final segment, use the last observation
                    goal_data = obs_data[-1]
                
                # Fill the segment with the goal values
                goal_obs_data[start_idx:end_idx] = goal_data
                start_idx = end_idx
            
            # Create dataset in next_obs group
            next_obs_grp.create_dataset(obs_key, data=goal_obs_data)
            
        elif isinstance(obs_item, h5py.Group):
            # It's a subgroup, create corresponding subgroup and recurse
            sub_next_obs_grp = next_obs_grp.create_group(obs_key)
            copy_obs_structure_with_goals(obs_item, sub_next_obs_grp, change_indices, all_indices)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create goal observations for end-effector positions and quaternions")
    parser.add_argument("--input_file", type=str, required=True, 
                       help="Path to the input HDF5 file")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Path to the output HDF5 file (default: adds '_with_goals' to input filename)")
    
    args = parser.parse_args()
    
    # Create output filename if not specified
    if args.output_file is None:
        base_name, ext = os.path.splitext(args.input_file)
        output_file = f"{base_name}_with_goals{ext}"
    else:
        output_file = args.output_file
    
    print(f"Input file: {args.input_file}")
    print(f"Output file: {output_file}")

    # Open input file in read mode and output file in write mode
    with h5py.File(args.input_file, "r") as input_f, h5py.File(output_file, "w") as output_f:
        
        # Copy everything from input to output file
        for key in input_f.keys():
            input_f.copy(key, output_f)
        
        # Now process the copied data to add goal observations
        data_group = output_f["data"]
        
        # Define which observation keys to include in next_obs
        target_obs_keys = ['eef_pos', 'eef_quat', 'gripper_pos', 'object']
        
        # Iterate through all demos
        for demo_key in data_group.keys():
            demo_group = data_group[demo_key]
            
            # Check that the expected dataset exists
            if "obs" not in demo_group or "gripper_pos" not in demo_group["obs"]:
                print(f"Skipping {demo_key}: 'obs/gripper_pos' not found.")
                continue

            gripper_pos = demo_group["obs"]["gripper_pos"][:,0]
            obs_group = demo_group["obs"]

            goal_values = (np.abs(gripper_pos) > 0.032).astype(np.int8)
            change_indices = np.where(np.diff(goal_values) != 0)[0] + 1
            
            # Add final index to change_indices for easier processing
            all_indices = np.concatenate([change_indices, [len(gripper_pos)]])
            
            # Create or get next_obs group
            if "next_obs" in demo_group:
                # Remove existing next_obs group and recreate it
                del demo_group["next_obs"]
            
            next_obs_grp = demo_group.create_group("next_obs")
            
            # Copy only specific observation keys
            copy_specific_obs_with_goals(obs_group, next_obs_grp, change_indices, all_indices, target_obs_keys)

    print(f"✅ Finished creating next_obs for all demos. Output saved to: {output_file}")

if __name__ == "__main__":
    main()