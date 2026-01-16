import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

def load_cube_configurations(file_path):
    """Load cube configurations from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['configurations']

def plot_cube_configurations(configurations, save_path=None):
    """Plot all cube configurations in a grid layout."""
    
    # Calculate grid dimensions
    n_configs = len(configurations)
    cols = 4  # 4 columns for better layout
    rows = (n_configs + cols - 1) // cols
    
    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Colors for the three cubes
    colors = ['blue', 'red', 'green']  # cube_1, cube_2, cube_3
    cube_names = ['Cube 1 (Blue)', 'Cube 2 (Red)', 'Cube 3 (Green)']
    
    for idx, config in enumerate(configurations):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Extract positions
        poses = config['poses']
        x_positions = [pose['pos'][0] for pose in poses]
        y_positions = [pose['pos'][1] for pose in poses]
        
        # Plot workspace boundaries (approximate based on your data)
        workspace_x = [0.35, 0.65]  # x range
        workspace_y = [-0.3, 0.3]   # y range
        
        # Draw workspace boundary
        workspace_rect = patches.Rectangle(
            (workspace_x[0], workspace_y[0]), 
            workspace_x[1] - workspace_x[0], 
            workspace_y[1] - workspace_y[0],
            linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.3
        )
        ax.add_patch(workspace_rect)
        
        # Plot cubes
        for i, (x, y, color, name) in enumerate(zip(x_positions, y_positions, colors, cube_names)):
            # Draw cube as a square (approximately 2cm x 2cm in real world)
            cube_size = 0.02  # 2cm in meters
            cube_rect = patches.Rectangle(
                (x - cube_size/2, y - cube_size/2), 
                cube_size, cube_size,
                linewidth=2, edgecolor='black', facecolor=color, alpha=0.7
            )
            ax.add_patch(cube_rect)
            
            # Add cube number
            ax.text(x, y, str(i+1), ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white')
        
        # Set equal aspect ratio and limits
        ax.set_xlim(workspace_x[0] - 0.05, workspace_x[1] + 0.05)
        ax.set_ylim(workspace_y[0] - 0.05, workspace_y[1] + 0.05)
        ax.set_aspect('equal')
        
        # Set title and labels
        ax.set_title(f"{config['name']}\n{config['description']}", fontsize=10, pad=10)
        ax.set_xlabel('X position (m)', fontsize=8)
        ax.set_ylabel('Y position (m)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    # Hide empty subplots
    for idx in range(n_configs, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)
    
    # Add legend
    legend_elements = [patches.Patch(facecolor=color, edgecolor='black', label=name) 
                      for color, name in zip(colors, cube_names)]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.suptitle('Cube Configurations for Stacking Task', fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def print_configuration_summary(configurations):
    """Print a summary of all configurations."""
    print("Configuration Summary:")
    print("=" * 50)
    
    for i, config in enumerate(configurations):
        print(f"{i+1:2d}. {config['name']}")
        print(f"    Description: {config['description']}")
        
        poses = config['poses']
        for j, pose in enumerate(poses):
            pos = pose['pos']
            print(f"    Cube {j+1}: x={pos[0]:5.2f}, y={pos[1]:6.2f}, z={pos[2]:6.4f}")
        print()

def analyze_workspace_coverage(configurations):
    """Analyze the workspace coverage of all configurations."""
    all_x = []
    all_y = []
    
    for config in configurations:
        for pose in config['poses']:
            all_x.append(pose['pos'][0])
            all_y.append(pose['pos'][1])
    
    print("Workspace Analysis:")
    print("=" * 30)
    print(f"X range: {min(all_x):.3f} to {max(all_x):.3f} m")
    print(f"Y range: {min(all_y):.3f} to {max(all_y):.3f} m")
    print(f"X span: {max(all_x) - min(all_x):.3f} m")
    print(f"Y span: {max(all_y) - min(all_y):.3f} m")
    print(f"Total configurations: {len(configurations)}")
    print(f"Total cube positions: {len(all_x)}")

def main():
    """Main function to run the visualization script."""
    # Path to your JSON file
    json_file_path = "scripts/imitation_learning/robomimic/data/bc_stack_task_test_cases_extended.json"
    
    # Check if file exists
    if not Path(json_file_path).exists():
        print(f"Error: File {json_file_path} not found!")
        print("Please make sure the path is correct.")
        return
    
    try:
        # Load configurations
        configurations = load_cube_configurations(json_file_path)
        print(f"Loaded {len(configurations)} cube configurations")
        
        # Print summary
        print_configuration_summary(configurations)
        
        # Analyze workspace
        analyze_workspace_coverage(configurations)
        
        # Create and show plots
        output_path = "cube_configurations_visualization.png"
        plot_cube_configurations(configurations, save_path=output_path)
        
    except Exception as e:
        print(f"Error loading or processing file: {e}")

if __name__ == "__main__":
    main()