"""Velocity command presets and configuration utilities.

This module provides preset velocity controller configurations and utilities
for managing velocity settings in IsaacLab environments.
"""

import json
import pathlib


def _convert_lists_to_tuples(config_dict):
    """Convert list values back to tuples for velocity config.

    JSON only supports lists, but velocity config expects tuples for ranges.
    """
    result = {}
    for key, value in config_dict.items():
        if isinstance(value, list):
            result[key] = tuple(value)
        else:
            result[key] = value
    return result


def _load_velocity_presets():
    """Load velocity presets from JSON file."""
    preset_file = pathlib.Path(__file__).parent / "velocity_presets.json"
    with open(preset_file, "r") as f:
        presets_raw = json.load(f)

    # Convert lists to tuples for all presets
    return {
        name: _convert_lists_to_tuples(config) for name, config in presets_raw.items()
    }


# Load presets from JSON file
VELOCITY_PRESETS = _load_velocity_presets()

# Default custom velocity controller configuration parameters
VELOCITY_CONFIG = {
    "lin_vel_x_range": (-2.0, 2.0),  # Linear velocity X range (m/s)
    "lin_vel_y_range": (-1.5, 1.5),  # Linear velocity Y range (m/s)
    "ang_vel_z_range": (-1.5, 1.5),  # Angular velocity Z range (rad/s)
    "heading_command": True,  # Use heading-based control
    "heading_control_stiffness": 0.8,  # Heading control stiffness
    "resampling_time_range": (8.0, 12.0),  # Command resampling time range (s)
    "rel_standing_envs": 0.05,  # Probability of standing environments
    "rel_heading_envs": 1.0,  # Probability of heading-based control
}


def apply_velocity_preset(preset_name):
    """Apply a preset velocity configuration.

    Args:
        preset_name: Name of the preset to apply

    Available presets:
        - "default": Balanced movement (current default)
        - "aggressive": Fast movements, quick changes
        - "conservative": Slow, careful movements
        - "exploration": Frequent direction changes, direct angular control
        - "precision": Very slow, precise movements
        - "high_speed": Maximum speed, minimal standing
        - "forward_only_slow": Forward-only movement, very slow with frequent standing
        - "reverse_only_slow": Reverse-only movement, very slow

    Returns:
        bool: True if preset was applied successfully, False otherwise
    """
    global VELOCITY_CONFIG

    if preset_name not in VELOCITY_PRESETS:
        print(f"Error: Unknown preset '{preset_name}'")
        print(f"Available presets: {list(VELOCITY_PRESETS.keys())}")
        return False

    preset_config = VELOCITY_PRESETS[preset_name]
    VELOCITY_CONFIG.update(preset_config)

    print(f"Applied velocity preset: '{preset_name}'")
    print("\nPreset Configuration:")
    for key, value in preset_config.items():
        print(f"  {key}: {value}")
    print()
    return True


def list_velocity_presets():
    """List all available velocity presets with descriptions."""
    print("Available Velocity Presets:")
    print("=" * 50)

    descriptions = {
        "default": "Balanced movement (default configuration)",
        "aggressive": "Fast movements, quick direction changes, minimal standing",
        "conservative": "Slow, careful movements with frequent standing",
        "exploration": "Frequent direction changes, direct angular velocity control",
        "precision": "Very slow, precise movements for delicate tasks",
        "high_speed": "Maximum speed capabilities, minimal interruptions",
        "forward_only_slow": "Forward-only movement, very slow with frequent standing",
        "reverse_only_slow": "Reverse-only movement, very slow",
    }

    for preset_name in VELOCITY_PRESETS.keys():
        description = descriptions.get(preset_name, "No description available")
        print(f"  '{preset_name}': {description}")
        print(
            f"    Speed: X={VELOCITY_PRESETS[preset_name]['lin_vel_x_range'][1]:.1f}m/s, "
            f"Y={VELOCITY_PRESETS[preset_name]['lin_vel_y_range'][1]:.1f}m/s, "
            f"Z={VELOCITY_PRESETS[preset_name]['ang_vel_z_range'][1]:.1f}rad/s"
        )
        print()


def update_velocity_config(**kwargs):
    """Update velocity configuration parameters.

    Args:
        **kwargs: Key-value pairs to update in VELOCITY_CONFIG

    Example usage:
        # For faster movement
        update_velocity_config(
            lin_vel_x_range=(-3.0, 3.0),
            lin_vel_y_range=(-2.0, 2.0),
            ang_vel_z_range=(-2.0, 2.0)
        )

        # For more aggressive heading control
        update_velocity_config(
            heading_control_stiffness=1.5,
            rel_heading_envs=0.8
        )

        # For more frequent command changes
        update_velocity_config(
            resampling_time_range=(3.0, 5.0)
        )

        # Or use presets:
        # apply_velocity_preset("aggressive")
        # apply_velocity_preset("conservative")
    """
    global VELOCITY_CONFIG
    for key, value in kwargs.items():
        if key in VELOCITY_CONFIG:
            VELOCITY_CONFIG[key] = value
            print(f"Updated {key}: {value}")
        else:
            print(f"Warning: Unknown velocity config parameter: {key}")

    # Print current configuration
    print("\nCurrent Velocity Configuration:")
    for key, value in VELOCITY_CONFIG.items():
        print(f"  {key}: {value}")
    print()
