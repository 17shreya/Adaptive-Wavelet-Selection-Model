"""
Utility functions for the pain-recognition package.
"""

from .io import (
    load_config,
    read_csv,
    read_json,
    read_yaml,
    write_csv,
    write_json,
    write_yaml,
)

from .paths import (
    ensure_directory,
    get_config_dir,
    get_config_path,
    get_data_dir,
    get_figures_dir,
    get_project_root,
    get_results_dir,
    resolve_dataset_root,
)

from .seeds import (
    DEFAULT_RANDOM_STATE,
    get_reproducibility_settings,
    set_global_seed,
)


__all__ = [
    "DEFAULT_RANDOM_STATE",
    "set_global_seed",
    "get_reproducibility_settings",

    "get_project_root",
    "get_config_dir",
    "get_config_path",
    "get_data_dir",
    "get_results_dir",
    "get_figures_dir",
    "resolve_dataset_root",
    "ensure_directory",

    "load_config",
    "read_csv",
    "write_csv",
    "read_yaml",
    "write_yaml",
    "read_json",
    "write_json",
]




from pain_recognition.utils import (
    get_config_path,
    load_config,
    set_global_seed,
)


config = load_config(
    get_config_path(
        "biovid.yaml"
    )
)


random_state = config[
    "reproducibility"
][
    "random_state"
]


set_global_seed(
    random_state
)


fs = config[
    "signal"
][
    "sampling_rate"
]


window_duration = config[
    "segmentation"
][
    "window_duration_seconds"
]


overlap = config[
    "segmentation"
][
    "overlap"
]
