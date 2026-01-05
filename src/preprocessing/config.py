from dataclasses import dataclass
import json
import pandas as pd

@dataclass
class ConfigLoader:
    """
    A class to load and merge configuration parameters from JSON files.
    Produces a unified dictionary `args` that includes all values from config.json and paths.json,
    plus derived paths such as df_path based on dataset type and threshold.
    """
    def __init__(self, config_path="src/config/ml_config.json", paths_path="src/config/ml_paths.json"):
        # Load configuration and paths from JSON
        with open(config_path) as f:
            self.config = json.load(f)
        with open(paths_path) as f:
            self.paths = json.load(f)

        # Merge both configs into a single flat dictionary
        self.args = self._resolve_args()

    def _resolve_args(self):
        args = {}

        # Flatten sections from config.json
        for section_name, section in self.config.items():
            if isinstance(section, dict):
                for k, v in section.items():
                    args[k] = v
            else:
                args[section_name] = section

        # Flatten sections from paths.json
        for section_name, section in self.paths.items():
            if isinstance(section, dict):
                for k, v in section.items():
                    args[k] = v
            else:
                args[section_name] = section

        # Compute and store df_path based on dataset type and threshold
        args["df_path"] = self._resolve_data_path(
            dataset_type=args["dataset_type"],
            threshold=args["threshold"],
            flat_args=args
        )

        return args

    def _resolve_data_path(self, dataset_type, threshold, flat_args):
        """
        Selects the correct dataframe path based on dataset type and threshold.
        Values must match keys defined in paths.json.
        """
        if dataset_type == "voxel":
            if not threshold:
                return flat_args["df_masked"]
            elif threshold == 0.2:
                return flat_args["df_masked_02"]
        elif dataset_type == "networks":
            if not threshold:
                return flat_args["yeo_noThr"]
            elif threshold == 0.2:
                return flat_args["yeo_02thr"]
            elif threshold == 0.1:
                return flat_args["yeo_01thr"]
        elif dataset_type == "parcellated":
            return flat_args["df_parcellated"]

        raise ValueError(f"Invalid dataset_type={dataset_type} or threshold={threshold}")

    def load_all(self):
        """
        Loads the main dataframe (df), metadata (meta), and returns also the input dataframe for downstream analysis.
        Returns:
            - args (dict): flattened config
            - df_input (DataFrame): dataframe with features (ready for modeling)
            - meta (DataFrame): metadata (if needed separately)
        """
        dataset_type = self.args["dataset_type"]

        if dataset_type == "voxel":
            df_input = pd.read_pickle(self.args["df_path"])
        elif dataset_type == "networks" or dataset_type == "parcellated":
            df_input = pd.read_csv(self.args["df_path"])
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        meta = pd.read_csv(self.args["df_meta"])

        return self.args, df_input, meta

