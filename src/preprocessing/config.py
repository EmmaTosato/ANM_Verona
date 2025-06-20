from dataclasses import dataclass
import json
from typing import Dict, Any, Tuple

@dataclass
class ConfigLoader:
    config_path: str = "src/parameters/config.json"
    paths_path: str = "src/parameters/paths.json"

    def load(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, str]]]:
        with open(self.config_path) as f:
            config = json.load(f)
        with open(self.paths_path) as f:
            paths = json.load(f)

        args = {}
        for section in config.values():
            if isinstance(section, dict):
                args.update(section)

        flat_paths = {}
        for section in paths.values():
            if isinstance(section, dict):
                flat_paths.update(section)

        args["df_path"] = self.resolve_data_path(
            dataset_type=args.get("dataset_type", "fdc"),
            threshold=args.get("threshold"),
            paths=paths
        )

        args.update(flat_paths)
        return args, config, paths

    @staticmethod
    def resolve_data_path(dataset_type: str, threshold, paths: Dict[str, Any]) -> str:
        flat_paths = {}
        for section in paths.values():
            if isinstance(section, dict):
                flat_paths.update(section)

        if dataset_type == "fdc":
            if not threshold:
                return flat_paths["df_masked"]
            if threshold == 0.2:
                return flat_paths["df_masked_02"]
            raise ValueError(f"No FDC path defined for threshold={threshold}")

        if dataset_type == "networks":
            if not threshold:
                return flat_paths["net_noThr"]
            if threshold == 0.2:
                return flat_paths["net_thr02"]
            if threshold == 0.1:
                return flat_paths["net_thr01"]
            raise ValueError(f"No network path defined for threshold={threshold}")

        raise ValueError(f"Unknown dataset_type: {dataset_type}")
