import json
import subprocess
import os
from copy import deepcopy

# Path to script and json
BASE_DIR = "/preprocessing"
SCRIPT_PATH = os.path.join(BASE_DIR, "regression.py")
JSON_PATH = os.path.join(BASE_DIR, "config.json")

# Base config
base_config = {
    "df_masked": "data/dataframes/fdc/df_gm.pkl",
    "threshold": False,
    "plot_cluster": True,
    "do_evaluation": False,
    "plot_regression": False,
    "color_by_group": True
}

targets = ["CDR_SB", "MMSE"]
df_masked_options = [
    ("data/dataframes/fdc/df_gm.pkl", False),
    ("data/dataframes/fdc/df_thr02_gm.pkl", 0.2)
]

covariate_flags = [True, False]

group_settings = [
    {"group_regression": False, "group_col": None},
    {"group_regression": True, "group_col": "labels_gmm_cdr"},
    {"group_regression": True, "group_col": "Group"},
    {"group_regression": True, "group_col": "labels_km"},
]

# Specific group_col for threshold case with MMSE
group_col_thr02_mmse = "labels_km_thr02"

def run_config(cfg):
    print("Running:", cfg)
    with open(JSON_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

    subprocess.run(["python", SCRIPT_PATH])


# Run all combinations
for target_variable in targets:
    for df_masked, threshold in df_masked_options:
        for flag_covariates in covariate_flags:
            for group_setting in group_settings:
                # Adjust group_col for threshold/MMSE case
                group_col = group_setting["group_col"]
                if target_variable == "MMSE" and threshold == 0.2 and group_col == "labels_km":
                    group_col = group_col_thr02_mmse
                elif not group_setting["group_regression"]:
                    group_col = "labels_km_thr02"

                # Build config
                config = deepcopy(base_config)
                config.update({
                    "df_masked": df_masked,
                    "threshold": threshold,
                    "target_variable": target_variable,
                    "flag_covariates": flag_covariates,
                    "group_regression": group_setting["group_regression"],
                    "group_col": group_col
                })

                run_config(config)
