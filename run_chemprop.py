import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

# Args: train_file, test_file, config_name
base_name = sys.argv[1]  # e.g., HLM
train_file = sys.argv[2]  # e.g., ADME_HLM_train.csv
test_file = sys.argv[3]  # e.g., ADME_HLM_test.csv

targets = ["HLM", "KSOL", "LOGD", "MDR1-MDCKII", "MLM"]
if base_name not in targets:
    raise ValueError(f"Invalid base_name {base_name}; must be one of {targets}")


train_path = os.path.join(".", train_file)
test_path = os.path.join(".", test_file)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., 20250220_153022
output_dir = os.path.join("runs", f"{base_name}_{timestamp}")
preds_file = f"runs/{base_name}_{timestamp}_preds.csv"

# Ensure runs/ exists
os.makedirs("runs", exist_ok=True)

# Train with RDKit features and r2
train_cmd = (
    f"chemprop train --data-path {train_path} "
    f"--molecule-featurizers v1_rdkit_2d_normalized --no-bond-feature-scaling "
    f"--metric r2 --dropout 0.2 --epochs 10 --batch-size 64 --output-dir {output_dir} --accelerator gpu --num-workers 4 "
    f"--split-type KMEANS --split-size 0.8 0.1 0.1 "
)
subprocess.run(train_cmd, shell=True, check=True)

# Predict command
predict_cmd = (
    f"chemprop predict --test-path {test_path} "
    f"--molecule-featurizers v1_rdkit_2d_normalized --no-bond-feature-scaling "
    f"--model-path {output_dir}/model_0/best.pt "
    f"--preds-path {preds_file} "
    f"--num-workers 4 --batch-size 64 "
)
subprocess.run(predict_cmd, shell=True, check=True)

# Read predictions
test_pred = pd.read_csv(preds_file)
print(test_pred.head())
