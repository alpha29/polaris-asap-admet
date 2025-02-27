import polars as pl

from polaris_asap_admet.logger import logger


def print_info(df: pl.DataFrame, show_columns: bool = True, show_unique: bool = False):
    """
    Print diagnostic info about this dataframe.
    """
    if show_columns:
        # columns = df.columns
        columns = []
        for i, j in zip(df.columns, df.dtypes):
            columns.append(f"{i}: {j}")
    else:
        columns = "<you asked not to see these>"
    logger.info(
        f"Shape: {df.shape}, size: {df.estimated_size(unit='gb')} GB ({df.estimated_size(unit='mb')} MB), columns: {columns}."
    )
    if show_unique:
        print(f"Unique:  {df.approx_n_unique()}")


def export_tensorboard_logs():
    """
    Export tensorboard logs to CSV, including timestamped directory names.
    """
    from tensorboard.backend.event_processing import event_accumulator
    import os

    for root, _, files in os.walk("runs/"):
        for file in files:
            if "events.out.tfevents" in file:
                path = os.path.join(root, file)
                run_name = path.split("/")[1]
                ea = event_accumulator.EventAccumulator(path)
                ea.Reload()
                for tag in ea.Tags()["scalars"]:
                    if "val_loss" in tag or "train_loss" in tag:
                        data = ea.Scalars(tag)
                        steps = [d.step for d in data]
                        values = [d.value for d in data]
                        # Extract directory name (e.g., HLM_20250227_012525) from root path
                        # cbrown:  ...nope, we fixed run_name above
                        #run_name = os.path.basename(root)  # Gets the last folder, e.g., HLM_20250227_012525
                        print(f"{run_name}_{tag}: Steps {min(steps)}–{max(steps)}, Values {min(values):.4f}–{max(values):.4f}")
                        # Optional: Write to CSV
                        # with open(f"runs/{run_name}_{tag}.csv", "w") as f:
                        #     f.write("step,value\n")
                        #     for step, value in zip(steps, values):
                        #         f.write(f"{step},{value}\n")
    print("Done.")

if __name__ == "__main__":
    export_tensorboard_logs()
