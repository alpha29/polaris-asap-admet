import numpy as np
import polars as pl

from polaris_asap_admet.io import (DATA_DIR_DIRTY, MDR1_MDCKII_train, MDR1_MDCKII_train_combined)
from polaris_asap_admet.logger import logger
from polaris_asap_admet.util import print_info

TARGETS = [
    "HLM",
    "KSOL",
    "LogD",
    "MDR1-MDCKII",
    "MLM",
]


def convert_mdr1_mdckii_units() -> pl.DataFrame:
    """
    Convert Computational-ADME MDR1-MDCK efflux ratio to MDR1-MDCKII in 10^-6 cm/s.
    Assumes log(efflux ratio) correlates with permeability, adjusted to match ASAP units.
    """
    logger.info("Converting MDR1-MDCKII units..")
    df_computational = pl.read_csv(
        DATA_DIR_DIRTY / "computational_adme_MDR1-MDCKII_dirty.csv"
    )
    print_info(df_computational)

    # Convert efflux ratio (unitless) to MDR1-MDCKII permeability (10^-6 cm/s)
    # Assume log(efflux) correlates with permeability—adjust units to match ASAP's 10^-6 cm/s
    # Placeholder: Log-transform efflux, scale to ASAP's range (e.g., 0–100 * 10^-6 cm/s)
    # Refine based on ASAP docs or competition data (e.g., mean/SD scaling)
    mdr1_cm_s = (
        np.log10(df_computational["efflux"]) * 1e-6
    )  # Log-transform, scale to 10^-6 cm/s
    # Adjust scaling factor based on ASAP range (e.g., multiply by 10 for typical permeability)
    scaling_factor = 10  # Placeholder—tune to match ASAP's 10^-6 cm/s distribution
    mdr1_cm_s = mdr1_cm_s * scaling_factor

    df_computational = df_computational.with_columns(
        mdr1_cm_s.alias("MDR1_MDCKII_10-6_cm_s")
    )

    # Drop efflux (optional, keep for debugging)
    df_computational = df_computational.drop("efflux")

    print(df_computational.head())
    df_computational.write_csv(
        DATA_DIR_DIRTY / "computational_adme_MDR1-MDCKII_converted.csv"
    )
    return df_computational

def combine():
    df_computational = pl.read_csv(DATA_DIR_DIRTY / "computational_adme_MDR1-MDCKII_converted.csv").rename({"MDR1_MDCKII_10-6_cm_s": "MDR1-MDCKII"})
    df_asap = MDR1_MDCKII_train.read()
    df_combined = pl.concat([df_computational, df_asap], how="vertical")
    MDR1_MDCKII_train_combined.save(df_combined)

def make():
    convert_mdr1_mdckii_units()
    combine()

if __name__ == "__main__":
    make()
