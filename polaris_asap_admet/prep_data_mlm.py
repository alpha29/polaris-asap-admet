import polars as pl

from polaris_asap_admet.io import DATA_DIR_DIRTY, asap_MLM_train, admet_MLM_train_combined, computational_adme_MLM_dirty, computational_adme_MLM_converted
from polaris_asap_admet.logger import logger
from polaris_asap_admet.util import print_info

TARGETS = [
    "HLM",
    "KSOL",
    "LogD",
    "MDR1-MDCKII",
    "MLM",
]


def convert_mlm_units():
    """
    ADME_public_set_3521.csv gives MLM data in log scale (mL/min/kg) - presumably that's kg of body weight.
    Convert to uL/min/mg.

    Also:  How similar are rats and mice, anyway?  Perhaps we'll find out!
    """
    logger.info("Converting MLM units...")
    #df_computational = pl.read_csv(DATA_DIR_DIRTY / "computational_adme_MLM_dirty.csv")
    df_computational = computational_adme_MLM_dirty.read()
    print_info(df_computational)

    scaling_factor = 0.5  # TODO: NOT sure I buy this value, need to double check
    # unlog
    rlm_ml_min_kg = 10 ** df_computational["LOG_RLM_CLint_ml_min_kg"]
    # convert ml to mikes
    rlm_ul_min_kg = rlm_ml_min_kg * 1_000
    # Estimated liver microsomal protein content: ~40 mg/g liver, ~20 g liver/kg body weight → ~800 mg protein/kg.
    # TODO - confirm.  We did this for HLM, not sure the same applies to MLM
    rlm_ul_min_mg = rlm_ul_min_kg / 800 * scaling_factor
    df_computational = df_computational.with_columns(
        rlm_ul_min_mg.alias("MLM_uL_min_mg")
    )
    # drop LOG_RLM_CLint
    df_computational = df_computational.drop("LOG_RLM_CLint_ml_min_kg")
    print(df_computational.head())
    #df_computational.write_csv(DATA_DIR_DIRTY / "computational_adme_MLM_converted.csv")
    computational_adme_MLM_converted.save(df_computational)
    return df_computational


def combine():
    #df_computational = pl.read_csv(DATA_DIR_DIRTY / "computational_adme_MLM_converted.csv").rename({"MLM_uL_min_mg": "MLM"})
    df_computational = computational_adme_MLM_converted.read().rename({"MLM_uL_min_mg": "MLM"})
    df_asap = asap_MLM_train.read()
    df_combined = pl.concat([df_computational, df_asap], how="vertical")
    admet_MLM_train_combined.save(df_combined)


def make():
    convert_mlm_units()
    combine()


if __name__ == "__main__":
    make()
