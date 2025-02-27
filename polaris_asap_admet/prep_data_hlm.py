import polars as pl

from polaris_asap_admet.io import DATA_DIR_DIRTY, HLM_train, HLM_train_combined
from polaris_asap_admet.logger import logger
from polaris_asap_admet.util import print_info

TARGETS = [
    "HLM",
    "KSOL",
    "LogD",
    "MDR1-MDCKII",
    "MLM",
]


def convert_hlm_units():
    """
    ADME_public_set_3521.csv gives HLM data in log scale (mL/min/kg) - presumably that's kg of body weight.
    Convert to uL/min/mg.
    """
    logger.info("Converting HLM units...")
    df_computational = pl.read_csv(DATA_DIR_DIRTY / "computational_adme_HLM_dirty.csv")
    print_info(df_computational)

    # Unlog and convert
    # scaling_factor = 0.05  # Adjust based on ASAP/docs (0.02–0.1)
    scaling_factor = 0.5  # 0.05 looked like crap - overestimates in vitro/in vivo scaling, compressing values to 0.3–0.6 uL/min/mg.
    hlm_ml_min_kg = 10 ** df_computational["LOG_HLM_CLint"]
    hlm_ul_min_kg = hlm_ml_min_kg * 1_000  # mL to uL
    # Estimated liver microsomal protein content: ~40 mg/g liver, ~20 g liver/kg body weight → ~800 mg protein/kg.
    hlm_ul_min_mg = hlm_ul_min_kg / 800 * scaling_factor  # uL/min/kg to uL/min/mg
    df_computational = df_computational.with_columns(
        hlm_ul_min_mg.alias("HLM_uL_min_mg")
    )
    df_computational = df_computational.filter(pl.col("HLM_uL_min_mg").is_not_null())
    df_computational = df_computational.drop("LOG_HLM_CLint")
    df_computational.write_csv(DATA_DIR_DIRTY / "computational_adme_HLM_converted.csv")
    print(df_computational.head())
    return df_computational


def combine():
    df_computational = pl.read_csv(
        DATA_DIR_DIRTY / "computational_adme_HLM_converted.csv"
    ).rename({"HLM_uL_min_mg": "HLM"})
    df_asap = HLM_train.read()
    df_combined = pl.concat([df_computational, df_asap], how="vertical")
    HLM_train_combined.save(df_combined)


def make():
    convert_hlm_units()
    combine()


if __name__ == "__main__":
    make()
