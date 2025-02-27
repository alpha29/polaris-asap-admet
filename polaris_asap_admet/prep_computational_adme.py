import polars as pl

from polaris_asap_admet.io import (DATA_DIR_DIRTY, computational_adme_dirty,
                                   computational_adme_raw)
from polaris_asap_admet.logger import logger

TARGETS = [
    "HLM",
    "KSOL",
    "LogD",
    "MDR1-MDCKII",
    "MLM",
]


def split_computational_adme() -> None:
    """
    split the computational ADME dataset
    """
    logger.info("Splitting the computational ADME dataset...")
    df = computational_adme_raw.read(show_columns=True, show_unique=True)
    dict_df = {}
    dict_df["HLM"] = (
        df.select(["SMILES", "LOG HLM_CLint (mL/min/kg)"])
        .rename({"SMILES": "CXSMILES", "LOG HLM_CLint (mL/min/kg)": "LOG_HLM_CLint"})
        .filter(pl.col("LOG_HLM_CLint").is_not_null())
    )
    dict_df["KSOL"] = (
        df.select(["SMILES", "LOG SOLUBILITY PH 6.8 (ug/mL)"])
        .rename({"SMILES": "CXSMILES", "LOG SOLUBILITY PH 6.8 (ug/mL)": "logS_ug_mL"})
        .filter(pl.col("logS_ug_mL").is_not_null())
    )
    dict_df["MDR1-MDCKII"] = (
        df.select(["SMILES", "LOG MDR1-MDCK ER (B-A/A-B)"])
        .rename({"SMILES": "CXSMILES", "LOG MDR1-MDCK ER (B-A/A-B)": "efflux"})
        .filter(pl.col("efflux").is_not_null())
    )
    dict_df["MLM"] = (
        df.select(["SMILES", "LOG RLM_CLint (mL/min/kg)"])
        .rename(
            {
                "SMILES": "CXSMILES",
                "LOG RLM_CLint (mL/min/kg)": "LOG_RLM_CLint_ml_min_kg",
            }
        )
        .filter(pl.col("LOG_RLM_CLint_ml_min_kg").is_not_null())
    )

    for tgt, df_tgt in dict_df.items():
        logger.info(f"Saving {tgt}...")
        computational_adme_dirty[tgt].save(df_tgt)

    logger.info("Done.")
