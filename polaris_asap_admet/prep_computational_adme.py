import numpy as np
import polars as pl
from polaris.competition import CompetitionSpecification
from polaris.dataset import Subset
from rdkit import Chem
from rdkit.Chem import Descriptors
from typeguard import typechecked

from polaris_asap_admet.io import (DATA_DIR_DIRTY, admet_train_clean,
                                   computational_adme_raw, test_raw, train_raw)
from polaris_asap_admet.logger import logger
from polaris_asap_admet.util import print_info

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
        df_tgt.write_csv(DATA_DIR_DIRTY / f"computational_adme_{tgt}_dirty.csv")
    logger.info("Done.")


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


# Calculate molar mass for each SMILES using RDKit
def get_molar_mass(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return (
                Descriptors.MolWt(mol) / 1000
            )  # Convert g/mol to mg/umol (for ug/mL to uM)
        return None
    except:
        return None


def convert_ksol_units() -> pl.DataFrame:
    """
    Convert Computational-ADME solubility from LOG SOLUBILITY PH 6.8 (ug/mL) to uM using RDKit molar masses.
    """
    logger.info("Converting KSOL units..")
    # Load Computational-ADME Solubility data (correct column name)
    df_computational = pl.read_csv(DATA_DIR_DIRTY / "computational_adme_KSOL_dirty.csv")
    print_info(df_computational)

    # Calculate molar mass for each SMILES
    molar_masses = df_computational["CXSMILES"].map_elements(get_molar_mass)
    df_computational = df_computational.with_columns(
        molar_masses.alias("molar_mass_mg_umol")
    )

    # Drop rows with missing molar masses
    df_computational = df_computational.filter(
        pl.col("molar_mass_mg_umol").is_not_null()
    )

    # Convert log(ug/mL) to uM using per-molecule molar mass
    # logS_ug_mL → S_ug_mL = 10 ** logS_ug_mL
    # S_uM = S_ug_mL / (molar_mass_mg_umol * 1e-3) = S_ug_mL / molar_mass_ug_umol (in ug/umol)
    ksol_um = (10 ** df_computational["logS_ug_mL"]) / (
        df_computational["molar_mass_mg_umol"] * 1e-3
    )
    df_computational = df_computational.with_columns(ksol_um.alias("KSOL_uM"))

    # Drop logS and molar mass (optional, keep for debugging)
    df_computational = df_computational.drop("logS_ug_mL", "molar_mass_mg_umol")

    print(df_computational.head())
    df_computational.write_csv(DATA_DIR_DIRTY / "computational_adme_KSOL_converted.csv")
    return df_computational


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


def convert_mlm_units():
    """
    ADME_public_set_3521.csv gives MLM data in log scale (mL/min/kg) - presumably that's kg of body weight.
    Convert to uL/min/mg.

    Also:  How similar are rats and mice, anyway?  Perhaps we'll find out!
    """
    logger.info("Converting MLM units...")
    df_computational = pl.read_csv(DATA_DIR_DIRTY / "computational_adme_MLM_dirty.csv")
    print_info(df_computational)

    scaling_factor = 0.5  # TODO: NOT sure I buy this value, need to double check
    # unlog
    rlm_ml_min_kg = 10 ** df_computational["LOG_RLM_CLint"]
    # convert ml to mikes
    rlm_ul_min_kg = rlm_ml_min_kg * 1_000
    # Estimated liver microsomal protein content: ~40 mg/g liver, ~20 g liver/kg body weight → ~800 mg protein/kg.
    # TODO - confirm.  We did this for HLM, not sure the same applies to MLM
    rlm_ul_min_mg = rlm_ul_min_kg / 800 * scaling_factor
    df_computational = df_computational.with_columns(
        rlm_ul_min_mg.alias("MLM_uL_min_mg")
    )
    # drop LOG_RLM_CLint
    df_computational = df_computational.drop("LOG_RLM_CLint")
    print(df_computational.head())
    df_computational.write_csv(DATA_DIR_DIRTY / "computational_adme_MLM_converted.csv")
    return df_computational
