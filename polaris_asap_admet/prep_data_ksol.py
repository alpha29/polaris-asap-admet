import polars as pl
from rdkit import Chem
from rdkit.Chem import Descriptors

from polaris_asap_admet.io import (DATA_DIR_DIRTY, asap_KSOL_train,
                                   admet_KSOL_train_combined, computational_adme_KSOL_dirty, computational_adme_KSOL_converted)
from polaris_asap_admet.logger import logger
from polaris_asap_admet.util import print_info

TARGETS = [
    "HLM",
    "KSOL",
    "LogD",
    "MDR1-MDCKII",
    "MLM",
]


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
    #df_computational = pl.read_csv(DATA_DIR_DIRTY / "computational_adme_KSOL_dirty.csv")
    df_computational = computational_adme_KSOL_dirty.read()
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
    #df_computational.write_csv(DATA_DIR_DIRTY / "computational_adme_KSOL_converted.csv")
    computational_adme_KSOL_converted.save(df_computational)
    return df_computational


def combine():
    #df_computational = pl.read_csv(DATA_DIR_DIRTY / "computational_adme_KSOL_converted.csv").rename({"KSOL_uM": "KSOL"})
    df_computational = computational_adme_KSOL_converted.read().rename({"KSOL_uM": "KSOL"})
    df_asap = asap_KSOL_train.read()
    df_combined = pl.concat([df_computational, df_asap], how="vertical")
    admet_KSOL_train_combined.save(df_combined)


def make():
    convert_ksol_units()
    combine()


if __name__ == "__main__":
    make()
