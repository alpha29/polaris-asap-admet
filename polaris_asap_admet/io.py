import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from polaris_asap_admet.logger import logger
from polaris_asap_admet.util import print_info

pl.Config(tbl_rows=500)
pl.Config(fmt_str_lengths=500)

POLARIS_ASAP_ADMET_HOME = os.environ["POLARIS_ASAP_ADMET_HOME"]
DATA_DIR = Path(POLARIS_ASAP_ADMET_HOME) / "data"
DATA_DIR_RAW = DATA_DIR / "raw"
DATA_DIR_CLEAN = DATA_DIR / "clean"
DATA_DIR_DIRTY = DATA_DIR / "dirty"
DATA_DIR_COMBINED = DATA_DIR / "combined"

# yeah don't do this, but we'll live with it
DATA_DIR_RAW.mkdir(parents=True, exist_ok=True)
DATA_DIR_CLEAN.mkdir(parents=True, exist_ok=True)
DATA_DIR_DIRTY.mkdir(parents=True, exist_ok=True)
DATA_DIR_COMBINED.mkdir(parents=True, exist_ok=True)


@dataclass
class NamedDataset:
    """
    put a tags dict in here someday or something
    """

    name: str
    filepath: Path | str

    def save(self, df: pl.DataFrame) -> None:
        logger.info(f"Saving {self.name} to {self.filepath}...")
        if str(self.filepath).endswith(".csv"):
            df.write_csv(self.filepath)
        elif str(self.filepath).endswith(".parquet"):
            df.write_parquet(self.filepath)
        else:
            raise ValueError(f"Unsupported file format: {self.filepath}")
        logger.info("Done.")

    def read(
        self,
        show_columns: bool = False,
        show_unique: bool = False,
        n: int | None = None,
    ) -> pl.DataFrame:
        logger.info(f"Reading {self.name} from {self.filepath}...")
        if str(self.filepath).endswith(".csv"):
            df = pl.read_csv(self.filepath, n_rows=n)
        elif str(self.filepath).endswith(".parquet"):
            df = pl.read_parquet(self.filepath, n_rows=n)
        else:
            raise ValueError(f"Unsupported file format: {self.filepath}")
        print_info(df, show_columns=show_columns, show_unique=show_unique)
        logger.info("Done.")
        return df


asap_train_raw = NamedDataset(
    name="asap_train_raw", filepath=DATA_DIR_RAW / "asap_train_raw.csv"
)
asap_test_raw = NamedDataset(
    name="asap_test_raw", filepath=DATA_DIR_RAW / "asap_test_raw.csv"
)
computational_adme_raw = NamedDataset(
    name="computational_adme_raw", filepath=DATA_DIR_RAW / "ADME_public_set_3521.csv"
)

##########################
# computational-adme data
##########################
computational_adme_HLM_dirty = NamedDataset(
    "computational_adme_HLM_dirty", DATA_DIR_DIRTY / "computational_adme_HLM_dirty.csv"
)
computational_adme_KSOL_dirty = NamedDataset(
    "computational_adme_KSOL_dirty",
    DATA_DIR_DIRTY / "computational_adme_KSOL_dirty.csv",
)
computational_adme_LogD_dirty = NamedDataset(
    "computational_adme_LogD_dirty",
    DATA_DIR_DIRTY / "computational_adme_LogD_dirty.csv",
)
computational_adme_MDR1_MDCKII_dirty = NamedDataset(
    "computational_adme_MDR1_MDCKII_dirty",
    DATA_DIR_DIRTY / "computational_adme_MDR1_MDCKII_dirty.csv",
)
computational_adme_MLM_dirty = NamedDataset(
    "computational_adme_MLM_dirty", DATA_DIR_DIRTY / "computational_adme_MLM_dirty.csv"
)

computational_adme_dirty = {
    "HLM": computational_adme_HLM_dirty,
    "KSOL": computational_adme_KSOL_dirty,
    "LogD": computational_adme_LogD_dirty,
    "MDR1-MDCKII": computational_adme_MDR1_MDCKII_dirty,
    "MLM": computational_adme_MLM_dirty,
}

computational_adme_HLM_converted = NamedDataset(
    "computational_adme_HLM_converted",
    DATA_DIR_DIRTY / "computational_adme_HLM_converted.csv",
)
computational_adme_KSOL_converted = NamedDataset(
    "computational_adme_KSOL_converted",
    DATA_DIR_DIRTY / "computational_adme_KSOL_converted.csv",
)
computational_adme_LogD_converted = NamedDataset(
    "computational_adme_LogD_converted",
    DATA_DIR_DIRTY / "computational_adme_LogD_converted.csv",
)
computational_adme_MDR1_MDCKII_converted = NamedDataset(
    "computational_adme_MDR1_MDCKII_converted",
    DATA_DIR_DIRTY / "computational_adme_MDR1_MDCKII_converted.csv",
)
computational_adme_MLM_converted = NamedDataset(
    "computational_adme_MLM_converted",
    DATA_DIR_DIRTY / "computational_adme_MLM_converted.csv",
)

computational_adme_converted = {
    "HLM": computational_adme_HLM_converted,
    "KSOL": computational_adme_KSOL_converted,
    "LogD": computational_adme_LogD_converted,
    "MDR1-MDCKII": computational_adme_MDR1_MDCKII_converted,
    "MLM": computational_adme_MLM_converted,
}

##########################
# ASAP-Discovery data, with just the columns we need for each target
##########################
asap_HLM_train = NamedDataset("asap_HLM_train", DATA_DIR_CLEAN / "asap_HLM_train.csv")
asap_KSOL_train = NamedDataset(
    "asap_KSOL_train", DATA_DIR_CLEAN / "asap_KSOL_train.csv"
)
asap_LogD_train = NamedDataset(
    "asap_LogD_train", DATA_DIR_CLEAN / "asap_LogD_train.csv"
)
asap_MDR1_MDCKII_train = NamedDataset(
    "asap_MDR1_MDCKII_train", DATA_DIR_CLEAN / "asap_MDR1_MDCKII_train.csv"
)
asap_MLM_train = NamedDataset("asap_MLM_train", DATA_DIR_CLEAN / "asap_MLM_train.csv")

asap_train_clean = {
    "HLM": asap_HLM_train,
    "KSOL": asap_KSOL_train,
    "LogD": asap_LogD_train,
    "MDR1-MDCKII": asap_MDR1_MDCKII_train,
    "MLM": asap_MLM_train,
}

##########################
# ASAP-Discovery data combined with whatever we've added from Computational-ADME, TDC Commons, etc.
# Should be suitable for training.
##########################
admet_HLM_train_combined = NamedDataset(
    "admet_HLM_train", DATA_DIR_COMBINED / "admet_HLM_train.csv"
)
admet_KSOL_train_combined = NamedDataset(
    "admet_KSOL_train", DATA_DIR_COMBINED / "admet_KSOL_train.csv"
)
admet_LogD_train_combined = NamedDataset(
    "admet_LogD_train", DATA_DIR_COMBINED / "admet_LogD_train.csv"
)
admet_MDR1_MDCKII_train_combined = NamedDataset(
    "admet_MDR1_MDCKII_train", DATA_DIR_COMBINED / "admet_MDR1_MDCKII_train.csv"
)
admet_MLM_train_combined = NamedDataset(
    "admet_MLM_train", DATA_DIR_COMBINED / "admet_MLM_train.csv"
)

admet_train_combined = {
    "HLM": admet_HLM_train_combined,
    "KSOL": admet_KSOL_train_combined,
    "LogD": admet_LogD_train_combined,
    "MDR1-MDCKII": admet_MDR1_MDCKII_train_combined,
    "MLM": admet_MLM_train_combined,
}

##########################
# TDC Commons datasets
##########################
tdc_lipophilicity_az_raw = NamedDataset(
    "tdc_lipophilicity_az_raw", DATA_DIR_RAW / "tdc_lipophilicity_az.csv"
)
tdc_lipophilicity_az_clean = NamedDataset(
    "tdc_lipophilicity_az_raw", DATA_DIR_CLEAN / "tdc_lipophilicity_az.csv"
)
