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


train_raw = NamedDataset(name="train_raw", filepath=DATA_DIR_RAW / "train_raw.csv")
test_raw = NamedDataset(name="test_raw", filepath=DATA_DIR_RAW / "test_raw.csv")
computational_adme_raw = NamedDataset(
    name="computational_adme_raw", filepath=DATA_DIR_RAW / "ADME_public_set_3521.csv"
)

##########################
# ASAP-Discovery data, with just the columns we need for each target
##########################
HLM_train = NamedDataset("admet_HLM_train", DATA_DIR_CLEAN / "admet_HLM_train.csv")
KSOL_train = NamedDataset("admet_KSOL_train", DATA_DIR_CLEAN / "admet_KSOL_train.csv")
LogD_train = NamedDataset("admet_LogD_train", DATA_DIR_CLEAN / "admet_LogD_train.csv")
MDR1_MDCKII_train = NamedDataset(
    "admet_MDR1_MDCKII_train", DATA_DIR_CLEAN / "admet_MDR1_MDCKII_train.csv"
)
MLM_train = NamedDataset("admet_MLM_train", DATA_DIR_CLEAN / "admet_MLM_train.csv")

admet_train_clean = {
    "HLM": HLM_train,
    "KSOL": KSOL_train,
    "LogD": LogD_train,
    "MDR1-MDCKII": MDR1_MDCKII_train,
    "MLM": MLM_train,
}

##########################
# ASAP-Discovery data combined with whatever we've added from Computational-ADME, TDC Commons, etc.
# Should be suitable for training.
##########################
HLM_train_combined = NamedDataset("admet_HLM_train", DATA_DIR_COMBINED / "admet_HLM_train.csv")
KSOL_train_combined = NamedDataset("admet_KSOL_train", DATA_DIR_COMBINED / "admet_KSOL_train.csv")
LogD_train_combined = NamedDataset("admet_LogD_train", DATA_DIR_COMBINED / "admet_LogD_train.csv")
MDR1_MDCKII_train_combined = NamedDataset(
    "admet_MDR1_MDCKII_train", DATA_DIR_COMBINED / "admet_MDR1_MDCKII_train.csv"
)
MLM_train_combined = NamedDataset("admet_MLM_train", DATA_DIR_COMBINED / "admet_MLM_train.csv")

admet_train_combined = {
    "HLM": HLM_train_combined,
    "KSOL": KSOL_train_combined,
    "LogD": LogD_train_combined,
    "MDR1-MDCKII": MDR1_MDCKII_train_combined,
    "MLM": MLM_train_combined,
}

##########################
# TDC Commons datasets
##########################
tdc_lipophilicity_az_raw = NamedDataset("tdc_lipophilicity_az_raw", DATA_DIR_RAW / "tdc_lipophilicity_az.csv")
tdc_lipophilicity_az_clean = NamedDataset("tdc_lipophilicity_az_raw", DATA_DIR_CLEAN / "tdc_lipophilicity_az.csv")
