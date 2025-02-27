
import polaris as po
import polars as pl
from polaris.competition import CompetitionSpecification
from typeguard import typechecked
from polaris.dataset import Dataset
from polaris_asap_admet.io import admet_train_clean, test_raw, train_raw, tdc_lipophilicity_az_raw, tdc_lipophilicity_az_clean
from polaris_asap_admet.logger import logger
from polaris_asap_admet.util import print_info
import pandas as pd
CHALLENGE = "antiviral-admet-2025"

####################################
# Polaris competition downloads
####################################

@typechecked
def load_comp(challenge: str = CHALLENGE) -> CompetitionSpecification:
    """
    Cache the competition dataset.

    Run `polaris login` before running this.
    """
    logger.info(f"Loading competition for challenge {challenge}...")
    competition = po.load_competition(f"asap-discovery/{challenge}")
    logger.info("Done. Caching...")
    # when you set this to "skip", polaris throws an exception I don't care to debug
    # cache_dir = competition.cache(if_exists="skip")
    cache_dir = competition.cache()
    logger.info(f"Cached data to {cache_dir}.")
    return competition


@typechecked
def get_df_train_for_comp(
    comp: CompetitionSpecification, save: bool = False
) -> pl.DataFrame:
    """
    Load training data as polars DataFrame
    Polaris has some dumbass bug where converting a competition subset to a dataframe fails, because polaris keeps adding duplicate columns to the DF?
    It doesn't make sense, I didn't dig into it, but this is the magic incantation that keeps that bug from manifesting.
    """
    logger.info(f"Loading training dataframe for comp {comp.name}...")
    df_train = pl.from_pandas(comp.get_train_test_split()[0].as_dataframe())
    print_info(df_train)
    if save:
        logger.info("Saving...")
        train_raw.save(df_train)
    return df_train


@typechecked
def get_df_test_for_comp(
    comp: CompetitionSpecification, save: bool = False
) -> pl.DataFrame:
    logger.info(f"Loading test data for comp {comp.name}...")

    test = comp.get_train_test_split()[1]
    d = {"CXSMILES": test.X}
    df_test = pl.DataFrame(d)
    print_info(df_test)
    if save:
        logger.info("Saving...")
        test_raw.save(df_test)
    return df_test


def download_comp_data():
    logger.info(f"Downloading competition data for challenge {CHALLENGE}...")
    comp = load_comp()
    df_train = get_df_train_for_comp(comp, save=True)
    df_test = get_df_test_for_comp(comp, save=True)
    logger.info("Done.")


TARGETS = [
    "HLM",
    "KSOL",
    "LogD",
    "MDR1-MDCKII",
    "MLM",
]


def split_train_by_targets():
    """
    Create separate training datasets for each target.
    """
    df = train_raw.read()
    for tgt in TARGETS:
        logger.info(f"Splitting training data for target {tgt}...")
        df_tgt = df.select(["CXSMILES", tgt]).filter(pl.col(tgt).is_not_null())
        print_info(df_tgt)
        logger.info(f"Saving {tgt}...")
        admet_train_clean[tgt].save(df_tgt)
    logger.info("Done.")

####################################
# TDC Commons downloads
####################################


def get_tdc_lipo_az_raw(ds_name: str = "tdcommons/lipophilicity-astrazeneca", save: bool = False) -> Dataset:
    """
    Fetch dataset from Polaris hub.
    """
    dataset = po.load_dataset(ds_name)
    df = pl.from_pandas(pd.DataFrame(dataset.table))
    print_info(df)
    if save:
        logger.info("Saving...")
        tdc_lipophilicity_az_raw.save(df)
    return df


def prep_tdc_lipo_az(df: pl.DataFrame, save: bool = False) -> pl.DataFrame:
    """
    Prep the raw AZ dataset for use in our model training - rename columns, etc.
    """
    df = df.select(["Drug", "Y"]).rename({"Drug": "CXSMILES", "Y": "LogD"})
    if save:
        tdc_lipophilicity_az_clean.save(df)
    return df

def make_tdc_lipo_az(save: bool = True) -> pl.DataFrame:
    df = get_tdc_lipo_az_raw("tdcommons/lipophilicity-astrazeneca", save=save)
    df = prep_tdc_lipo_az(df, save=save)
    return df
