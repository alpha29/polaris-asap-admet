import polars as pl

from polaris_asap_admet.io import (admet_LogD_train_combined, asap_LogD_train,
                                   tdc_lipophilicity_az_clean)


def combine():
    df_tdc = tdc_lipophilicity_az_clean.read()
    df_asap = asap_LogD_train.read()
    df_combined = pl.concat([df_tdc, df_asap], how="vertical")
    admet_LogD_train_combined.save(df_combined)


def make():
    combine()


if __name__ == "__main__":
    make()
