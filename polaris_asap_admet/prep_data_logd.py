import polars as pl
from polaris_asap_admet.io import tdc_lipophilicity_az_clean, LogD_train, LogD_train_combined

def combine():
    df_tdc = tdc_lipophilicity_az_clean.read()
    df_asap = LogD_train.read()
    df_combined = pl.concat([df_tdc, df_asap], how="vertical")
    LogD_train_combined.save(df_combined)

def make():
    combine()

if __name__ == "__main__":
    make()
