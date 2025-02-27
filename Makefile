all:
	echo "Hi there, do something else"

download-computational-adme-data:
	wget -O data/raw/ADME_public_set_3521.csv https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/refs/heads/main/ADME_public_set_3521.csv

download-comp-data:
	python -c "from polaris_asap_admet.download import download_comp_data; download_comp_data();"

split-train-by-targets:
	python -c "from polaris_asap_admet.download import split_train_by_targets; split_train_by_targets();"

split-computational-adme:
	python -c "from polaris_asap_admet.prep_computational_adme import split_computational_adme; split_computational_adme();"

tdc-lipo-az:
	python -c "from polaris_asap_admet.download import make_tdc_lipo_az; make_tdc_lipo_az();"

prep-data-hlm:
	python -c "from polaris_asap_admet.prep_data_hlm import make; make();"

prep-data-ksol:
	python -c "from polaris_asap_admet.prep_data_ksol import make; make();"

prep-data-logd:
	python -c "from polaris_asap_admet.prep_data_logd import make; make();"

prep-data-mdr1:
	python -c "from polaris_asap_admet.prep_data_mdr1_mdckii import make; make();"

prep-data-mlm:
	python -c "from polaris_asap_admet.prep_data_mlm import make; make();"

prep-data: prep-data-hlm prep-data-ksol prep-data-logd prep-data-mdr1 prep-data-mlm

start-tensorboard:
	tensorboard --logdir runs/ --port 6007

open-tensorboard:
	open http://localhost:6007/


run-hlm:
	python run_chemprop.py HLM data/combined/admet_HLM_train.csv data/raw/test_raw.csv

run-ksol:
	python run_chemprop.py KSOL data/combined/admet_KSOL_train.csv data/raw/test_raw.csv

run-logd:
	python run_chemprop.py LOGD data/combined/admet_LogD_train.csv data/raw/test_raw.csv

run-mdr1:
	python run_chemprop.py MDR1-MDCKII data/combined/admet_MDR1_MDCKII_train.csv data/raw/test_raw.csv

run-mlm:
	python run_chemprop.py MLM data/combined/admet_MLM_train.csv data/raw/test_raw.csv
