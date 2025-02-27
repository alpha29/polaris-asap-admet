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
