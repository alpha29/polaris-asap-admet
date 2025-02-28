# polaris-asap-admet

Code and data for the [ADMET (Absorption-Distribution-Metabolism-Excretion-Toxicology)](https://polarishub.io/competitions/asap-discovery/antiviral-admet-2025) sub-challenge of the [Polaris ASAP Discovery Antiviral Competition](https://polarishub.io/blog/antiviral-competition).

## Prerequisites

`polaris-asap-admet` uses `uv` for dependency and virtual-environment management.  So, make sure you have [installed uv system-wide](https://docs.astral.sh/uv/getting-started/installation/), e.g.:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```
# Create venv and install dependencies
uv sync
source .venv/bin/activate

# Setup .env
cp .env.example .env
source .env
```

## Usage

```
polaris login --overwrite
make download
make split-by-targets
make prep-data

# or, run chemprop for individual targets via (e.g.) `make run-hlm`
make run
```


## Author

[C.J. Brown](cbrown@alpha29.com)
