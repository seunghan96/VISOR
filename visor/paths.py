"""Locations of data and third-party code. Every path is configurable with an environment variable.

Generated files (candidate dumps, checkpoints, results) are written relative to the working directory:
sel_dump/, mf_runs/, selector_runs/, results/, px_rows/, px_tsfm_out/.
"""
import os

# Root of the raw and prepared data (see README, "Data preparation").
DATA = os.environ.get("VISOR_DATA", "data")
# Prepared FNSPID daily candles and news (scripts/data/build_fnspid*.py, build_kit_mem.py).
MEM = os.environ.get("VISOR_MEM", os.path.join(DATA, "kit_mem"))
FNSPID = os.path.join(MEM, "fnspid")
# FNSPID raw release (Stock_price/full_history.zip, Stock_news/*.csv).
FNSPID_RAW = os.environ.get("VISOR_FNSPID_RAW", os.path.join(DATA, "raw", "fnspid"))
# Time-MMD csv files as preprocessed by MM-TSFlib (one folder per dataset).
TMMD_RAW = os.environ.get("VISOR_TMMD_RAW", os.path.join(DATA, "raw", "time_mmd", "mmtsflib", "data"))
# FinTexTS parquet files.
FINTEXTS_RAW = os.environ.get("VISOR_FINTEXTS_RAW", os.path.join(DATA, "fintexts_raw", "data"))

# Third-party repositories (cloned by the user, never shipped with this code).
KIT = os.environ.get("KIT_ROOT", "third_party/KiT")              # https://github.com/Luciferbobo/KiT
MMTSFLIB = os.environ.get("MMTSFLIB_ROOT", "third_party/MM-TSFlib")  # https://github.com/AdityaLab/MM-TSFlib
TATS_DATA = os.environ.get("TATS_DATA", "third_party/TaTS/data")  # https://github.com/iDEA-iSAIL-Lab-UIUC/TaTS
CFA = os.environ.get("CFA_ROOT", "third_party/CFA")               # released code of the CFA paper
