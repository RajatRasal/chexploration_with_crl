import os
import shutil
from dotenv import load_dotenv

load_dotenv()


def main():
    chexpert_data_folder = os.path.join(os.getenv("CHEXPERT_FOLDER"), "meta")
    shutil.copy(
        os.path.join(chexpert_data_folder, "CHEXPERT DEMO.xlsx"),
        "datafiles/chexpert/",
    )
    shutil.copy(
        os.path.join(chexpert_data_folder, "train.csv"),
        "datafiles/chexpert/",
    )
    shutil.copy(
        os.path.join(os.getenv("MIMIC_CXR_FOLDER"), "mimic-cxr-2.0.0-metadata.csv.gz"),
        "datafiles/mimic/",
    )
    shutil.copy(
        os.path.join(os.getenv("MIMIC_CXR_FOLDER"), "mimic-cxr-2.0.0-chexpert.csv.gz"),
        "datafiles/mimic/",
    )
    mimic_iv_data_folder = os.path.join(os.getenv("MIMIC_IV_FOLDER"), "mimic-iv-1.0/core")
    shutil.copy(
        os.path.join(mimic_iv_data_folder, "admissions.csv"),
        "datafiles/mimic/",
    )
    shutil.copy(
        os.path.join(mimic_iv_data_folder, "patients.csv.gz"),
        "datafiles/mimic/",
    )
