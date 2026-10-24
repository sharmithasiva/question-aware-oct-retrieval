import pandas as pd
import pytest

from oct_retrieval import config


def _clin_rows(subject, eye_id, patient_id, visits, trial="TREX DME", arm="GILA", lat="OD", n_img=3):
    """visits: list of (label, cst, bcva). Builds raw image-level rows like Clinical_Data_Images.xlsx."""
    rows = []
    for label, cst, bcva in visits:
        for i in range(n_img):
            if trial == "TREX DME":
                path = f"/TREX DME/{arm}/{subject}/{label}/{lat}/TREXJ_{i:06d}.tif"
            else:
                path = f"/Prime_FULL/{subject}/{label}/{lat}/{i}.tif"
            rows.append(dict(File_Path=path, BCVA=bcva, CST=cst, Eye_ID=eye_id, Patient_ID=patient_id))
    return rows


@pytest.fixture
def raw_clinical():
    from oct_retrieval.load import _with_parsed_paths
    rows = (
        _clin_rows("0201GOD", 1, 201, [("V1", 500, 60), ("V2", 400, 65), ("V10", 300, 70), ("V3", 350, 68)])
        + _clin_rows("0201TOS", 2, 201, [("V1", 280, 80), ("V2", 290, 79)], arm="TREX", lat="OS")
        + _clin_rows("01-001", 58, 58, [("W0", 275, 97), ("W4", None, None), ("W12", 265, 98)], trial="Prime_FULL",
                     lat="OS")
    )
    return _with_parsed_paths(pd.DataFrame(rows), "File_Path", "synthetic.xlsx")


@pytest.fixture(scope="session")
def real_data():
    if not config.CLINICAL_XLSX.exists():
        pytest.skip("OLIVES label files not present")
    from oct_retrieval.episodes import build_all
    return build_all()
