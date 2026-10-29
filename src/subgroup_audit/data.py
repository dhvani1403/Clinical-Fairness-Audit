"""Dataset loaders for the three public datasets used in the replication.

Each loader returns a :class:`Dataset` with

* ``X``          -- model features (numeric, one-hot encoded), protected attributes included,
                    as in Kearns et al. and the GerryFair reference code;
* ``y``          -- binary label (1 = the "positive" outcome described in ``label_desc``);
* ``groups``     -- protected attributes as *categorical* columns, used for the
                    single-attribute check and the enumerated intersectional audit;
* ``X_prime``    -- protected attributes as *numeric* columns, used by the
                    linear-threshold auditor from the paper.

Raw files are read from ``data/raw`` (see ``scripts/download_data.sh``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    groups: pd.DataFrame
    X_prime: pd.DataFrame
    label_desc: str
    notes: list[str] = field(default_factory=list)
    # Optional per-row measure of clinical need (e.g. number of chronic conditions),
    # used for the Obermeyer-style check of whether risk scores track need equally.
    need: np.ndarray | None = None


# --------------------------------------------------------------------------- #
# Communities and Crime (UCI) -- the dataset used in Kearns et al. (2018)
# --------------------------------------------------------------------------- #

# The 18 protected attributes used by the authors' GerryFair repository
# (dataset/communities_protected.csv).
CC_PROTECTED = [
    "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp",
    "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap",
    "PctSpeakEnglOnly", "PctNotSpeakEnglWell", "PctForeignBorn",
    "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian",
]

# A readable subset, binarised at the median, for the enumerated audit.
# Communities are places, not people, so "groups" here are kinds of community
# (e.g. "high share of Black residents x high share of residents 65+").
CC_GROUP_ATTRS = {
    "racepctblack": "pct_black",
    "racePctWhite": "pct_white",
    "racePctHisp": "pct_hispanic",
    "racePctAsian": "pct_asian",
    "PctForeignBorn": "pct_foreign_born",
    "agePct65up": "pct_age_65plus",
    "agePct12t29": "pct_age_12_29",
}


def load_communities(raw_dir: Path = RAW_DIR) -> Dataset:
    names = [
        line.split()[1]
        for line in (raw_dir / "communities.names").read_text(errors="ignore").splitlines()
        if line.lower().startswith("@attribute")
    ]
    df = pd.read_csv(raw_dir / "communities.data", header=None, names=names, na_values="?")
    df = df.drop(columns=["state", "county", "community", "communityname", "fold"])
    # Police (LEMAS) columns are missing for ~84% of communities; impute the median.
    df = df.fillna(df.median(numeric_only=True))

    target = df.pop("ViolentCrimesPerPop")
    # Top 30% most violent communities = 1, matching the GerryFair preprocessing (583/1994).
    y = (target >= target.quantile(0.70)).astype(int).to_numpy()

    groups = pd.DataFrame({
        label: np.where(df[col] > df[col].median(), "high", "low")
        for col, label in CC_GROUP_ATTRS.items()
    })
    return Dataset(
        name="communities",
        X=df.astype(float),
        y=y,
        groups=groups,
        X_prime=df[CC_PROTECTED].astype(float),
        label_desc="community is in the top 30% for violent crime per capita",
        notes=[
            "Rows are communities, not people; protected attributes are demographic shares.",
            "Missing police statistics imputed with the column median.",
        ],
    )


# --------------------------------------------------------------------------- #
# UCI Adult income
# --------------------------------------------------------------------------- #

ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]


def _age_band(age: pd.Series, edges: list[int], labels: list[str]) -> pd.Series:
    return pd.cut(age, bins=[-np.inf, *edges, np.inf], labels=labels, right=False).astype(str)


def load_adult(raw_dir: Path = RAW_DIR) -> Dataset:
    parts = [
        pd.read_csv(raw_dir / f, header=None, names=ADULT_COLS, skipinitialspace=True,
                    na_values="?", comment="|")
        for f in ("adult.data", "adult.test")
    ]
    df = pd.concat(parts, ignore_index=True).dropna()
    y = df.pop("income").str.rstrip(".").eq(">50K").astype(int).to_numpy()
    df = df.drop(columns=["fnlwgt", "education"])  # sampling weight; duplicate of education_num

    race = df["race"].replace({"Amer-Indian-Eskimo": "Other", "Asian-Pac-Islander": "Asian-PI"})
    groups = pd.DataFrame({
        "sex": df["sex"].to_numpy(),
        "race": race.to_numpy(),
        "age_band": _age_band(df["age"], [30, 45, 60], ["<30", "30-44", "45-59", "60+"]).to_numpy(),
    })
    X = pd.get_dummies(df, drop_first=False, dtype=float)
    X_prime = pd.get_dummies(groups, dtype=float)
    X_prime["age"] = df["age"].to_numpy(dtype=float)
    return Dataset(
        name="adult",
        X=X.reset_index(drop=True),
        y=y,
        groups=groups.reset_index(drop=True),
        X_prime=X_prime.reset_index(drop=True),
        label_desc="income above $50K",
        notes=["Train and test files concatenated, rows with missing values dropped."],
    )


# --------------------------------------------------------------------------- #
# Diabetes 130-US hospitals (UCI) -- 30-day readmission
# --------------------------------------------------------------------------- #

# Discharge dispositions that make readmission impossible (expired / hospice).
_DIABETES_EXCLUDE_DISCHARGE = {11, 13, 14, 19, 20, 21}
_DIABETES_CATEGORICAL = [
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "max_glu_serum", "A1Cresult", "insulin", "metformin", "change", "diabetesMed",
]
_DIABETES_NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]


def _icd9_chapter(code: pd.Series) -> pd.Series:
    """Map primary diagnosis ICD-9 codes to coarse chapters (Strack et al., 2014)."""
    num = pd.to_numeric(code, errors="coerce")
    out = pd.Series("other", index=code.index)
    out[code.astype(str).str.startswith(("V", "E"))] = "other"
    out[((num >= 390) & (num <= 459)) | (num == 785)] = "circulatory"
    out[((num >= 460) & (num <= 519)) | (num == 786)] = "respiratory"
    out[((num >= 520) & (num <= 579)) | (num == 787)] = "digestive"
    out[(num >= 250) & (num < 251)] = "diabetes"
    out[(num >= 800) & (num <= 999)] = "injury"
    out[(num >= 710) & (num <= 739)] = "musculoskeletal"
    out[((num >= 580) & (num <= 629)) | (num == 788)] = "genitourinary"
    out[(num >= 140) & (num <= 239)] = "neoplasms"
    return out


def load_diabetes(raw_dir: Path = RAW_DIR) -> Dataset:
    df = pd.read_csv(raw_dir / "diabetic_data.csv", na_values="?", low_memory=False)
    # One encounter per patient (the first) so the same person can't sit in train and test.
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    df = df[~df["discharge_disposition_id"].isin(_DIABETES_EXCLUDE_DISCHARGE)]
    df = df[df["gender"].isin(["Female", "Male"])].reset_index(drop=True)

    y = df["readmitted"].eq("<30").astype(int).to_numpy()

    race = df["race"].fillna("Unknown").replace({"AfricanAmerican": "Black", "Caucasian": "White"})
    decade = df["age"].str.extract(r"\[(\d+)-")[0].astype(int)
    age_band = pd.cut(decade, bins=[-1, 39, 59, 79, 200],
                      labels=["<40", "40-59", "60-79", "80+"]).astype(str)
    groups = pd.DataFrame({"sex": df["gender"], "race": race, "age_band": age_band})

    feats = df[_DIABETES_NUMERIC].astype(float).copy()
    feats["age_decade"] = decade.astype(float)
    feats["primary_dx"] = _icd9_chapter(df["diag_1"])
    cats = df[_DIABETES_CATEGORICAL].astype(str).fillna("None")
    # Race-blind model (race is used to audit, not to predict), as is common for
    # deployed clinical risk scores; sex and age stay in as standard clinical inputs.
    X = pd.concat([feats, cats, groups[["sex"]]], axis=1)
    X = pd.get_dummies(X, dtype=float)

    X_prime = pd.get_dummies(groups[["sex", "race"]], dtype=float)
    X_prime["age_decade"] = decade.astype(float)
    return Dataset(
        name="diabetes",
        X=X,
        y=y,
        groups=groups,
        X_prime=X_prime,
        label_desc="patient readmitted within 30 days",
        notes=[
            "First encounter per patient only; patients discharged to hospice or who died excluded.",
            "Race is not a model input; it is used only for the audit.",
            "A false negative is a high-risk patient the model misses, so FNR is the clinically "
            "relevant error; we audit both FPR and FNR.",
        ],
    )


# --------------------------------------------------------------------------- #
# MEPS 2015 (AHRQ Medical Expenditure Panel Survey, full-year file HC-181)
# --------------------------------------------------------------------------- #

MEPS_CONDITIONS = {
    "HIBPDX": "high blood pressure", "CHDDX": "coronary heart disease", "ANGIDX": "angina",
    "MIDX": "heart attack", "OHRTDX": "other heart disease", "STRKDX": "stroke",
    "EMPHDX": "emphysema", "CHBRON31": "chronic bronchitis", "CHOLDX": "high cholesterol",
    "CANCERDX": "cancer", "DIABDX": "diabetes", "ARTHDX": "arthritis", "ASTHDX": "asthma",
}
_MEPS_CATEGORICAL = ["REGION15", "MARRY15X", "POVCAT15", "INSCOV15", "RTHLTH31", "MNHLTH31",
                     "ADSMOK42", "ACTLIM31", "WLKLIM31", *MEPS_CONDITIONS]
_MEPS_NUMERIC = ["PCS42", "MCS42", "K6SUM42", "PHQ242"]
_MEPS_VISITS = ["OBTOTV15", "OPTOTV15", "ERTOT15", "IPNGTD15", "HHTOTD15"]


def load_meps(raw_dir: Path = RAW_DIR) -> Dataset:
    cols = ["AGE15X", "SEX", "RACEV2X", "HISPANX", "PERWT15F",
            *_MEPS_CATEGORICAL, *_MEPS_NUMERIC, *_MEPS_VISITS]
    df = pd.read_sas(raw_dir / "meps" / "h181.ssp", format="xport")[cols]
    df = df[(df["PERWT15F"] > 0) & (df["AGE15X"] >= 18)].reset_index(drop=True)

    # High utilisation = 10+ visits/nights in the year (office, outpatient, ER,
    # inpatient nights, home health): the label used in AIF360's MEPS benchmark. Like
    # the cost label in Obermeyer et al. (2019), it is a *proxy* for health need.
    visits = df[_MEPS_VISITS].clip(lower=0).sum(axis=1)
    y = (visits >= 10).astype(int).to_numpy()

    race = np.select(
        [df["HISPANX"] == 1, df["RACEV2X"] == 1, df["RACEV2X"] == 2, df["RACEV2X"].isin([4, 5, 6, 10, 11])],
        ["Hispanic", "White", "Black", "Asian"], default="Other")
    age_band = pd.cut(df["AGE15X"], bins=[17, 29, 44, 64, 200], labels=["18-29", "30-44", "45-64", "65+"]).astype(str)
    groups = pd.DataFrame({"sex": np.where(df["SEX"] == 1, "Male", "Female"), "race": race, "age_band": age_band})

    # Negative MEPS codes (-1 inapplicable, -7/-8/-9 refused/unknown) become "missing".
    cats = df[_MEPS_CATEGORICAL].astype(int).where(df[_MEPS_CATEGORICAL] >= 0, -1).astype(str)
    nums = df[_MEPS_NUMERIC].where(df[_MEPS_NUMERIC] >= 0)
    nums = pd.concat([nums.fillna(nums.median()), nums.isna().add_suffix("_missing").astype(float)], axis=1)
    feats = pd.concat([df[["AGE15X"]].astype(float), nums, cats, groups[["sex"]]], axis=1)
    X = pd.get_dummies(feats, dtype=float)

    X_prime = pd.get_dummies(groups[["sex", "race"]], dtype=float)
    X_prime["age"] = df["AGE15X"].astype(float)
    need = (df[list(MEPS_CONDITIONS)] == 1).sum(axis=1).to_numpy()
    return Dataset(
        name="meps",
        X=X,
        y=y,
        groups=groups,
        X_prime=X_prime,
        label_desc="10 or more healthcare visits in the year (high utilisation)",
        notes=[
            "Adults (18+) in the MEPS 2015 full-year consolidated file with a positive survey weight.",
            "Race is not a model input; it is used only for the audit.",
            "Survey weights are not used, so rates describe the sample, not the US population.",
        ],
        need=need,
    )


LOADERS = {"communities": load_communities, "adult": load_adult, "diabetes": load_diabetes, "meps": load_meps}


def load(name: str, raw_dir: Path = RAW_DIR) -> Dataset:
    return LOADERS[name](raw_dir)
