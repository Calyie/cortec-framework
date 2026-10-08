"""The public description of the NHANES table: bounds and bin edges from documentation and
clinical convention (WHO BMI categories, ACC/AHA blood-pressure stages), never from the data.
`conditional` and `stratify` are left empty so the tool's auto-configuration derives them, which
is how the paper's NHANES release was made ("a release nobody tuned")."""
from cortec import Schema

SCHEMA = Schema(
    name="nhanes",
    numerical={"age_years": (18, 80), "bmi": (14.0, 60.0), "waist_cm": (55.0, 175.0),
               "systolic_bp": (70, 220), "diastolic_bp": (30, 130)},
    categorical={"sex": ["male", "female"],
                 "race_ethnicity": ["mexican_american", "other_hispanic", "white_nh", "black_nh",
                                    "asian_nh", "other_multi"],
                 "education": ["lt_9th", "9_11th", "hs_grad", "some_college", "college_grad"],
                 "income_bracket": ["under_1x", "1_2x", "2_4x", "over_4x"]},
    target="diabetes", positive="YES", negative="NO",
    bins={"age_years": [18, 30, 40, 50, 60, 70, 80],
          "bmi": [14, 18.5, 25, 30, 35, 40, 60],
          "waist_cm": [55, 80, 94, 102, 120, 175],
          "systolic_bp": [70, 100, 120, 130, 140, 160, 220],
          "diastolic_bp": [30, 60, 80, 90, 100, 130]},
)
