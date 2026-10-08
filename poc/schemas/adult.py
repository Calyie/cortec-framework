"""The public description of the UCI Adult table: bounds and bin edges from the dataset's
documentation, category lists from adult.names plus the "?" the file uses for a missing value, never from the data. `conditional` and `stratify`
are left empty so the tool's auto-configuration derives them. The paper's Adult tables use a
hand-tuned hierarchy in the research harness (education and hours bands); this schema shows the
shipped tool's turn-key path on the same data, so its numbers are not the paper's Table 3."""
from cortec import Schema

SCHEMA = Schema(
    name="adult",
    numerical={"age": (17, 90), "fnlwgt": (12285, 1490400), "education_num": (1, 16),
               "capital_gain": (0, 99999), "capital_loss": (0, 4356), "hours_per_week": (1, 99)},
    categorical={
        "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
                      "State-gov", "Without-pay", "Never-worked", "?"],
        "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm",
                      "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate",
                      "5th-6th", "Preschool"],
        "marital_status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                           "Married-spouse-absent", "Married-AF-spouse"],
        "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                       "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                       "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                       "Armed-Forces", "?"],
        "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
        "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
        "sex": ["Female", "Male"],
        "native_country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                           "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
                           "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
                           "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
                           "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
                           "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                           "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands", "?"],
    },
    target="income", positive=">50K", negative="<=50K",
    bins={"age": [17, 25, 30, 35, 40, 45, 50, 55, 60, 70, 90],
          "fnlwgt": [12285, 50000, 100000, 150000, 200000, 250000, 300000, 400000, 600000, 1490400],
          "education_num": [1, 5, 9, 10, 11, 12, 13, 14, 15, 16],
          "capital_gain": [0, 1, 1000, 3000, 5000, 7500, 10000, 20000, 50000, 99999],
          "capital_loss": [0, 1, 500, 1000, 1500, 2000, 2500, 3000, 4356],
          "hours_per_week": [1, 10, 20, 30, 35, 40, 41, 45, 50, 60, 99]},
)
