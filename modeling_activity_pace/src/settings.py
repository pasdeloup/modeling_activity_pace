import datetime as dt


channel_names = ["volume", "repetition", "organicity", "liked"]
n_channels = len(channel_names)

MIN_DATE = dt.date(2022, 1, 1)
MAX_DATE = dt.date(2023, 5, 19)
INSTANT_ZERO = dt.datetime(2022, 12, 26)

ANSWERS_PATH = "data/processed/answers/processed_records.csv"
DICT_ITER_PATH = "results/dict_learning/all_iterations/"

ANSWERS_ACTIVITY_COLUMNS = [
    "activity_wake_up",
    "activity_transport",
    "activity_work",
    "activity_sports",
    "activity_friends",
    "activity_asleep",
]

ACTIVITY_LABELS = [
    "wake up",
    "transport.",
    "work",
    "sports",
    "friends",
    "asleep",
]

PARAM_GRID = {
    "penalty": ["l1", "l2"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "fit_intercept": [True, False],
    "solver": ["liblinear", "saga"],
    "tol": [1e-4, 1e-3, 1e-2],
}

RS = 13
TEST_SIZE = 0.33
N_SUBDIVISION_1HOUR = 7 * 24
SERIES_LENGTH = 166

DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
HOURS = [f"{i}h" for i in range(24)]

time_labels_full = [f"{day},{hour}" for day in DAYS for hour in HOURS]
time_labels = time_labels_full[1:-1]