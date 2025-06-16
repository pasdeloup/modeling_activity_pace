# Modeling Activity-Driven Music Listening with PACE

Code for our paper "Modeling Activity-Driven Music Listening with PACE" by L. Marey, B. Sguerra and M. Moussallam, accepted for publication in the proceedings of 2024 ACM SIGIR Conference on Human Information Interaction and Retrieval (CHIIR'24).

## Dependencies

```
pandas==2.1.2
numba==0.58.1

numpy==1.26.1
scikit-learn==1.3.2
statsmodels==0.14.0

matplotlib==3.8.1
seaborn==0.13.0
```

## Docker/Poetry/Makefile

We provide an optionnal Docker+Poetry+Makefile environment for easy deployment

To build the Docker image:
```
make build
```

To launch an interactive session within the container
```
make run-bash
```

To run a script *from the container*
```
poetry run python modeling_activity_pace/<script.py>
```

## Scripts

1. User logs are transformed into time series using `compute_time_series.py`.
2. Dictionary Learning algorithm is run using `compute_dictionary.py`.
3. The selection of the best iteration in dictionary learning is done using `choose_dictionary.py`.
4. `compute_baselines.py` computes baselines scores and scores of PACE embeddings.
5. `analyse_models.py` plots logistic regression coefficients and related statistical reports.
6. `make_fig1.py` saves the plot of Figure 1.

## Data

Input data folder must be organized as follows :


```
pace/
│
└── data/
  └── answers/
      └── processed_records.csv
  ├── streams/
  │ ├── processed_streams_000000000000.csv
  │ ├── ...
  │ └── processed_streams_000000000091.csv
```

Where ```one_year_all_respondents.csv``` files are stream history csv files with columns :  ```user_id, ts_listen, media_id, context_id, context_type, listening_time, context_4```.

```records.csv``` being Records survey answer csv files, with columns ```ResponseId, uid, Status, Progress, Duration (in seconds), Q_consent, B_contexts_deezer_1, B_contexts_deezer_2, B_contexts_deezer_4, B_contexts_deezer_5, B_contexts_deezer_10, B_contexts_deezer_12, E_birth_year, E_age_range, E_gender```.

And ```user_favorites.csv``` having columns  ```user_id, item_id, item_type```.

## About alphacsc

The ```alphacsc``` package enables dictionary learning with multivariate time series :
We slightly modified the package from ```version 0.4.0``` to be able to extract information during dictionary learning.

Dupré La Tour, T., Moreau, T., Jas, M., & Gramfort, A. (2018). Multivariate Convolutional Sparse Coding for Electromagnetic Brain Signals. Advances in Neural Information Processing Systems (NIPS).
Jas, M., Dupré La Tour, T., Şimşekli, U., & Gramfort, A. (2017). Learning the Morphology of Brain Signals Using Alpha-Stable Convolutional Sparse Coding. Advances in Neural Information Processing Systems (NIPS), pages 1099–1108.


## Contact

[research@deezer.com](mailto:research@deezer.com)
