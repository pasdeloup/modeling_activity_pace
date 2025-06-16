#!/bin/bash

echo "**********************************************"
echo "1. User logs are transformed into time series "
echo "**********************************************"
poetry run python modeling_activity_pace/compute_time_series.py

echo "****************************************"
echo "2. Dictionary Learning algorithm is run "
echo "****************************************"
#poetry run python modeling_activity_pace/compute_dictionary.py

echo "**********************************************************"
echo "3. Selection of the best iteration in dictionary learning "
echo "**********************************************************"
#poetry run python modeling_activity_pace/choose_dictionary.py

echo ""
echo "***********************************************************"
echo "4. Computes baselines scores and scores of PACE embeddings "
echo "***********************************************************"
#poetry run python modeling_activity_pace/compute_baselines.py

echo ""
echo "******************************************"
echo "5. Plots logistic regression coefficients "
echo "   and related statistical reports        "
echo "******************************************"
#poetry run python modeling_activity_pace/analyse_models.py

echo -e ""
echo "******************************"
echo "6. Saves the plot of Figure 1 "
echo "******************************"
#poetry run python modeling_activity_pace/make_fig1.py
