# Code for analysing the Covidom database

This codebase gathers the different notebooks used to exploit the data from Covidom to uncover patterns in PCR results

## How to use

Install packages from `requirements.txt`.


- `univariate_analysis.ipynb` perform univariate analysis of testing factors and PCR results. It produces the plot from Figure 1 and 2

- `train_tree.ipynb` produces Figure 3 and related decision tree figures, as well as temporal figures.

- `eval_tree.ipynb` produces Figure 4.

The data from `data/shuffled.csv` are generated data with the same columns as the one that we use for analysis -- it is therefore *fake data*, but can be used with our scripts.

The real data is available upon request. It has to be analysed on APHP servers.
