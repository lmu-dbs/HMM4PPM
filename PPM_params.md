# PPM Parameter settings

We provide a full set of model parameters used for training and evaluating the PPM models. This overview is referenced in the paper within footnote 6. Each model is executed once for a given Markov order of $h \in [1, 2, 3]$ for the specified set of parameters below.

## BEST
We use the BEST proposed model by [[R2]](#R2) and specify the originally proposed process stages as the newly identified HMM process phases. The parameter set for the model is specified as follows:

|Parameter          |   Values|
| --------          |   ------- |
|pattern_size       |   $[3, 5, 7, 11, 17, 21]$ |
|min_freq           |   $0.00000001$|
|selection_method   |   PROB_LEN_DIST|

## Most-probable path
The most-probable path algorithm previously proposed for suffix prediction by [[R3]](#R3) is extended by HMM-based process phases. We train phase-specific transition probabilities and integrate the phase transition probabilities for sequence extrapolation into the algorithm. Since the algorithm itself is free of additional parameters, we perform single runs of the algorithm for the given set of HMMs used for phase identification on a given dataset.

## Classification and Regression Tree (CART)
We implemented a sequence-trained CART model that is based on `sklearn.tree.DecisionTreeClassifier` for categorical and `sklearn.tree.DecisionTreeRegressor` for continuous target variables. The tree model uses a parameter `seq_len` that specifies the amount of history used in the training of the model. We test models with parameters from the following parameter set:

|Parameter          |   Values|
| --------          |   ------- |
|seq_len            |   $[3, 5, 10]$ |


## LSTM
The LSTM is adapted from [[R5]](#R5) and extended by process phases as additional feature variable. We do not include the role embedding from [[R5]](#R5) as to keep the feature space between the HMM phase identification and the PPM model training identical. We test models with parameters from the following parameter set:
|Parameter                  |   Values|
| --------                  |   ------- |
|seq_len                    |   $[3, 5, 10]$ |
|hidden_size                |   $100$ |
|epochs                     |   $200$|
|early_stopping_patience    |   $42$|
|learning_rate              |   $0.001$|
|batch_size                 |   $32$|
|dropout                    |   $0.2$|
|variant                    |   $[\text{'full\textunderscore shared'}, \text{'shared\textunderscore categorical'}]$|

# References

<a id="R2">[R2]</a> Simon Rauch, Christian M. M. Frey, Andrea Maldonado, Daniel Schuster, Gabriel Tavares, and Thomas Seidl. 2026. Hierarchical structuring of bilaterally expanding subtrace patterns for efficient tree-based activity suffix prediction. Process Science 3, 1 (2026). doi:10.1007/s44311-026-00050-y

<a id="R3">[R3]</a> Sjoerd van der Spoel, Maurice van Keulen, and Chintan Amrit. 2013. Process Prediction in Noisy Data Sets: A Case Study in a Dutch Hospital. Springer, Berlin, Heidelberg, 60–83. doi:10.1007/978-3-642-40919-6_4

<a id="R5">[R5]</a> Manuel Camargo, Marlon Dumas, and Oscar González-Rojas. 2019. Learning Accurate LSTM Models of Business Processes. Springer International Publishing, 286–302. doi:10.1007/978-3-030-26619-6_19