# PPM Parameter settings

We provide a full set of model parameters used for training and evaluating the PPM models. This overview is referenced in the paper within footnote 6. Each model is executed once for a given Markov order of $h \in [1, 2, 3]$ for the specified set of parameters below.

## BEST
We use the BEST proposed model by [[R2]](#R2) and specify the originally proposed process stages as the newly identified HMM process phases. The parameter set for the model is specified as follows:

|Parameter          |   Values|
| --------          |   ------- |
|pattern_size       |   $[3, 5, 7, 11, 17, 21]$ |
|min_freq           |   $0.00000001$|
|selection_method   |   $`\mathtt{PROB\_LEN\_DIST}`$|

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
|variant                    |   $`[\mathtt{'full\_shared'}, \mathtt{'shared\_categorical'}]`$|

# References

<a id="R2">[R2]</a> Simon Rauch, Christian M. M. Frey, Andrea Maldonado, Daniel Schuster, Gabriel Tavares, and Thomas Seidl. 2026. Hierarchical structuring of bilaterally expanding subtrace patterns for efficient tree-based activity suffix prediction. Process Science 3, 1 (2026). doi:<https://doi.org/10.1007/s44311-026-00050-y>

<a id="R4">[R4]</a> Manuel Camargo, Marlon Dumas, and Oscar González-Rojas. 2019. Learning Accurate LSTM Models of Business Processes. Springer International Publishing, 286–302. doi:<https://doi.org/10.1007/978-3-030-26619-6_19>