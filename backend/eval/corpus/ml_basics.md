# Machine Learning Basics — Lecture Notes

## 1. Overfitting

Overfitting occurs when a model learns the training data too well, including
the noise, so it fails to generalise to unseen examples. A classic symptom is
a large gap between training accuracy and validation accuracy. Remedies
include regularisation (L1/L2), early stopping, dropout for neural networks,
and collecting more data.

The bias–variance tradeoff frames this formally: overfitting corresponds to
high variance (model is too sensitive to the specific training sample) and
low bias. Underfitting is the opposite — the model is too rigid to capture
the underlying pattern.

## 2. Cross-Validation

k-fold cross-validation partitions the training data into k equally sized
folds. For each fold, the model is trained on the remaining k-1 folds and
evaluated on the held-out fold. Averaging across folds yields a more stable
estimate of generalisation error than a single train/test split, especially
on small datasets. Common choices are k=5 and k=10.

Stratified k-fold preserves the class distribution within each fold and is
standard for classification with imbalanced classes.

## 3. Regularisation

L2 regularisation (ridge) penalises the squared magnitude of the weights and
produces smooth, small-weight solutions. L1 regularisation (lasso) penalises
the absolute magnitude and tends to produce sparse solutions — many weights
go exactly to zero, which doubles as feature selection.

Elastic Net combines L1 and L2 with a mixing hyperparameter. It is useful
when features are correlated: pure lasso may pick one at random, while
elastic net spreads the weight across the correlated group.

## 4. Gradient Descent

Gradient descent minimises a differentiable loss by iteratively taking
steps in the direction of the negative gradient. The learning rate
controls the step size; too large and the optimiser diverges, too small
and training stalls. Stochastic gradient descent (SGD) uses one example
per step, mini-batch SGD a handful, and batch gradient descent the full
dataset.

Momentum accumulates a velocity term from past gradients and smooths
updates. Adam adapts per-parameter learning rates using first- and
second-moment estimates of the gradient and is the default choice for
many deep-learning tasks.
