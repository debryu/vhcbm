"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""
import numpy as np
import scipy
from sklearn import ensemble
import pdb
import torch
from tqdm import tqdm 
from loguru import logger

def _compute_dci(x_train, ys_train, x_test, ys_test):
  """Computes score based on both training and testing codes and factors."""
  L = len(ys_train)
  concepts = []
  excluded_concepts = []
  for i in range(L):
      if len(ys_train[i].unique()) == 1:
        excluded_concepts.append(i)
      else:
        concepts.append(i)
  ys_train = ys_train[concepts]
  ys_test = ys_test[concepts]
  logger.debug('Excluded concepts:', excluded_concepts,' due to not having a single occurence in the training set')
  

  scores = {}
  importance_matrix, train_err, test_err = compute_importance_gbt(
      x_train, ys_train, x_test, ys_test)

  assert importance_matrix.shape[0] == x_train.shape[0]
  assert importance_matrix.shape[1] == ys_train.shape[0]
  scores["informativeness_train"] = train_err
  scores["informativeness_test"] = test_err
  scores["disentanglement"] = disentanglement(importance_matrix)
  scores["completeness"] = completeness(importance_matrix)
  scores['importance_matrix'] = importance_matrix
  return scores


def compute_importance_gbt(x_train, y_train, x_test, y_test):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  logger.info(f'Factors: {num_factors}, Concepts: {num_codes}')
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  logger.debug(f'Size of importance matrix: {importance_matrix.shape}')
  train_loss = []
  test_loss = []
  for i in tqdm(range(num_factors), desc='Computing DCI'):
    model = ensemble.GradientBoostingClassifier()
    model.fit(x_train.T, y_train[i, :])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    
  return importance_matrix, 0,0 


def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  m = scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""

  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)