import torch
import hw2_utils as utils
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split



def gaussian_theta(X, y):
    '''
    Arguments:
        X (S x N FloatTensor): features of each object
        y (S LongTensor): label of each object, y[i] = 0/1

    Returns:
        mu (2 x N Float Tensor): MAP estimation of mu in N(mu, sigma2)
        sigma2 (2 x N Float Tensor): MAP estimation of mu in N(mu, sigma2)

    '''
    # Separate samples by class
    X0 = X[y == 0]  # samples with label 0
    X1 = X[y == 1]  # samples with label 1

    # Calculate mean for each class
    mu0 = X0.mean(dim=0)
    mu1 = X1.mean(dim=0)
    mu = torch.stack([mu0, mu1])

    # Calculate variance for each class
    sigma2_0 = X0.var(dim=0, unbiased=False)
    sigma2_1 = X1.var(dim=0, unbiased=False)
    sigma2 = torch.stack([sigma2_0, sigma2_1])

    return mu, sigma2

def gaussian_p(y):
    '''
    Arguments:
        y (S LongTensor): label of each object

    Returns:
        p (float or scalar Float Tensor): MLE of P(Y=0)

    '''
    # MLE of P(Y=0) is the proportion of samples with label 0
    return (y == 0).float().mean()

def gaussian_classify(mu,sigma2, p, X):
    '''
    Arguments:
        mu (2 x N Float Tensor): returned value #1 of `gaussian_MAP`
        sigma2 (2 x N Float Tensor): returned value #2 of `gaussian_MAP`
        p (float or scalar Float Tensor): returned value of `bayes_MLE`
        X (S x N LongTensor): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (S LongTensor): label of each object for classification, y[i] = 0/1

    '''
    # Calculate log likelihood for each class
    # log P(X|Y=k) = -0.5 * sum(log(2*pi*sigma2) + (x - mu)^2 / sigma2)

    # For class 0
    log_likelihood_0 = -0.5 * torch.sum(
        torch.log(2 * torch.pi * sigma2[0]) + ((X - mu[0]) ** 2) / sigma2[0],
        dim=1
    )

    # For class 1
    log_likelihood_1 = -0.5 * torch.sum(
        torch.log(2 * torch.pi * sigma2[1]) + ((X - mu[1]) ** 2) / sigma2[1],
        dim=1
    )

    # Calculate posterior log probabilities using Bayes rule
    # log P(Y=0|X) = log P(X|Y=0) + log P(Y=0)
    # log P(Y=1|X) = log P(X|Y=1) + log P(Y=1)
    log_posterior_0 = log_likelihood_0 + torch.log(p)
    log_posterior_1 = log_likelihood_1 + torch.log(1 - p)

    # Classify based on which posterior is larger
    y = (log_posterior_1 > log_posterior_0).long()

    return y
