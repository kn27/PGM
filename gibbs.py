#implelement gibbs sampling for mixture models
#start from the simplest case - Gaussian mixture models 
#think about real world model
#implement nonnegative matrix factorization? 
#poisson hierachial model

import random
from numpy.random import dirichlet, multinomial, multivariate_normal
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

def generative(c = 3, n = 1000, alpha = None, sigma0 = 2, lambda0 = 1):
    c = 3
    if alpha is None:
        alpha = [1/c] * c  
    #theta = dirichlet(alpha)
    theta = [1/c] * c
    beta = [[multivariate_normal(mean=[0,0],cov=np.array([[lambda0,0],[0,lambda0]])), np.array([[sigma0,0], [0, sigma0]]] for _ in range(c)]
    #mu = [multivariate_normal(mean=[0,0],cov=np.array([[sigma0,0],[0,sigma0]])) for _ in range(c)]
    #print(mu)
    x = []
    for _ in range(n):
        z = multinomial(1,theta)
        t = [i for i,item in enumerate(z) if item == 1].pop()
        x.append(multivariate_normal(beta[t][0],beta[t][1]))
    return np.array(x)

def initialize(c=3, n=1000):
    z = np.zeros(1000)
    beta = np.array([[0,1] for _ in range(c)])
    return (beta, z)

def hamiltonian(x,c):
    return None 

def gibbs(x, c, alpha, lambda0, sigma = 1, model_sigma = False):
    beta, z = initialize()
    assert len(z) == len(x)

    def check_converenge():
        return True

    count = 0
    while count<10:
        count += 1

        #sample z
        for i in range(len(x)):
            temp = multinomial(1, theta * stats.norm.cdf(x[i], beta[i][0], beta[i][1]))
            z[i] = [i for i,item in enumerate(temp) if item == 1].pop()

        #sample theta:
        proportion = np.array([alpha] * c)
        for i in range(len(z)):
            proportion[z[i]] += 1
        theta = dirichlet(proportion)

        #sample beta:
        for k in range(c):
            x_k = x[z==k]
            beta[k][0] = np.mean([x_k]) * (len(x_k)/sigma^2)/(len(x_k)/sigma^2 + 1/lambda0^2)
            if model_sigma:
                beta[k][1] = np.array([[1/(len(x_k)/sigma^2 + 1/lambda0^2), 0],[0,1/(len(x_k)/sigma^2 + 1/lambda0^2)]])

    return (beta, theta, z)



if __name__ == "__main__":
    x = generative(n = 100, c= 3, sigma0)
    beta, theta,z = gibbs(x,c, )
    
    print(x)
    plt.scatter(x[:,0],x[:,1])
    plt.show()