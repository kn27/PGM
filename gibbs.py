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

def generate(model = 'gaussian_mixture', **kwargs):
    if model == 'gaussian_mixture':
        n = kwargs.pop('n', 1000)
        c = kwargs.pop('c', 3)
        #alpha = kwargs.pop('alpha')
        sigma = kwargs.pop('sigma',1)
        lambda_ = kwargs.pop('lambda_',100)
        
        #if alpha is None:
        #    alpha = [1/c] * c  
        #theta = dirichlet(alpha)
        theta = [1/c] * c
        mu = [multivariate_normal(mean=[0,0],cov=[[lambda_,0],[0,lambda_]]) for _ in range(c)]
        cov = [[[sigma,0], [0, sigma]] for _ in range(c)] 
        x = []
        for _ in range(n):
            z = multinomial(1,theta)
            t = [i for i,item in enumerate(z) if item == 1].pop()
            x.append(multivariate_normal(mu[t],cov[t]))
        return np.array(x)

def plot(x,c,z,beta,model):
    markers = ['o', 'v', '^']
    colors = ['r', 'b', 'y']
    for cluster in range(c):
        xfilter = [i for i, temp in enumerate(z) if temp == cluster]
        plt.scatter(x[xfilter,0], x[xfilter,1], marker = markers[cluster], color = colors[cluster])
        if model == 'gaussian_mixture':
            plt.plot(*beta[cluster]['mu'], marker = markers[cluster], color = 'black') 
    plt.show()

def joint_pdf(x, c ,z, theta, beta, model, **kwargs):
    if model == 'gaussian_mixture':
        
        return np.sum(stats.multivariate_normal.logpdf(x[i], beta[z[i]]['mu'], 1) for i in range(len(x))) \
            + np.sum(stats.multivariate_normal.logpdf(beta[j]['mu'], [0,0], 1) for j in range(c))  \
            + np.sum(stats.multinomial.pmf([list(z).count(i) for i in range(c)],len(x),theta))
    elif model == 'bernoulli-beta':
        return np.sum(stats.bernoulli.logpmf(x[i], beta[z[i]]['p']) for i in range(len(x))) \
            + np.sum(stats.beta.logpdf(beta[j]['p'], a = kwargs['beta.a'], b = kwargs['beta.b']) for j in range(c))  \
            + np.sum(stats.multinomial.pmf([list(z).count(i) for i in range(c)],len(x),theta))
    elif model == 'poisson-gamma':
        return np.sum(stats.poisson.logpmf(x[i], beta[z[i]]['mu']) for i in range(len(x))) \
            + np.sum(stats.gamma.logpdf(beta[j]['mu'], a = kwargs['gamma.a']) for j in range(c))  \
            + np.sum(stats.multinomial.pmf([list(z).count(i) for i in range(c)],len(x),theta))
    else:
        raise ValueError(f'Model {model} is not yet covered. Maybe you can code it up')

def likelihood(model,x,beta):
    if model == 'gaussian_mixture':
        return stats.multivariate_normal.pdf(x,beta['mu'])
    elif model == 'bernoulli-beta':
        return stats.bernoulli.pmf(x, beta['p'])
    elif model == 'poisson-gamma':
        return stats.poisson.pmf(x, beta['mu'])

def conjugate_conditional(model,x,**kwargs):
    if model == 'gaussian_mixture':
        if len(x) == 0:
            return {'mu':multivariate_normal(mean=[0,0],cov=[[100,0],[0,100]])}
        else:
            sigma = kwargs['sigma']
            lambda_ = kwargs['lambda_']
            return {'mu':multivariate_normal(np.mean(x,0) * (len(x)/sigma**2)/(len(x)/sigma**2 + 1/lambda_**2), \
                    np.array([[1/(len(x)/sigma**2 + 1/lambda_**2), 0],[0,1/(len(x)/sigma**2 + 1/lambda_**2)]]))}
    elif model == 'gaussian_mixture':
            if len(x) == 0:
                return {'mu':multivariate_normal(mean=[0,0],cov=[[100,0],[0,100]])}
            else:
                sigma = kwargs['sigma']
                lambda_ = kwargs['lambda_']
                return {'mu':multivariate_normal(np.mean(x,0) * (len(x)/sigma**2)/(len(x)/sigma**2 + 1/lambda_**2), \
                        np.array([[1/(len(x)/sigma**2 + 1/lambda_**2), 0],[0,1/(len(x)/sigma**2 + 1/lambda_**2)]]))}
    elif model == 'gaussian_mixture':
            if len(x) == 0:
                return {'mu':multivariate_normal(mean=[0,0],cov=[[100,0],[0,100]])}
            else:
                sigma = kwargs['sigma']
                lambda_ = kwargs['lambda_']
                return {'mu':multivariate_normal(np.mean(x,0) * (len(x)/sigma**2)/(len(x)/sigma**2 + 1/lambda_**2), \
                        np.array([[1/(len(x)/sigma**2 + 1/lambda_**2), 0],[0,1/(len(x)/sigma**2 + 1/lambda_**2)]]))}

    
def gibbs(x,model,**kwargs):
    max_iter = kwargs.pop('max_iter', 50)
    if model == 'gaussian_mixture':
        c = kwargs.pop('c', 3)
        alpha = kwargs.pop('alpha')
        z = kwargs.pop('z')
        assert len(z) == len(x)
        
        beta = kwargs.pop('beta')
        assert len(beta) == c
        
        n_iter = 0 
        sample  = {'theta':[], 'z':[], 'beta':[]}
        plot(x,c,z,beta,model)
        l = 0
        theta = np.array([1/c] * c)
        while True:
            l0 = l
            l = joint_pdf(x, c, z, theta, beta, model, **kwargs)
            print(f'Iter {n_iter}: Log likelihood = {l}')
            if any([n_iter >= max_iter, abs(l/(l0+0.0000001)-1)<10e-5]):
                break
            
            #Sample z
            for i in range(len(x)):
                proportion = theta * np.array([likelihood(model, x[i], beta[j]) for j in range(c)])
                temp = multinomial(1, proportion/np.sum(proportion))
                z[i] = [i for i,item in enumerate(temp) if item == 1].pop()
            sample['z'].append(z)
            
            #Sample theta
            proportion = alpha
            for i in range(len(z)):
                proportion[z[i]] += 1
            theta = dirichlet(proportion)   #theta = np.array([1/3,1/3, 1/3])
            sample['theta'].append(theta)

            #Sample beta:
            for k in range(c):
                beta[k] = conjugate_conditional(model,x[z==k,:],**kwargs)
            sample['beta'].append(beta)
            n_iter += 1 
        plot(x,c,z,beta,model)
        return sample


def test_gaussian_mixture():
    c = 3
    sigma = 1
    lambda_ = 10
    alpha = np.array([1/3] * 3)
    x = generate('gaussian_mixture', sigma = sigma, lambda_ = lambda_)
    sample = gibbs(x = x, 
                   model = 'gaussian_mixture', 
                   c = c, 
                   alpha = alpha, 
                   beta = [{'mu':multivariate_normal(mean=[0,0],cov=[[lambda_,0],[0,lambda_]])} for _ in range(c)], 
                   z = np.array([random.randint(0,c-1) for _ in range(len(x))]), 
                   sigma = sigma, 
                   lambda_ = lambda_)

def senators():
    votes = '/home/temp/votes.csv'
    with open(votes, 'r') as file:
        x = [line.split(',') for line in file]

if __name__ == "__main__":
    np.random.seed(0)
    base()