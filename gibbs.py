import random
import numpy.random as random
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

def generate(model = 'gaussian_mixture', **kwargs):
    n = kwargs['n']
    c = kwargs['c']
    theta = [1/c] * c
    x = []
    z = []
    if model == 'gaussian_mixture':
        #alpha = kwargs.pop('alpha')
        sigma = kwargs['sigma']
        lambda_ = kwargs['lambda_']
        #if alpha is None:
        #    alpha = [1/c] * c  
        #theta = random.dirichlet(alpha)
        beta = [{'mu':random.multivariate_normal(mean=[0,0],cov=[[lambda_,0],[0,lambda_]])} for _ in range(c)]
        for _ in range(n):
            z.append([i for i,item in enumerate(random.multinomial(1,theta)) if item == 1].pop())
            x.append(random.multivariate_normal(beta[z[-1]]['mu'],[[1,0],[0,1]]))        
    elif model == 'bernoulli_beta':
        beta_alpha = kwargs['beta_alpha']
        beta_beta = kwargs['beta_beta']
        binomial_n = kwargs['binomial_n']
        beta = [{'p': random.beta(beta_alpha, beta_beta)} for _ in range(c)]
        for _ in range(n):
            z.append([i for i,item in enumerate(random.multinomial(1,theta)) if item == 1].pop())
            x.append(random.binomial(binomial_n, beta[z[-1]]['p']))
    elif model == 'poisson_gamma':
        gamma_k = kwargs.pop('gamma_k',1)
        gamma_theta = kwargs.pop('gamma_theta', 0.2)
        beta = [{'mu':random.gamma(gamma_k, gamma_theta)} for _ in range(c)]
        for _ in range(n):
            z.append([i for i,item in enumerate(random.multinomial(1,theta)) if item == 1].pop())
            x.append(random.poisson(beta[z[-1]]['mu']))
    print(f'True beta: {beta}, true theta:{theta}')
    print(f'True joint: {joint_pdf(x, c, z, theta, beta, model, **kwargs)}')
    return np.array(x)

def plot(x,c,z,beta,model):
    if model == 'gaussian_mixture':
        markers = ['o', 'v', '^']
        colors = ['r', 'b', 'y']
        for cluster in range(c):
            xfilter = [i for i, temp in enumerate(z) if temp == cluster]
            plt.scatter(x[xfilter,0], x[xfilter,1], marker = markers[cluster], color = colors[cluster])
            plt.plot(*beta[cluster]['mu'], marker = markers[cluster], color = 'black') 
        plt.show()
    else:
        return None

def joint_pdf(x, c ,z, theta, beta, model, **kwargs):
    if model == 'gaussian_mixture':
        return sum(stats.multivariate_normal.logpdf(x[i], beta[z[i]]['mu'], 1) for i in range(len(x))) \
            + sum(stats.multivariate_normal.logpdf(beta[j]['mu'], [0,0], 1) for j in range(c))  \
            + stats.multinomial.pmf([list(z).count(i) for i in range(c)],len(x),theta)
    elif model == 'bernoulli_beta':
        return sum(stats.binom.logpmf(x[i], kwargs['binomial_n'], beta[z[i]]['p']) for i in range(len(x))) \
            + sum(stats.beta.logpdf(beta[j]['p'], a = kwargs['beta_alpha'], b = kwargs['beta_beta']) for j in range(c))  \
            + stats.multinomial.pmf([list(z).count(i) for i in range(c)],len(x),theta)
    elif model == 'poisson_gamma':
        return sum(stats.poisson.logpmf(x[i], beta[z[i]]['mu']) for i in range(len(x))) \
            + sum(stats.gamma.logpdf(beta[j]['mu'], k = kwargs['gamma_k'], scale = kwargs['gamma_theta']) for j in range(c))  \
            + stats.multinomial.pmf([list(z).count(i) for i in range(c)],len(x),theta)
    else:
        raise ValueError(f'Model {model} is not yet covered. Maybe you can code it up')

def likelihood(model,x,beta, **kwargs):
    if model == 'gaussian_mixture':
        return stats.multivariate_normal.pdf(x,beta['mu'])
    elif model == 'bernoulli_beta':
        return stats.binom.pmf(x, kwargs['binomial_n'],  beta['p'])
    elif model == 'poisson_gamma':
        return stats.poisson.pmf(x, beta['mu'])

def conjugate_conditional(model,x,**kwargs):
    if model == 'gaussian_mixture':
        sigma = kwargs['sigma']
        lambda_ = kwargs['lambda_']    
        if len(x) == 0:
            return {'mu':random.multivariate_normal(mean=[0,0],cov=[[100,0],[0,100]])}
        else:
            return {'mu':random.multivariate_normal(np.mean(x,0) * (len(x)/sigma**2)/(len(x)/sigma**2 + 1/lambda_**2), \
                    np.array([[1/(len(x)/sigma**2 + 1/lambda_**2), 0],[0,1/(len(x)/sigma**2 + 1/lambda_**2)]]))}
    elif model == 'bernoulli_beta':
        beta_alpha = kwargs['beta_alpha']
        beta_beta = kwargs['beta_beta']   
        if len(x) == 0:
            return {'p':random.beta(beta_alpha, beta_beta)}
        else:
            return {'p':random.beta(beta_alpha + np.sum(x), beta_beta + len(x)*kwargs['binomial_n']- np.sum(x))}
    elif model == 'random.poisson_random.gamma':
        random.gamma_k = kwargs['random.gamma_k']
        random.gamma_scale = kwargs['random.gamma_scale']
        if len(x) == 0:
            return {'mu':random.gamma(random.gamma_k, random.gamma_scale)}
        else:
            return {'mu':random.gamma(random.gamma_k + np.sum(x), random.gamma_scale/(len(x) * random.gamma_scale + 1))}

    
def gibbs(x,model,**kwargs):
    max_iter = kwargs.pop('max_iter', 1000)
    threshold = kwargs.pop('threshold', 10e-4)
    c = kwargs.pop('c')
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
        print(f'Iter {n_iter}: Log likelihood = {l}, beta: {beta}, Theta: {theta}')
        if any([n_iter >= max_iter, abs(l/(l0+0.000001)-1)<threshold]):
            break
        
        #Sample z
        for i in range(len(x)):
            proportion = theta * np.array([likelihood(model, x[i], beta[j], **kwargs) for j in range(c)])
            temp = random.multinomial(1, proportion/np.sum(proportion))
            z[i] = [i for i,item in enumerate(temp) if item == 1].pop()
        sample['z'].append(z)
        
        #Sample theta
        proportion = alpha
        for i in range(len(z)):
            proportion[z[i]] += 1
        theta = random.dirichlet(proportion)   #
        #theta = np.array([1/3,1/3, 1/3])
        sample['theta'].append(theta)

        #Sample random.beta:
        for k in range(c):
            beta[k] = conjugate_conditional(model,x[z==k],**kwargs)
        sample['beta'].append(beta)
        n_iter += 1 
    plot(x,c,z,beta,model)
    return sample


def test_gaussian_mixture():
    c = 3
    sigma = 1
    lambda_ = 10
    alpha = np.array([1/c] * c)
    x = generate('gaussian_mixture', sigma = sigma, lambda_ = lambda_)
    sample = gibbs(x = x, 
                   model = 'gaussian_mixture', 
                   c = c, 
                   alpha = alpha, 
                   beta = [{'mu':random.multivariate_normal(mean=[0,0],cov=[[lambda_,0],[0,lambda_]])} for _ in range(c)], 
                   z = np.array([random.randint(0,c-1) for _ in range(len(x))]), 
                   sigma = sigma, 
                   lambda_ = lambda_)

def test_bernoulli_beta():
    c = 3
    beta_alpha = 1
    beta_beta = 1
    binomial_n = 100
    alpha = np.array([1/3] * 3)
    x = generate('bernoulli_beta', beta_alpha = beta_alpha, beta_beta = beta_beta, binomial_n = binomial_n)
    sample = gibbs(x = x, 
                   model = 'bernoulli_beta', 
                   c = c, 
                   alpha = alpha, 
                   beta = [{'p':random.beta(beta_alpha, beta_beta)} for _ in range(c)], 
                   z = np.array([random.randint(0,c-1) for _ in range(len(x))]), 
                   beta_alpha = beta_alpha, 
                   beta_beta = beta_beta,
                   binomial_n = binomial_n)

def test_poisson_gamma():
    c = 3
    gamma_k = 1
    gamma_theta = 0.2
    alpha = np.array([1/3] * 3)
    x = generate('poisson_gamma', gamma_k = gamma_k, gamma_theta = gamma_theta)
    sample = gibbs(x = x, 
                   model = 'poisson_gamma', 
                   c = c, 
                   alpha = alpha, 
                   beta = [{'mu':random.gamma(gamma_k, gamma_theta)} for _ in range(c)], 
                   z = np.array([random.randint(0,c-1) for _ in range(len(x))]), 
                   gamma_k = gamma_k,
                   gamma_theta = gamma_theta)

def senators():
    votes = '/home/keane/temp/senate/votes.csv'
    with open(votes) as file:
        votes = [line.split(',') for line in file]
        votes = np.array([round(vote.count('1')/(vote.count('1') + vote.count('0')) * 1000) for vote in votes])
    c = 4
    sample = gibbs(x = votes, 
                   model = 'bernoulli_beta', 
                   c = c, 
                   alpha = np.array([1/c]*c), 
                   beta = [{'p':random.beta(1,1)} for _ in range(c)], 
                   z = np.array([random.randint(0,c) for _ in range(len(votes))]), 
                   beta_alpha = 1,
                   beta_beta = 1,
                   binomial_n = 1000)
    print()

if __name__ == "__main__":
    np.random.seed(0)
    #test_bernoulli_beta()
    senators()

#TODO: Stability of calculation -> Why is it more preferable to use log?
#TODO: 