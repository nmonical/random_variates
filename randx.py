import numpy as np
import math
from scipy.stats import norm
from scipy import stats

""" 
desert_island generates Unif(0,1) random numbers which are then used in the generation of random variates from other distributions. 
    n = number of uniform random numbers required for the generation of one random number from the goal distribution
    size = number of random numbers to be be generated from the goal distribution
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M6L3)
"""
def desert_island(n,size,seed):
    X_i_minus_1=seed
    output=[]
    for i in range(size):
        sample=[]
        for j in range(n):
            X_i = 16807*X_i_minus_1%(2**31-1)
            sample.append(X_i/(2**31-1))
            X_i_minus_1=X_i
        output.append(sample)
    return output

"""
rand_unif uses Inverse Transform Theory to generate uniform random numbers between a and b. 
    a = minimum value
    b = maximum value
    size = number of random numbers to be be generated
    seed = starting seed
"""
def rand_unif(a,b,size, seed=42):
    if not isinstance(a,(float, int)):
        raise ValueError('a must be a number')
    if not isinstance(b,(float, int)):
        raise ValueError('b must be a number')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    rand_unifs=desert_island(1,size,seed)
    unif_output=[]
    for i in rand_unifs:
        unif_output.append(a+(b-a)*i[0])
    return unif_output

"""
rand_triangular uses Inverse Transform Theory to generate random numbers from a triangular distribution between a and b. 
    a = minimum value
    b = maximum value
    c = mode value
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from math.wm.edu
"""

def rand_triangular(a,c,b,size, seed=42):
    if not isinstance(a, (float, int)):
        raise TypeError('a must be numeric')
    if not isinstance(b, (float, int)):
        raise TypeError('b must be numeric')
    if not isinstance(c, (float, int)):
        raise TypeError('c must be numeric')    
    if c > b or c < a:
        raise ValueError('c must be between a and b')
    if a > b:
        raise ValueError('a must be less than b')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    rand_unifs=desert_island(1,size,seed)
    tri_output=[]
    for i in rand_unifs:
        if i[0] < (c-a)/(b-a):
            tri_output.append(a+((b-a)*(c-a)*i[0])**(1/2))
        else:
            tri_output.append(b-((b-a)*(b-c)*(1-i[0]))**(1/2))
    return tri_output

"""
rand_exp uses Inverse Transform Theory to generate random numbers from an exponential distribution with parameter lambda(l). 
    l = lambda (rate)
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L2)
"""

def rand_exp(l, size, seed=42):
    if not isinstance(l, (float, int)):
        raise TypeError('l must be numeric')
    if l < 0:
        raise ValueError('l must be greater than or equal to zero')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    rand_unifs=desert_island(1,size,seed)
    exp_output=[]
    for i in rand_unifs:
        exp_output.append((-1/l)*np.log(1-i[0]))
    return exp_output

"""
rand_weibull uses Inverse Transform Theory to generate random numbers from a Weibull distribution with parameters lambda(l), and beta(b). 
    l = lambda (rate)
    b = beta (shape)
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L2)
"""
def rand_weibull(l, k, size, seed=42):
    if not isinstance(l,(float,int)):
        raise TypeError('l must be numeric')
    if l <= 0:
        raise ValueError('l must be greater than zero')
    if not isinstance(k,(float,int)):
        raise TypeError('k must be numeric')
    if k <= 0:
        raise ValueError('k must be greater than zero')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    rand_unifs=desert_island(1,size,seed)
    weibull_output=[]
    for i in rand_unifs:
        weibull_output.append(l*(-math.log(1-i[0]))**(1/k))
    return weibull_output

"""
rand_erlang uses Inverse Transform Theory and Convolution to generate random numbers from an Erlang distribution with parameters lambda(l), and number(n). 
    n = number (shape)
    l = lambda (rate)
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L6)
"""
def rand_erlang(n, l, size, seed=42):
    if not isinstance(n,int):
        raise TypeError('n must be an integer')
    if n <= 0:
        raise ValueError('n must be greater than zero')
    if not isinstance(l,(float,int)):
        raise TypeError('l must be numeric')
    if l <= 0:
        raise ValueError('l must be greater than zero')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')   
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    rand_unifs=desert_island(n,size,seed)
    erlang_output=[]
    for i in rand_unifs:
        prod=1
        for j in i:
            prod*=j
        erlang_output.append((-1/l)*math.log(prod))
    return erlang_output
"""
rand_gamma uses Inverse Transform Theory and Acceptance-Rejection to generate random numbers from a gamma distribution with parameters alpha(a) and beta(b). 
    a = alpha (shape)
    b = beta (rate)
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Simulation and Modeling Analysis, Law, Averill M., page 454-456
"""
def rand_gamma(a, b, size, seed=42):
    if not isinstance(a,(float,int)):
        raise TypeError('a must be numeric')
    if a <= 0:
        raise ValueError('a must be greater than zero')
    if not isinstance(b,(float,int)):
        raise TypeError('b must be numeric')
    if b <= 0:
        raise ValueError('b must be greater than zero')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')   
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    rand_unifs=desert_island(2,size,seed)
    gamma_output=[]
    c=(math.e+a)/math.e
    if a==1:
        for i in rand_unifs:
            gamma_output.append(b*(-1/1)*np.log(1-i[0]))
    if a < 1:
        for i in rand_unifs:
            if c*i[0] > 1:
                Y=-(math.log((c-c*i[0])/a))
                if i[1] <= Y**(a-1):
                    gamma_output.append(Y)
            else:
                Y=(c*i[0])**(1/a)
                if i[1] <= math.exp(-Y):
                    gamma_output.append(b*Y)
    else:
        v=1/(2*a-1)**(1/2)
        w=a-math.log(4)
        q=a+1/v
        x=(a+1)/1
        y=4.5
        z=1+math.log(y)
        for i in rand_unifs:
            V=v*math.log(i[0]/(1-i[0]))
            Y=a*math.exp(V)
            Z=i[0]**2*i[1]
            W=w+q*V-Y
            if W+z-y*Z>=0:
                gamma_output.append(b*Y)
            elif W>=math.log(Z):
                gamma_output.append(b*Y)
        return gamma_output
    
"""
rand_normal uses Inverse Transform Theory to generate random numbers from a normal distribution with parameters mean(m), and variance(v). 
    m = mean 
    v = variance
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L3)
"""
def rand_normal(m, v, size, seed=42):
    if not isinstance(m,(float,int)):
        raise TypeError('m must be numeric')
    if not isinstance(v,(float,int)):
        raise TypeError('v must be numeric')
    if v < 0:
        raise ValueError('v must be greater than or equal to zero')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    rand_unifs=desert_island(1,size,seed)
    norm_output=[]
    for i in rand_unifs:
        norm_output.append(norm.ppf(i[0], m, v))
    return norm_output

"""
rand_bern uses Inverse Transform Theory to generate random numbers from a Bernoulli distribution with parameter p. 
    p = probability of success 
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L4)
"""
def rand_bern(p, size, seed=42):
    if not isinstance(p,(float, int)):
        raise TypeError('p must be numeric')
    if p < 0 or p > 1:
        raise ValueError('p must be between 0 and 1')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    rand_unifs=desert_island(1,size,seed)
    bern_output=[]
    for i in rand_unifs:
        if i[0] <= p:
            bern_output.append(1)
        else:
            bern_output.append(0)
    return bern_output
    
"""
rand_bin uses Inverse Transform Theory and Convolution to generate random numbers from a binomial distribution with parameters n and p. 
    n = number of trials
    p = probability of success 
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L6)
"""
def rand_bin(n, p, size, seed=42):
    if not isinstance(n,int):
        raise TypeError('n must be an integer')
    if n <= 0:
        raise ValueError('n must be greater than zero')
    if not isinstance(p,(float, int)):
        raise TypeError('p must be numeric')
    if p < 0 or p > 1:
        raise ValueError('p must be between 0 and 1')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    rand_unifs=desert_island(n,size,seed)
    bin_output=[]
    for i in rand_unifs:
        total=0
        for j in i:
            if j <= p:
                total+=1
        bin_output.append(total)
    return bin_output

"""
rand_geom uses Inverse Transform Theory to generate random numbers from a geometric distribution with parameters p. 
    p = probability of success 
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L4)
"""
def rand_geom(p, size, seed=42):
    if not isinstance(p,(float, int)):
        raise TypeError('p must be numeric')
    if p < 0 or p > 1:
        raise ValueError('p must be between 0 and 1')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    rand_unifs=desert_island(1,size,seed)
    geom_output=[]
    for i in rand_unifs:
        geom_output.append(math.ceil(math.log(i[0])/math.log(1-p)))
    return geom_output

"""
rand_negbin uses Inverse Transform Theory and Convolution to generate random numbers from a negative binomial distribution with parameters n and p. 
    n = number of successes
    p = probability of success 
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L6)
"""
def rand_negbin(n, p, size, seed=42):
    if not isinstance(n,int):
        raise TypeError('n must be an integer')
    if n <= 0:
        raise ValueError('n must be greater than zero')
    if not isinstance(p,(float, int)):
        raise TypeError('p must be numeric')
    if p < 0 or p > 1:
        raise ValueError('p must be between 0 and 1')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    rand_unifs=desert_island(n,size,seed)
    negbin_output=[]
    for i in rand_unifs:
        count=0
        for j in i:
            count+=math.ceil(math.log(j)/math.log(1-p))
        negbin_output.append(count)
    return negbin_output

"""
rand_bin uses Inverse Transform Theory and Convolution to generate random numbers from a poisson distribution with parameter lambda(l). 
    l = lambda (rate)
    size = number of random numbers to be be generated
    seed = starting seed
Algorithm from Goldsman Lecture Notes (M7L10)
"""
def rand_poisson(l, size, seed=42):
    if not isinstance(l,(float,int)):
        raise TypeError('l must be numeric')
    if l <= 0:
        raise ValueError('l must be greater than zero')
    if not isinstance(size,int):
        raise TypeError('size must be an integer')
    if size <= 0:
        raise ValueError('size must be greater than zero')
    if not isinstance(seed,int):
        raise TypeError('seed must be an integer')
    if seed <= 0:
        raise ValueError('seed must greater than zero')
    rand_unifs=desert_island(l+10,size,seed)
    poisson_output=[]
    for i in rand_unifs:
        p=1
        X=-1
        a=math.exp(-l)
        for j in i:
            if p >= a:
                p=p*j
                X=X+1
            else:
                break
        poisson_output.append(X)
    return poisson_output




