import numpy as np
import matplotlib.pyplot as plt
import time
import random
import statistics
from scipy.stats import norm
import scipy.stats as stats
import pylab 


def generateU():
    return random.uniform(0.0,1.0)

def generateV():
    return random.uniform(-1.0,1.0)

class MySimpleAlgorithm:

    name = 'Simple'

    def generate(self,mu,sigma,n):
        
        Y = np.zeros((n),dtype=float)
        for elem in range(n):
            Y[elem] = (np.sum([generateU() for i in range(12)]) - 6.)*sigma + mu
        
        return Y

class BoxMullerAlgorithm:

    name = 'Box–Mul'

    def generate(self,mu,sigma,n):
        
        Y = np.zeros((n),dtype=float)
        for elem in range(0,n,2):
            Phi = 2. * np.pi * generateU()
            R = np.sqrt(-2.* np.log(generateU()))
            Y[elem]   = R * np.cos(Phi)
            Y[elem+1] = R * np.sin(Phi)

        return Y


class MarsagliaPolarAlgorithm:

    name = 'Ma pol'

    def generate(self,mu,sigma,n):
        
        Y = np.zeros((n),dtype=float)
        for elem in range(0,n,2):

            while True:
                U1 = 2.0 * generateU() - 1.0
                U2 = 2.0 * generateU() - 1.0
                W = U1*U1 + U2*U2
                if (W < 1.):
                    break

            C = np.sqrt(-2.0* W**(-1.0) * np.log(W))
            Y[elem]   = C * U1
            Y[elem+1] = C * U2

        return Y


class MarsagliaBrayAlgorithm:

    name = 'Ma-Bray'

    def generate(self,mu,sigma,n):
        
        # According to: https://www.doc.ic.ac.uk/~wl/papers/07/csur07dt.pdf

        p1 = .8638
        p2 = .1107
        p3 = .0228002039
        p4 = 1. - (p1 + p2 + p3)

        a = 17.49731196
        b = 4.73570326
        c = 2.15787544
        d = 2.36785163 

        Y = np.zeros((n),dtype=float)

        for elem in range(n):
            s = generateU()

            if s<p1:
                Y[elem]   = 2.0*(generateU() + generateU() + generateU() - 1.5)
            elif s<=(p1+p2):
                Y[elem]   = 1.5*(generateU() + generateU() - 1.0)
            elif s<=(p1+p2+p3):
                while True:
                    x = 6.0*generateU() - 3.0
                    x_abs = np.abs(x)
                    y = 0.358*generateU()

                    if x_abs<=1.0:
                        g_x = a*np.exp(-0.5*x*x)-b*(3.0-x*x)-c*(1.5-x_abs)
                    if (1.0 < x_abs <= 1.5):
                        g_x = a*np.exp(-0.5*x*x)-d*(3.0-x_abs)**2.0-c*(1.5-x_abs)
                    if (1.5 < x_abs <= 3.0):
                        g_x = a*np.exp(-0.5*x*x)-d*(3.0-x_abs)**2.0
                    else:
                        g_x = 0.0
                    if (y < g_x):
                        Y[elem] = x
                        break
            else:
                rr = 3.0
                while True:
                    while True:
                        aa = generateV()
                        bb = generateV()
                        dd = aa*aa + bb*bb

                        if (0.0 < dd < 1.0):
                            break
                    
                    tt = np.sqrt((rr*rr-2.0*np.log(dd))/dd)
                    xx = tt*aa
                    yy = tt*bb

                    if (np.abs(xx)>rr):
                        Y[elem] = xx
                        break
                    elif (np.abs(yy)>rr):
                        Y[elem] = yy
                        break
                
        return Y


class InversionMethod:

    name = 'Invers'

    # norm.ppf gives the inverse of gauss CDF, with mu=0 and sigma=1.

    def generate(self,mu,sigma,n):
        
        Y = np.zeros((n),dtype=float)
        for elem in range(0,n):
            U1 = generateU()
            Y[elem] = norm.ppf(U1)

        return Y

class NumpyRandom:
    """numpy normal distribution"""
    
    name = 'np.rndm'

    def generate(self,mu,sigma,n):
        return np.random.normal(mu,sigma,n)


class RandomNormalvariate:
    """numpy normal distribution"""
    
    name = 'nrmvar'

    def generate(self,mu,sigma,n):
        return [random.normalvariate(mu, sigma) for i in range(n)]

class RandomGauss:
    """numpy normal distribution"""
    
    name = 'gauss'

    def generate(self,mu,sigma,n):
        return [random.gauss(mu, sigma) for i in range(n)]


class NormalDistribution:
    """A simple example generating population with normal distribution"""

    def __init__(self,n,mu=0,sigma=1.0,strategy=NumpyRandom()):

        self.n        = n
        self.mu       = mu
        self.sigma    = sigma

        # Generate normal distribution.
        start = time.time()
        self.population = strategy.generate(self.mu, self.sigma, self.n)
        end = time.time()

        self.name = strategy.name
        self.time = end-start
        self.mean = np.mean(self.population)
        self.stdv = np.std(self.population)

        # print("Strategy         : ",strategy.name)
        # print("Time elapsed     : ",end-start)
        # print('Population mean  : ',np.mean(self.population))
        # print('Population st.dev: ',np.std(self.population))

    def plot_histogram(self,title,bins):

        plt.hist(self.population, density=True, bins=bins)
        plt.title(title)
        plt.ylabel('Probability')
        plt.xlabel('Data')

        plt.show()


if __name__ == '__main__':


    # Set up parameters.
    n_samples = 10000
    n_repetitions = 100

    strategies = [ MySimpleAlgorithm(),
                   BoxMullerAlgorithm(),
                   MarsagliaPolarAlgorithm(),
                   MarsagliaBrayAlgorithm(),
                   InversionMethod(),
                   NumpyRandom(),
                   RandomNormalvariate(),
                   RandomGauss() 
                   ]

    task = 'repeatitions'
    #task = 'histogram'
    #task = 'qq-plot'

    if task == 'repeatitions':
            
        names = []
        for i_strategy in range(len(strategies)):
            names.append(strategies[i_strategy].name[:8])


        means = np.zeros((len(strategies),n_repetitions),dtype=float)
        stdvs = np.zeros((len(strategies),n_repetitions),dtype=float)
        times = np.zeros((len(strategies),n_repetitions),dtype=float)

        for i_repeat in range(n_repetitions):
            print('repetition: ', i_repeat)

            for i_strategy in range(len(strategies)):

                normal_dist = NormalDistribution(n=n_samples,strategy=strategies[i_strategy])

                means[i_strategy,i_repeat] = normal_dist.mean
                stdvs[i_strategy,i_repeat] = normal_dist.stdv
                times[i_strategy,i_repeat] = normal_dist.time


        # Plot figure.
        plt.figure()

        # Mean of means.
        plt.subplot(221)
        plt.bar(names,np.mean(means,axis=1))
        plt.title('Mean of means')

        # Standard deviation of means.
        plt.subplot(222)
        plt.bar(names,np.std(means,axis=1))
        plt.title('Standard deviation of means')

        # linear
        plt.subplot(223)
        plt.bar(names,np.mean(stdvs-1,axis=1))
        plt.title('Mean of standard deviations minus 1')

        # symmetric log
        plt.subplot(224)
        plt.bar(names,np.mean(times,axis=1))
        plt.yscale('symlog')
        plt.title('Mean of time (seconds)')

        plt.show()

        print(names)
        print(np.mean(times,axis=1))
        print(np.mean(stdvs-1,axis=1))


    if task == 'histogram':

        # Plot histogram.

        x = np.linspace(-4, 4, 100)

        for i_strategy in range(len(strategies)):

            plt.plot(x, stats.norm.pdf(x, 0.0, 1.0),c='k')
            normal_dist = NormalDistribution(n=n_samples,strategy=strategies[i_strategy])
            normal_dist.plot_histogram(strategies[i_strategy].name,bins=100)
            
            plt.show()

        

    if task == 'qq-plot':

        # Plot q-q.

        for i_strategy in range(len(strategies)):

            normal_dist = NormalDistribution(n=n_samples,strategy=strategies[i_strategy]) 
            stats.probplot(normal_dist.population, dist="norm", fit=False, plot=plt)
            plt.title(strategies[i_strategy].name)
            plt.show()
