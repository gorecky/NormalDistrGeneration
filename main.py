import numpy as np
import matplotlib.pyplot as plt
import time
import random
import statistics
from scipy.stats import norm


def generateU():
    return random.uniform(0.0,1.0)

def generateV():
    return random.uniform(-1.0,1.0)

class MySimpleAlgorithm:

    name = '4.5.1 Simple Algorithm'

    def generate(self,mu,sigma,n):
        
        Y = np.zeros((n),dtype=float)
        for elem in range(n):
            X = [generateU() for i in range(12)]
            Y[elem] = (np.sum(X) - 6.)*sigma + mu
        
        return Y

class BoxMullerAlgorithm:

    name = '4.5.2 Box–Muller algorithm'

    def generate(self,mu,sigma,n):
        
        Y = np.zeros((n),dtype=float)
        for elem in range(0,n,2):
            U1 = generateU()
            U2 = generateU()
            Phi = 2. * np.pi * U1
            R = np.sqrt(-2.* np.log(U2))
            Y[elem]   = R * np.cos(Phi)
            Y[elem+1] = R * np.sin(Phi)

        return Y



class MarsagliaPolarAlgorithm:

    name = '4.5.3 Marsaglia’s polar algorithm'

    def generate(self,mu,sigma,n):
        
        Y = np.zeros((n),dtype=float)
        for elem in range(0,n,2):

            while True:
                U1 = 2.0 * generateU() - 1.0
                U2 = 2.0 * generateU() - 1.0
                W = U1**2.0 + U2**2.0
                if (W < 1.):
                    break

            C = np.sqrt(-2.0* W**(-1.0) * np.log(W))
            Y[elem]   = C * U1
            Y[elem+1] = C * U2

        return Y


class MarsagliaBrayAlgorithm:

    name = '4.5.4 Marsaglia–Bray algorithm'

    def generate(self,mu,sigma,n):
        
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



    name = '4.x.x Inversion Method'

    def generate(self,mu,sigma,n):
        
        Y = np.zeros((n),dtype=float)
        for elem in range(0,n):
            U1 = generateU()
            Y[elem] = norm.ppf(U1)

        return Y

class NumpyRandom:
    """numpy normal distribution"""
    
    name = 'numpy.random.normal'

    def generate(self,mu,sigma,n):
        return np.random.normal(mu,sigma,n)


class RandomNormalvariate:
    """numpy normal distribution"""
    
    name = 'random.normalvariate'

    def generate(self,mu,sigma,n):
        return [random.normalvariate(mu, sigma) for i in range(n)]

class RandomGauss:
    """numpy normal distribution"""
    
    name = 'random.gauss'

    def generate(self,mu,sigma,n):
        return [random.gauss(mu, sigma) for i in range(n)]


class NormalDistribution:
    """A simple example generating population with normal distribution"""

    def __init__(self,mu,sigma,n,strategy=NumpyRandom()):

        self.n        = n
        self.mu       = mu
        self.sigma    = sigma

        # Generate normal distribution.
        start = time.time()
        self.population = strategy.generate(self.mu, self.sigma, self.n)
        end = time.time()

        print("Strategy         : ",strategy.name)
        print("Time elapsed     : ",end-start)
        print('Population mean  : ',np.mean(self.population))
        print('Population st.dev: ',np.std(self.population))

    def plot_histogram(self):

        plt.hist(self.population, density=True, bins=100)
        plt.ylabel('Probability')
        plt.xlabel('Data')

        plt.show()


if __name__ == '__main__':

    normal_dist = NormalDistribution(0,1,100000,strategy=MarsagliaBrayAlgorithm())
    #normal_dist = NormalDistribution(0,1,10000,strategy=MySimpleAlgorithm())

    normal_dist.plot_histogram()

        
        