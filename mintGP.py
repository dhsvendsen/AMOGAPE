import autograd
import autograd.numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky, solve
from scipy.optimize import minimize

# Developer notes
# 1) Cholesky decomposition produces NaNs (probably because K+I*s**2 is not pos semidef) causing solve to complain
# 2) Including gradient for likelihood made optimization much faster

class mintGP():
    """
    The implementation is based on Algorithm 2.1 of Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams.
    
    Takes 2D np-arrays
    """
    
    def __init__(self):
        self.jitter = 1e-9
        
    def fit(self, X, Y):
        self.Ystd = Y.std(0); self.Ymean = Y.mean(0);
        self.Xstd = X.std(0); self.Xmean = X.mean(0);
        self.Y = (Y-self.Ymean)/self.Ystd
        self.X = (X-self.Xmean)/self.Xstd
        self.n = np.shape(X)[0]
        # initialize with heuristics
        self.lengthscale = 1. #np.median(pdist(X, metric='euclidean'))
        self.likelihood_variance =  .1
        
        ###############################################################
        # Gradient descent on marginal likelihood with scipy L-BFGS-B #
        ###############################################################
        theta0 = np.array([self.lengthscale, self.likelihood_variance])
        bnds = ((1e-10, None), (1e-9, None))
        try:
            sol = minimize(self.neg_log_marg_like, theta0, args=(),
                       method='L-BFGS-B', bounds=bnds, jac=True) 
        except ValueError:
            theta0 = np.array([self.lengthscale, self.likelihood_variance + 1000*self.jitter])
            sol = minimize(self.neg_log_marg_like, theta0, args=(),
                       method='L-BFGS-B', bounds=bnds, jac=True) 
        self.lengthscale, self.likelihood_variance = sol.x
        self.final_nll = sol.fun
        self.marginal_likelihood = np.exp(-sol.fun)
        
        # for prediction:
        K,_ = self.K(self.lengthscale, self.X, self.X)
        self.L = cholesky( K + (self.likelihood_variance+self.jitter)*np.eye(self.n), lower=True)
        
    def prebuild(self, X, Y, theta):
        self.Ystd = Y.std(0); self.Ymean = Y.mean(0);
        self.Xstd = X.std(0); self.Xmean = X.mean(0);
        self.Y = (Y-self.Ymean)/self.Ystd
        self.X = (X-self.Xmean)/self.Xstd
        self.n = np.shape(X)[0]
        
        self.lengthscale, self.likelihood_variance = theta[0], theta[1]
        K,_ = self.K(self.lengthscale, self.X, self.X)
        self.L = cholesky( K + (self.likelihood_variance+self.jitter)*np.eye(self.n), lower=True)
        
    ##########################
    # Likelihood computation #
    ##########################
    
    def neg_log_marg_like(self, theta):
        """
        Compute negative log marginal likelihood for hyperparameter optimization
        """
        K, D = self.K(theta[0], self.X ,self.X)
        self.kk = K
        L = cholesky( K + (theta[1]+self.jitter)*np.eye(self.n), lower=True)
        self.L = L
        alpha = solve(L.T, solve(L,self.Y, lower=True) )
        logmarglike = \
        - 0.5*np.dot(self.Y.T, alpha)[0,0]     \
        - np.sum( np.log( np.diag( L ) ) )   \
        - 0.5*self.n*np.log(2*np.pi)
        
        # compute gradients
        prefactor = np.dot(alpha, alpha.T) - solve(L.T, solve(L, np.eye(self.n) ) )
        Kd_lengthscale = np.multiply( D/theta[0]**3, K)
        Kd_likelihood_variance = np.eye(self.n)
        logmarglike_grad = 0.5*np.array( [ np.trace( np.dot(prefactor, Kd_lengthscale) ),
                                           np.trace( np.dot(prefactor, Kd_likelihood_variance) )] )

        return -logmarglike, -logmarglike_grad
    
    def nlml_grad(self):
        """
        Return gradient of negative log marginal likelihood
        """
        return self.logmarglike_grad
    
    
    ######################
    # Kernel computation #
    ######################
    
    def K(self, lengthscale, X, Z=None):
        """ Returns the EQ kernel matrix
        """
        n1 = np.shape(X)[0]
        n1sq = np.sum(np.square(X), 1)
        
        if Z is None:
            n2 = n1
            n2sq = n1sq
            D = np.abs( (np.ones([n2, 1])*n1sq).T + np.ones([n1, 1])*n2sq -2*np.dot(X,X.T) )
        else:
            n2 = np.shape(Z)[0]
            n2sq = np.sum(np.square(Z), 1)
            D = np.abs( (np.ones([n2, 1])*n1sq).T + np.ones([n1, 1])*n2sq -2*np.dot(X,Z.T) )
        
        return np.exp(-D/(2*lengthscale**2)), D
    
    def scalarK(self, x, z, lengthscale):
        return( np.exp( np.linalg.norm(x - z)**2/(2*lengthscale**2) ) )
    
    ###########################
    # Predictive distribution #
    ###########################
    
    def predict(self, Xnew, predvar=False):
        """ Returns the predictive mean and variance of the GP
        """
        Xnew = (Xnew - self.Xmean)/self.Xstd
        alpha = solve(self.L.T, solve(self.L,self.Y*self.Ystd+self.Ymean) )
        if predvar:
            m = np.shape(Xnew)[0]
            Knew_N,_ = self.K(self.lengthscale, Xnew, self.X)
            Knew_new = np.array( [self.scalarK(Xnew[i], Xnew[i], self.lengthscale) for i in range(m)] ).reshape([m,1])
            v = solve(self.L, Knew_N.T)
            return np.dot(Knew_N, alpha), np.diag( Knew_new + self.likelihood_variance - np.dot(v.T, v) ).reshape(m,1)
        else:
            Knew_N,_ = self.K(self.lengthscale, Xnew, self.X)
            return np.dot(Knew_N, alpha)
    
   
    ###############################################################################   
    # Functionality for AMOGAPE: Gradient of predictive mean, predictive variance #
    ###############################################################################
    
    def mu_grad(self, Xnew):
        """ Returns gradient of the predictive mean
        """
        mu = lambda x: self.predict(x)
        auto_grad = autograd.grad(mu)
        return auto_grad(Xnew)
    
    def geometric_info(self, Xnew):
        """ Returns the geometric info defined as the norm of the
        gradient of the predictive mean || \nabla \mu_{GP}(x) ||
        """
        return np.linalg.norm( self.mu_grad(Xnew) )
    
    def geometric_info_gradient(self, Xnew):
        """ Returns the gradient of the geometric info
        """
        geometry = lambda x: self.predict(x)
        auto_geo_grad = autograd.grad( geometry )
        return auto_geo_grad(Xnew)
    
    def mu_grad_old(self, Xnew):
        """ Old mu_grad using actual math.. who needs that when you have autograd
        """
        alpha = solve(self.L.T, solve(self.L, self.Y*self.Ystd+self.Ymean ) )
        Knew_N,_ = self.K(self.lengthscale, (Xnew-self.Xmean)/self.Xstd, self.X)
        normalX = (self.X * self.Xstd) + self.Xmean
        return np.diag( (-1/(self.lengthscale*self.Xstd)**2)*np.dot( np.tile(Xnew.T, self.n) - normalX.T, np.multiply(Knew_N.T, alpha) ) )
    
    def predvar_fast(self,Xnew):
        """ Compute predictive variance using already computed cholesky decomposition
        """
        Xnew = (Xnew - self.Xmean)/self.Xstd
        m = np.shape(Xnew)[0]
        Knew_N,_ = self.K(self.lengthscale, Xnew, self.X)
        Knew_new = np.array( [self.scalarK(Xnew[i], Xnew[i], self.lengthscale) for i in range(m)] ).reshape([m,1])
        v = np.linalg.solve(self.L, Knew_N.T)
        print(np.shape(v))
        return np.diag( Knew_new + self.likelihood_variance - np.dot(v.T, v) ).reshape(m,1)
    
    def predvar(self,Xnew):
        """ Compute predictive variance in a way that little autograd gets
        """
        Xnew = (Xnew - self.Xmean)/self.Xstd
        m = np.shape(Xnew)[0]
        Knew_N,_ = self.K(self.lengthscale, Xnew, self.X)
        Knew_new = np.array( [self.scalarK(Xnew[i], Xnew[i], self.lengthscale) for i in range(m)] ).reshape([m,1])
        K,_ = self.K(self.lengthscale, self.X, self.X) + (self.likelihood_variance+self.jitter)*np.eye(self.n)
        v = np.linalg.solve(K, Knew_N.T)
        return np.diag(- np.dot(Knew_N, v) + self.likelihood_variance + 1 ).reshape(m,1) 
 