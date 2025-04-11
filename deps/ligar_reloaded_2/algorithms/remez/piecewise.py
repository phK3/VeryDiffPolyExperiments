# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:49:24 2022

@author: Edoardo
"""

import numpy as np

"""
Return the p-norm of a numpy polynomial (Chebyshev is allowed)

Arguments:
    poly -> numpy polynomial (Chebyshev is allowed)
    p -> order of the norm (e.g. p=0, p=1, ..., p=np.inf)
"""
def polynorm(poly, p):
    
    # extract coefficient of the monomial basis
    a = poly.convert(kind=np.polynomial.Polynomial).coef
    
    return np.linalg.norm(a, p)

"""
Piecewise polynomial class
"""
class PiecewisePoly():
    
    """
    Construct a piecewise polynomial
        d_0 < p_0(x) <= d_1 < p_1(x) <= d_2 < ... <= d_n
    where d_i are discontinuity points and p_i(x) the polynomials
    
    Arguments:
        polylist -> a list of N numpy polynomials (Chebyshev is allowed)
        discolist -> list of N+1 discontinuity points (they will be sorted)
    """
    def __init__(self, polylist, discolist):
        
        if len(polylist) != len(discolist) - 1:
            raise ValueError("PiecewisePoly(): the lists of discontinuities must be exactly one element longer than the list of polynomials")
        
        self.polylist = polylist
        self.discolist = np.sort(discolist)
    
    """
    Wrapper for numpy's piecewise evaluation method
    
    Arguments:
        x -> function input (use numpy arrays for parallel evaluation)
    """
    def __call__(self, x):
        
        # build a matrix where element [i,j] is true
        # if polylist[i] is active for input x[j]
        lower = (x[np.newaxis,:] <= self.discolist[1:,np.newaxis])
        upper = (x[np.newaxis,:] > self.discolist[:-1,np.newaxis])
        condlist = np.logical_and(lower, upper)
        
        return np.piecewise(x, condlist, self.polylist)
    
    """
    Return a copy of the current piecewise polynomial
    that only include pieces that overlap with the given domain
    
    Arguments:
        domain -> iterable representing the interval domain [a,b] of interest
    """
    def restrict(self, domain):
        
        a, b = domain
        if b < a:
            raise ValueError("PiecewisePoly.restrict(): empty domain")
        
        # identify and return the piece-wise intervals that overlap with [a,b]
        lid = np.searchsorted(self.discolist, a, "right") - 1
        uid = np.searchsorted(self.discolist, b)
        
        # deal with degenerate intervals at a discontinuity (a == b == d_i)
        if lid == uid:
            lid = lid - 1
        
        # saturate the index values
        lid = np.maximum(lid, 0)
        uid = np.minimum(uid, len(self.discolist) - 1)
        
        # return a constant zero for undefined functions
        if lid >= uid:
            discolist = [a,b]
            polylist = [np.polynomial.Polynomial([0])]
            return PiecewisePoly(polylist, discolist)
        
        return PiecewisePoly(self.polylist[lid:uid], self.discolist[lid:uid+1])
    
    """
    Return the first derivative of the piecewise polynomial
    which is a piecewise polynomial itself
    """
    def gradient(self):
        
        derlist = [poly.deriv() for poly in self.polylist]
            
        return PiecewisePoly(derlist, self.discolist)
    
    """
    Compute the function norm of the piecewise polynomial
    
    Arguments:
        p -> order of the norm (e.g. p=0, p=1, ..., p=np.inf)
    """
    def norm(self, p):
        
        # return the maximum norm among all piece-wise polynomials
        norms = [polynorm(poly, p)
                 for poly in self.polylist]
        
        return np.max(norms)
    
    """
    Compute the location at which the piecewise polynomial crosses zero
    Constant pieces that output zero are ignored
    """
    def roots(self):
        
        roots = []
        for i in range(len(self.polylist)):
            
            # keep only the roots that belong to each piecewise domain
            r_all = self.polylist[i].roots()
            r_real = r_all[np.isreal(r_all)]
            in_domain = np.logical_and(r_real > self.discolist[i],
                                       r_real <= self.discolist[i+1])
            roots.append(r_real[in_domain])
        
        # flatten the result into a single numpy array
        return np.hstack(roots)

"""
Rectified Linear Unit (ReLU) activation function
"""
class ReLU(PiecewisePoly):
    
    """
    Constructor
    """
    def __init__(self):
        
        # prepare the polynomials and discontinuities
        # ReLU(x <= 0) = 0
        # ReLU(x > 0) = x
        polyneg = np.polynomial.Chebyshev([0])
        polypos = np.polynomial.Chebyshev([0, 1])
        polylist = [polyneg, polypos]
        discolist = [-np.inf, 0, +np.inf]
        
        super().__init__(polylist, discolist)
