# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:02:37 2022

@author: Edoardo
"""

import numpy as np

from algorithms.remez.piecewise import polynorm
from algorithms.remez.piecewise import PiecewisePoly
from algorithms.remez.piecewise import ReLU

def polynormTest():
    
    cpoly = [-1, 2, 4, -3]
    ccheb = [-1, 2, 4, -3]
    ccheb2poly = [ccheb[0] - ccheb[2],
                  ccheb[1] - 3 * ccheb[3],
                  2 * ccheb[2],
                  4 * ccheb[3]]
    
    poly = np.polynomial.Polynomial(cpoly)
    cheb = np.polynomial.Chebyshev(ccheb)
    
    for p in [0, 1, 2, np.inf]:
        assert polynorm(poly, p) == np.linalg.norm(cpoly, p)
        assert polynorm(cheb, p) == np.linalg.norm(ccheb2poly, p)

def PiecewisePolyCallTest():
    
    p0 = np.polynomial.Polynomial([1, 2, 3])
    p1 = np.polynomial.Chebyshev([-1, 2, -2])
    p2 = np.polynomial.Polynomial([0, 0, 1, 1])
    polylist = [p0, p1, p2]
    discolist = [-438, -2, 1, 15]
    
    piece = PiecewisePoly(polylist, discolist)
    
    for i in range(len(polylist)):
        x = np.linspace(discolist[i], discolist[i+1], 50)
        y_ref = polylist[i](x[1:])
        y_piece = piece(x[1:])
        
        assert np.allclose(y_ref, y_piece)

def PiecewisePolyRestrictTest():
    
    pnull = np.polynomial.Polynomial([0])
    p0 = np.polynomial.Polynomial([1, 2, 3])
    p1 = np.polynomial.Chebyshev([-1, 2, -2])
    p2 = np.polynomial.Polynomial([0, 0, 1, 1])
    polylist = [p0, p1, p2]
    discolist = [-438, -2, 1, 15]
    
    piece = PiecewisePoly(polylist, discolist)
    
    domains = [[-1000,-900], # []
               [-1000,-438], # []
               [-1000,-437], # [p0]
               [-1000,-2],   # [p0]
               [-438,-438],  # []
               [-10, 0],     # [p0,p1]
               [-2,-2],      # [p0]
               [1,1],        # [p1]
               [-2,1],       # [p1]
               [-10,10],     # [p0,p1,p2]
               [1,15],       # [p2]
               [15,15],      # [p2]
               [15,16]]      # []
    polyref = [[pnull],
               [pnull],
               [p0],
               [p0],
               [pnull],
               [p0,p1],
               [p0],
               [p1],
               [p1],
               [p0,p1,p2],
               [p2],
               [p2],
               [pnull]]
    discoref = [[-1000,-900],
                [-1000,-438],
                [-438,-2],
                [-438,-2],
                [-438,-438],
                [-438,-2,1],
                [-438,-2],
                [-2,1],
                [-2,1],
                [-438, -2, 1, 15],
                [1,15],
                [1,15],
                [15,16]]
    
    for i, domain in enumerate(domains):

        subpiece = piece.restrict(domain)
        
        assert subpiece.polylist == polyref[i]
        assert (subpiece.discolist == discoref[i]).all()

def PiecewisePolyGradientTest():
    
    p0 = np.polynomial.Polynomial([1, 2, 3])
    p1 = np.polynomial.Chebyshev([-1, 2, -2])
    p2 = np.polynomial.Polynomial([0, 0, 1, 1])
    polylist = [p0, p1, p2]
    discolist = [-438, -2, 1, 15]
    
    piece = PiecewisePoly(polylist, discolist)
    grad = piece.gradient()
    
    assert (grad.discolist == discolist).all()
    assert len(grad.polylist) == 3
    assert (grad.polylist[0].coef == [2, 6]).all()
    assert (grad.polylist[1].convert(kind=np.polynomial.Polynomial).coef
            == [2, -8]).all()
    assert (grad.polylist[2].coef == [0, 2, 3]).all()

def PiecewisePolyNormTest():
    
    p0 = np.polynomial.Polynomial([1, 2, 3])
    p1 = np.polynomial.Chebyshev([-1, 2, -2])
    p2 = np.polynomial.Polynomial([0, 0, 1, 1])
    polylist = [p0, p1, p2]
    discolist = [-438, -2, 1, 15]
    
    piece = PiecewisePoly(polylist, discolist)
    
    norms = [np.linalg.norm(p0.coef, 2),
             np.linalg.norm(p1.convert(kind=np.polynomial.Polynomial).coef, 2),
             np.linalg.norm(p2.coef, 2)]
    
    assert piece.norm(2) == np.max(norms)

def PiecewisePolyRootsTest():
    
    p0 = np.polynomial.Polynomial([1, 2, 3])    # has complex roots
    p1 = np.polynomial.Chebyshev([-1, 2, -2])   # has two roots (see r_ref)
    p2 = np.polynomial.Polynomial([0, 0, 1, 1]) # has out-of-domain roots [-1, 0]
    polylist = [p0, p1, p2]
    discolist = [-438, -2, 1, 15]
    
    r_ref = [(1-np.sqrt(5))/4,(1+np.sqrt(5))/4]
    
    piece = PiecewisePoly(polylist, discolist)
    roots = piece.roots()

    assert np.allclose(np.sort(roots), np.sort(r_ref))

def ReLUTest():
    
    act = ReLU()
    step = act.gradient()
    
    x = np.linspace(-56, 17, 3000)
    y = np.maximum(x, 0)
    z = (x >= 0) * 1
    
    assert (act(x) == y).all()
    assert (step(x) == z).all()
    assert np.isclose(act.norm(1), 1)

def runTests():
    
    print(">>> testing module piecewise.py")
    
    polynormTest()
    
    PiecewisePolyCallTest()
    PiecewisePolyRestrictTest()
    PiecewisePolyGradientTest()
    PiecewisePolyNormTest()
    PiecewisePolyRootsTest()
    
    ReLUTest()
    
    print(">>> done!")

if __name__ == "__main__":
    runTests()
