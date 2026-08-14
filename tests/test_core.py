import numpy as np
import pytest
from odeweaver import fit, polynomial_library


def logistic(dt=0.002, steps=2500):
    r,k=1.5,2.0; x=np.empty(steps); x[0]=0.2
    for i in range(steps-1): x[i+1]=x[i]+dt*r*x[i]*(1-x[i]/k)
    return x[:,None],dt


def test_recovers_logistic_terms():
    x,dt=logistic(); m=fit(x,dt,degree=2,threshold=0.08)[0]
    c=dict(zip(m.terms,m.coefficients))
    assert abs(c['x0']-1.5)<0.08
    assert abs(c['x0*x0']+0.75)<0.08


def test_library_shape():
    lib,names=polynomial_library(np.ones((5,2)),2)
    assert lib.shape==(5,6) and names[0]=='1'


def test_bad_degree_rejected():
    with pytest.raises(ValueError): polynomial_library(np.ones((5,2)),5)
