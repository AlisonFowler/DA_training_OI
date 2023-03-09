#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:32:56 2023

@author: alisonfowler
"""
# Demonstration of optimal interpolation using Streamlit

import math
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def soar2(x1, x2, L):
    '''
    function to compute correlation between two points x1, x2
    according to their separation
    
    uses a SOAR correlation function with length scale L
    (L = 0 => uncorrelated)

    rho = [1 + (dx/L)]*exp(-dx/L); dx = |x1-x2|
    '''

    d = abs(x1 - x2)
    if d == 0:
        return 1.0
    elif L > 0.0:
        return (1.0 + d/L) * math.exp(-d/L)
    else:
        return 0.0


def get_background_function(bg_type, x):
    '''Returns three diffrent options for the background function.'''

    if bg_type == 'Flat':
        return np.zeros(len(x))
    elif bg_type == 'Linear':
        y2 = 0.5
        y1 = -0.5
        x2 = x[-1]
        x1 = x[0]
        m = (y2 - y1) / (x2 - x1)
        return m * (x - x1) + y1
    elif bg_type == 'Sine wave':
        return 0.3 * np.sin(2 * math.pi * x/xn)
    else:
        raise ValueError('Unknown value of bg_type: ' + bg_type)




# set x values at which equations are solved
x0 = 0.
xn = 10.
nx = 100 # will have nx+1 points
x = np.linspace(x0, xn, nx + 1)

# initialise analysis vectors
xdim = len(x)
a  = np.zeros(xdim) # analysis for 2 obs case
a1 = np.zeros(xdim) # analysis for o1 only case
a2 = np.zeros(xdim) # analysis for o2 only case

bg_func ='Flat'#, 'Linear', 'Sine wave'
fb = get_background_function(bg_func, x)
x1=2
x2=3
o1=1
o2=-1
sigo=0.2
sigf=0.2
Lf=1
Lo=0
# These parameters are not adjustable via the Streamlit widgets

# get value of background field at observation locations, Hxb

# Create a function that estimates the background at any point
# by interpolating linearly between the given background points
Hfb = interpolate.interp1d(x, fb, 'linear')

# Compute y-Hxb at each obs location
d1 = o1 - Hfb(x1)
d2 = o2 - Hfb(x2)

# Compute weighted correlation for each obs
cf12 = soar2(x1, x2, Lf) # background
co12 = soar2(x1, x2, Lo) # observations

sum_sigma = sigf*sigf+sigo*sigo

sum_C = sigf*sigf*cf12 + sigo*sigo*co12 # var_b*rho(x1,x2) + var_o*rho(x1,x2)


# structure functions of B
BH1 = sigf*sigf*np.array([soar2(xi, x1, Lf) for xi in x]) # rho(x,x1), (xdim x 1) vector
BH2 = sigf*sigf*np.array([soar2(xi, x2, Lf) for xi in x]) # rho(x,x2), (xdim x 1) vector
  
Nf =   sum_sigma*sum_sigma - sum_C*sum_C

incr = (BH1*(sum_sigma*d1-sum_C*d2)+BH2*(sum_sigma*d2-sum_C*d1))/Nf
  
a  = fb + incr        # analysis for 2 obs case

# Plot the data
fig = plt.figure()
plt.errorbar([x1, x2], [o1, o2], [sigo, sigo], marker='o', linestyle='None', label="Observations")  # observation locations
plt.plot(x, a, 'r--', label="Analysis")        # analysis in red dashed line
#plt.plot(x, a1, 'b:', label="Analysis (first obs only)")  # analysis with o1 only in blue dotted line
#plt.plot(x, a2, 'b-', label="Analysis (second obs only)") # analysis with o2 only in blue dashed line
plt.plot(x, fb, 'k', label="Background")                  # background in black solid line
plt.legend()
plt.xlim(0, 10)
plt.ylim(-1.2, 1.2)
