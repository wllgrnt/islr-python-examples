"""
conceptualQ3.py

Plot the classification error, Gini index, and cross-entropy for a two-class
problem as p_m1 goes from 0 to 1
"""
import math
import numpy as np
import matplotlib.pyplot as plt
def classificationError(p_m1):
    """
    This is defined as E 1 - max_k{p_mk}
    """
    p_m2 = 1 - p_m1
    E = 1 - max(p_m1, p_m2)  # You could do E=min(p_m1, p_m2) instead 
    return E

def giniIndex(p_m1):
    """
    G = sum_k { p_mk(1-p_mk }
    """
    G = p_m1*(1-p_m1)*2  
    return G

def crossEntropy(p_m1):
    """
    D = - sum_k { p_mk*log(p_mk) }
    """
    p_m2 = 1 - p_m1
    D = - p_m1*math.log(p_m1) - p_m2*math.log(p_m2)
    return D



classificationErrors = []
ginis = []
crossEntropies = []
p_m1s = np.linspace(0.001,0.999)
for p_m1 in p_m1s:
    print(p_m1)
    classificationErrors.append(classificationError(p_m1))
    ginis.append(giniIndex(p_m1))
    crossEntropies.append(crossEntropy(p_m1))

plt.scatter(p_m1s, classificationErrors)
plt.scatter(p_m1s, ginis)
plt.scatter(p_m1s, crossEntropies)
plt.show()

