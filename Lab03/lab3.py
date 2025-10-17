import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

#from pgmpy.models import BayesianNetwork  ------deprecated
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx

#Exercitiul 1
email_model = DiscreteBayesianNetwork([('S', 'L'), ('S', 'O'), ('L', 'M'), ('S', 'M')])

pos = nx.circular_layout(email_model)
nx.draw(email_model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)
plt.show()

#Subpunctul a)
print(email_model.local_independencies(['S','L','O','M']))

#Subpunctul b)
CPD_S = TabularCPD(variable='S', variable_card=2, values=[[0.6], [0.4]])
print(CPD_S)

CPD_O = TabularCPD(variable='O', variable_card=2,values=[[0.9, 0.3],
                                                         [0.1, 0.7]],
                                                        evidence=['S'],
                                                        evidence_card=[2])
print(CPD_O)

CPD_L = TabularCPD(variable='L', variable_card=2,values=[[0.7, 0.2],
                                                         [0.3, 0.8]],
                                                        evidence=['S'],
                                                        evidence_card=[2])
print(CPD_L)

CPD_M = TabularCPD(variable='M', variable_card=2, values=[[0.8, 0.4, 0.5, 0.1],
                                                          [0.2, 0.6, 0.5, 0.9]],
                                                          evidence=['S', 'L'],
                                                          evidence_card=[2, 2])

print(CPD_M)

email_model.add_cpds(CPD_S, CPD_O, CPD_L, CPD_M)
email_model.get_cpds()
email_model.check_model()


infer = VariableElimination(email_model)
posterior_p = infer.query(["S"], evidence={'O':1, 'L':1, 'M':1})
print(posterior_p)



#Exercitiul 2
zar_model = DiscreteBayesianNetwork([('Z', 'A'), ('A', 'R')])
# Z-zar, A-adaugare, R-rezultat
pos = nx.circular_layout(zar_model)
nx.draw(zar_model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)
plt.show()

print(zar_model.local_independencies(['Z','A','R']))

CPD_Z = TabularCPD(variable='Z', variable_card=3, values=[[3/6], [1/6], [2/6]], state_names={'Z':['2or3or5', '6', '1or4']})
print(CPD_Z)

CPD_A = TabularCPD(variable='A', variable_card=3,values=[[1, 0.0, 0.0],
                                                         [0.0, 1, 0.0],
                                                         [0.0, 0.0, 1]
                                                         ],
                                                        evidence=['Z'],
                                                        evidence_card=[3],
                                                        state_names={'A':['black', 'red', 'blue'], 'Z':['2or3or5', '6', '1or4']})
print(CPD_A)

CPD_R = TabularCPD(variable='R', variable_card=2,values=[[0.7, 0.6, 0.7],
                                                         [0.3, 0.4, 0.3]
                                                         ],
                                                        evidence=['A'],
                                                        evidence_card=[3],
                                                        state_names={'R':['not_red', 'red'], 'A':['black', 'red', 'blue']})
print(CPD_R)

zar_model.add_cpds(CPD_Z, CPD_A, CPD_R)
zar_model.check_model()
infer = VariableElimination(zar_model)
posterior_p = infer.query(["R"])
print(posterior_p)
# R(red) = 0.3167 - varianat de rezolvare cu o retea Bayesiana produce rezulatate mai bune.
# De fapt as spune ca produce rezultate perfecte