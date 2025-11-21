import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import random

#from pgmpy.models import BayesianNetwork  ------deprecated
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx


#Exercitiul 1
home_model = DiscreteBayesianNetwork([('O', 'H'), ('O', 'W'), ('W', 'R'), ('H', 'R'), ('H', 'E'), ('R', 'C'),])

pos = nx.circular_layout(home_model)
nx.draw(home_model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)
plt.show()

#Subpunctul a)
print(home_model.local_independencies(['O','H','W','R','E','C']))

CPD_O = TabularCPD(variable='O', variable_card=2, values=[[0.3], [0.7]],
                       state_names={'O': ['Cold', 'Mild']})

CPD_H = TabularCPD(variable='H', variable_card=2,values=[[0.9, 0.2],
                                                         [0.1, 0.8]],
                                                        evidence=['O'],
                                                        evidence_card=[2],
                   state_names={'H': ['Yes', 'No'], 'O': ['Cold', 'Mild']})

CPD_W = TabularCPD(variable='W', variable_card=2,values=[[0.1, 0.6],
                                                         [0.9, 0.4]],
                                                        evidence=['O'],
                                                        evidence_card=[2],
                   state_names={'W': ['Yes', 'No'], 'O': ['Cold', 'Mild']})

CPD_R = TabularCPD(variable='R', variable_card=2,values=[[0.6, 0.3, 0.9, 0.5],
                                                          [0.4, 0.7, 0.1, 0.5]],
                                                        evidence=['W', 'H'],
                                                        evidence_card=[2,2],
                   state_names={'R': ['Warm', 'Cool'], 'H': ['Yes', 'No'], 'W': ['Yes', 'No']})

CPD_E = TabularCPD(variable='E', variable_card=2,values=[[0.8, 0.2],
                                                         [0.2, 0.8]],
                                                        evidence=['H'],
                                                        evidence_card=[2],
                   state_names={'E': ['High', 'Low'], 'H': ['Yes', 'No']})

CPD_C = TabularCPD(variable='C', variable_card=2,values=[[0.85, 0.40],
                                                         [0.15, 0.60]],
                                                        evidence=['R'],
                                                        evidence_card=[2],
                   state_names={'C': ['Comfortable', 'Uncomfortable'], 'R': ['Warm', 'Cool']})
print(CPD_O)
print(CPD_H)
print(CPD_W)
print(CPD_R)
print(CPD_E)
print(CPD_C)

home_model.add_cpds(CPD_O, CPD_H, CPD_W, CPD_R, CPD_E,CPD_C)
home_model.get_cpds()
home_model.check_model()


infer = VariableElimination(home_model)
posterior_p1 = infer.query(["H"], evidence={'C':'Comfortable'})
posterior_p2 = infer.query(["E"], evidence={'C':'Comfortable'})
print(posterior_p1)
print(posterior_p2)