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



#Exercitiul 2
# a) Simularea jocului de 1000 de ori
random.seed(50)
simulations = 10000
p0_wins = 0
p1_wins = 0

for _ in range(simulations):
    starter = random.choice([0, 1])
    n = random.randint(1, 6)
    other_player = 1 - starter
    m = 0

    for _ in range(2 * n):
        if other_player == 0:  # P0 fair
            if random.randint(0, 1) == 1:  # 1/2 sanse
                m += 1
        else:  # P1 trucat
            if random.randint(1, 7) <= 4:  # 4/7 sanse
                m += 1


    if n >= m:
        if starter == 0: p0_wins += 1
        else: p1_wins += 1
    else:
        if other_player == 0: p0_wins += 1
        else: p1_wins += 1


print(f"Simulare (10000 jocuri):")
print(f"P0 câștigă: {p0_wins} ({p0_wins/simulations*100:.2f}%)")
print(f"P1 câștigă: {p1_wins} ({p1_wins/simulations*100:.2f}%)")
print()

#b) Retea Bayesiana
game_model = DiscreteBayesianNetwork([   ('T', 'S'),  ('S', 'D'),  ('S', 'C'),  ('D', 'C')  ])

# Toss
CPD_T = TabularCPD(variable='T', variable_card=2,
                   values=[[0.5],   # T=0
                           [0.5]])  # T=1

# Starter
CPD_S = TabularCPD(variable='S', variable_card=2,
                   values=[[1, 0],   # T=0->S=0
                           [0, 1]],  # T=1->S=1
                   evidence=['T'],
                   evidence_card=[2])

# Dice
CPD_D = TabularCPD(variable='D', variable_card=6,
                   values=[[1/6, 1/6],
                           [1/6, 1/6],
                           [1/6, 1/6],
                           [1/6, 1/6],
                           [1/6, 1/6],
                           [1/6, 1/6]],
                   evidence=['S'],
                   evidence_card=[2])

# Coin
coin_values = [[], []]
for s in [0, 1]:  # Starter
    for n in range(1, 7):  # Dice
        other = 1-s
        p_heads = 0.5 if other==0 else 4/7

        # Formula Combinari de 'n' luate cate 'k'
        prob_m1 = 2 * n * p_heads * ((1 - p_heads) ** (2 * n - 1))
        prob_not_m1 = 1 - prob_m1

        coin_values[0].append(prob_not_m1)  # C=0
        coin_values[1].append(prob_m1)      # C=1

CPD_C = TabularCPD(variable='C', variable_card=2,
                   values=coin_values,
                   evidence=['S', 'D'],
                   evidence_card=[2, 6])

game_model.add_cpds(CPD_T, CPD_S, CPD_D, CPD_C)
pos = nx.circular_layout(game_model)
nx.draw(game_model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)
plt.show()

#c) Inferenta  P(S | C=1)
inference = VariableElimination(game_model)
result = inference.query(variables=['S'], evidence={'C': 1})
print("P(Starter | m=1 heads in round 2):")
print(result)
print()

























