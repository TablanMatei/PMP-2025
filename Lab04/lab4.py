import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx

from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor

#Exercitiul 1 a)
model = MarkovNetwork([('A1', 'A2'), ('A1', 'A3'), ('A2', 'A4'), ('A2', 'A5'), ('A3', 'A4'), ('A4', 'A5')])
pos = nx.circular_layout(model)
nx.draw(model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)
plt.show()

cliques = list(nx.find_cliques(model))
cliques_sorted = sorted([sorted(clique) for clique in cliques])
for i, clique in enumerate(cliques, 1):
    print(f"{i}: {clique}")


#Exercitiul 1 b)
factor_a1_a2 = DiscreteFactor(variables=['A1', 'A2'], cardinality=[2, 2], values=[np.exp(-1-1), np.exp(-1+1), np.exp(1-1), np.exp(1+1)])
factor_a1_a3 = DiscreteFactor(variables=['A1', 'A3'], cardinality=[2, 2], values=[np.exp(-1-1), np.exp(-1+1), np.exp(1-1), np.exp(1+1)])
factor_a3_a4 = DiscreteFactor(variables=['A1', 'A4'], cardinality=[2, 2], values=[np.exp(-1-1), np.exp(-1+1), np.exp(1-1), np.exp(1+1)])
factor_a2_a4_a5 = DiscreteFactor(variables=['A2', 'A4', 'A5'], cardinality=[2, 2, 2],
            values=[np.exp(-1-1-1), np.exp(-1-1+1), np.exp(-1+1-1), np.exp(-1+1+1), np.exp(1-1-1), np.exp(1-1+1), np.exp(1+1-1), np.exp(1+1+1)])

model.add_factors(factor_a1_a2, factor_a1_a3, factor_a3_a4, factor_a2_a4_a5)
model.get_factors()

for factor in model.get_factors():
    print(factor)
    print()

print(model.get_local_independencies())

joint = factor_a1_a2 * factor_a1_a3 * factor_a3_a4 * factor_a2_a4_a5
Sum = np.sum(joint.values)
joint.values = joint.values / Sum
print(joint)