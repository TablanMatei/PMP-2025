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





#Exercitiul 2
np.random.seed(42)
original_image = np.random.randint(0, 2, size=(5, 5))  # Binary image

print("IMAGINEA ORIGINALA:")
print(original_image)


num_pixels = original_image.size
noisy_image = original_image.copy()
num_noisy = int(0.1 * 25)  # 10% din 25 pixeli

for _ in range(num_noisy):
    i = np.random.randint(0, 5)             # Rand random
    j = np.random.randint(0, 5)             # Coloana random
    noisy_image[i, j] = 1 - noisy_image[i, j]         # Flip pixel

print("\nIMAGINEA CU ZGOMOT:")
print(noisy_image)
print(f"\nPixeli modificati: {num_noisy}/{num_pixels}")


model2 = MarkovNetwork()

nodes = []
for i in range(5):
    for j in range(5):
        node_name = f"X_{i}_{j}"
        nodes.append(node_name)
        model2.add_node(node_name)

print(f"\nNUMAR DE NODURI: {len(nodes)}")

edges = []
for i in range(5):
    for j in range(5):
        current = f"X_{i}_{j}"

        if j < 4:
            neighbor = f"X_{i}_{j + 1}"
            edges.append((current, neighbor))

        if i < 4:
            neighbor = f"X_{i + 1}_{j}"
            edges.append((current, neighbor))

model2.add_edges_from(edges)
print(f"NUMAR DE MUCHII: {len(edges)}")


plt.figure(figsize=(12, 10))
pos = {}
for i in range(5):
    for j in range(5):
        pos[f"X_{i}_{j}"] = (j, -i)

nx.draw(model2, pos, with_labels=True, node_color='lightblue',
        node_size=800, font_size=8, font_weight='bold', edge_color='gray', width=1)
plt.title('Markov Random Field', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.show()

# Display images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Imaginea Originala', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(noisy_image, cmap='gray', vmin=0, vmax=1)
axes[1].set_title('Imaginea cu Zgomot', fontsize=12, fontweight='bold')
axes[1].axis('off')
plt.show()

