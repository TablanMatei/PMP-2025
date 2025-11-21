import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from hmmlearn.hmm import CategoricalHMM

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




#Exerictiul2
states = ["Walking", "Running", "Resting"]
n_states = len(states)
print('Number of hidden states :',n_states)

observations = ["Medium", "Low", "High"]
n_observations = len(observations)
print('Number of observations  :',n_observations)

start_probability = np.array([1/3, 1/3, 1/3])

transition_probability = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
    [0.3, 0.2, 0.5]
])

emission_probability = np.array([
    [0.1, 0.7, 0.2],
    [0.05, 0.25, 0.7],
    [0.8, 0.15, 0.05]
])

model = CategoricalHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability



state_to_index = {"Medium": 0, "High": 1, "Low": 1}
observations_sequence = np.array([[state_to_index[g]] for g in ["Medium","High","Low"]])


G = nx.DiGraph()

for i in range(n_states):
    for j in range(n_states):
        if transition_probability[i, j] > 0:
            G.add_edge(states[i], states[j], weight=transition_probability[i, j])

pos = nx.circular_layout(G)
edges = G.edges(data=True)

nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', arrows=True)
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()

logprob_score = model.score(observations_sequence)
prob_score = np.exp(logprob_score)

print()
print("Log probability (score) :", logprob_score)
print("Probability (score)     :", prob_score)
print()

hidden_state_names = states
hidden_states = model.predict(observations_sequence)
print("Most likely hidden states (test difficulties):")
print([hidden_state_names[i] for i in hidden_states])


log_probability, hidden_states_viterbi = model.decode(observations_sequence,  lengths = len(observations_sequence), algorithm='viterbi')

print('Log Probability :',log_probability)
print('Probability :',np.exp(log_probability))
print("Most likely hidden states:")
print([hidden_state_names[i] for i in hidden_states_viterbi])