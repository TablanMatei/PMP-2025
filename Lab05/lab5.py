print("Lab5!")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import CategoricalHMM
import networkx as nx

#Subpunctul a)

states = ["Difficult", "Medium", "Easy"]
n_states = len(states)
print('Number of hidden states :',n_states)

observations = ["FB", "B", "S", "NS"]
n_observations = len(observations)
print('Number of observations  :',n_observations)

start_probability = np.array([1/3, 1/3, 1/3]) #The probability of the difficulty of the first test is equal for all three.

transition_probability = np.array([
    [0.0, 0.5, 0.5], #If, at some point, a difficult test is given, the next test can only be medium or easy, with equal probability.
    [0.5, 0.25, 0.25], # However, if a medium or easy test is given, then the next test will be difficult with probability 0.5, or medium or easy with equal probability 0.25.
    [0.5, 0.25, 0.25] # However, if a medium or easy test is given, then the next test will be difficult with probability 0.5, or medium or easy with equal probability 0.25.
])

emission_probability = np.array([
    [0.1, 0.2, 0.4, 0.3], #Difficult test:
    [0.15, 0.25, 0.5, 0.1], #Medium test:
    [0.2, 0.3, 0.4, 0.1] #Easy test:
])

model = CategoricalHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability



grade_to_index = {"FB": 0, "B": 1, "S": 2, "NS": 3}
observations_sequence = np.array([[grade_to_index[g]] for g in ["FB","FB","S","B","B","S","B","B","NS","B","B"]])


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


#Subpunctul b)
logprob_score = model.score(observations_sequence)
prob_score = np.exp(logprob_score)

print()
print("Log probability (score) :", logprob_score)
print("Probability (score)     :", prob_score)
print()


#Subpunctul c)
hidden_state_names = states
hidden_states = model.predict(observations_sequence)
print("Most likely hidden states (test difficulties):")
print([hidden_state_names[i] for i in hidden_states])


log_probability, hidden_states_viterbi = model.decode(observations_sequence,  lengths = len(observations_sequence), algorithm='viterbi')

print('Log Probability :',log_probability)
print('Probability :',np.exp(log_probability))
print("Most likely hidden states:")
print([hidden_state_names[i] for i in hidden_states_viterbi])