import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# Seed pentru reproducibilitate
np.random.seed(12)

# Parametrii reali
true_mu = 170       # Înălțimea medie reală în cm
true_sigma = 10     # Deviația standard reală în cm

# Generăm datele observate (înălțimile studenților)
observed_heights = np.random.normal(true_mu, true_sigma, size=100)

# Afișăm un histogram al înălțimilor observate
plt.hist(observed_heights, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Distribuția Înălțimilor Observate ale Studenților')
plt.xlabel('Înălțime (cm)')
plt.ylabel('Număr de Studenți')
plt.show()