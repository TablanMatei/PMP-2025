'''
Am vrut sa lucrez local in PyCharm
Aparent PyTensor si vers 3.12 de Python nu se pupa prea bine
Asa ca a trebuitt sa intru la Edit configuration si sa adaug variabila
PYTENSOR_FLAGS cu valoarea optimizer=fast_compile,cxx=

Am luat exemplul de pe GitHub
A trebuit sa il refactorizez si sa organizez codul in functii fiindca
aparent Windows nu poate porni procese paralele daca rulez pm.sample()
intr-un fisier care nu este protejat de: if __name__ == "__main__":

Iar dupa a rulat fara eroare.
Codul acoperea cerintele, l-am inteles si a trebuit doar sa il modific
sa foloseasca datele din problema, vezi #MODIFICAT pe randurile
unde a trebuit sa modific
'''


import pytensor
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Optional debug:
# pytensor.config.cxx = ""
# pytensor.config.optimizer = "fast_compile"


def run_weak_prior_model(data, x_bar):
    with pm.Model() as weak_model:
        mu = pm.Normal("mu", mu=x_bar, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

        trace_weak = pm.sample(
            2000,
            tune=2000,
            target_accept=0.9,
            random_seed=42,
            cores=1
        )
    return trace_weak


def run_strong_prior_model(data):
    with pm.Model() as strong_model:  #MODIFICAT
        mu = pm.Normal("mu", mu=50, sigma=1) # strong belief: mean near 50 dB
        sigma = pm.HalfNormal("sigma", sigma=10)
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

        trace_strong = pm.sample(
            2000,
            tune=2000,
            target_accept=0.9,
            random_seed=42,
            cores=1 #MODIFICAT pentru Windows
        )
    return trace_strong


def main():
    data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])  #MODIFICAT
    x_bar = data.mean()

    print(f"Sample mean = {x_bar:.2f}, Sample std = {data.std(ddof=1):.2f}")

    # Weak prior model
    trace_weak = run_weak_prior_model(data, x_bar)
    summary_weak = az.summary(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)

    print("\nPosterior summaries (Weak Prior):")
    print(summary_weak)

    # Frequentist comparisons
    print("\nFrequentist estimates:")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"SD:   {np.std(data, ddof=1):.2f}")

    # Strong prior model
    trace_strong = run_strong_prior_model(data)
    summary_strong = az.summary(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95)

    print("\nPosterior summaries (Strong Prior):")
    print(summary_strong)

    # Plot weak prior posterior
    az.plot_posterior(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)
    plt.suptitle("Posterior with Weak Prior", fontsize=14)
    plt.show()

    # Plot strong prior posterior
    az.plot_posterior(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95)
    plt.suptitle("Posterior with Strong Prior", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()



'''
Sectiunile de cod care indica rezolvarea fiecarei cerinte:
(Copy-Paste din codul de mai sus)

a)
in main:
data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])  
x_bar = data.mean()  # x = 58.0  
  
# in run_weak_prior_model() :
mu = pm.Normal("mu", mu=x_bar, sigma=10)      # μ ∼ N(x, 10²)  
sigma = pm.HalfNormal("sigma", sigma=10)      # σ ∼ HalfNormal(10)  
y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)


b)
trace_weak = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42, cores=1)
summary_weak = az.summary(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)
print("\nPosterior summaries (Weak Prior):")  
print(summary_weak)


c)
summary_weak = az.summary(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95)
print("\nPosterior summaries (Weak Prior):")
print(summary_weak)

print("\nFrequentist estimates:")
print(f"Mean: {np.mean(data):.2f}")
print(f"SD:   {np.std(data, ddof=1):.2f}")


d)
mu = pm.Normal("mu", mu=50, sigma=1)  # μ ∼ N(50, 1²)
sigma = pm.HalfNormal("sigma", sigma=10)
y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

trace_strong = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42, cores=1)
summary_strong = az.summary(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95)

print("\nPosterior summaries (Strong Prior):")
print(summary_strong)


# Grafice pentru interpretare vizuala
az.plot_posterior(trace_weak, var_names=["mu", "sigma"], hdi_prob=0.95) #Pentru b) și c)
az.plot_posterior(trace_strong, var_names=["mu", "sigma"], hdi_prob=0.95) # Pentru d)
'''

'''
Outputul obtinut, plus cele doua grafice:

C:\Users\Matei\AppData\Local\Microsoft\WindowsApps\.venv\Scripts\python.exe C:\Users\Matei\PycharmProjects\PMP1\.venv\Include\lab7.py 
Sample mean = 58.00, Sample std = 2.00
Initializing NUTS using jitter+adapt_diag...
Sequential sampling (2 chains in 1 job)
NUTS: [mu, sigma]
                                                                               
                              Step      Grad      Sampli…                      
  Progre…   Draws   Diverg…   size      evals     Speed     Elapsed   Remain…  
 ───────────────────────────────────────────────────────────────────────────── 
  -------   4000    0         0.752     3         223.13    0:00:17   0:00:00  
                                                  draws/s                      
  -------   4000    0         0.682     7         108.31    0:00:36   0:00:00  
                                                  draws/s                      
                                                                               
Sampling 2 chains for 2_000 tune and 2_000 draw iterations (4_000 + 4_000 draws total) took 37 seconds.
We recommend running at least 4 chains for robust computation of convergence diagnostics

Posterior summaries (Weak Prior):
         mean     sd  hdi_2.5%  hdi_97.5%  ...  mcse_sd  ess_bulk  ess_tail  r_hat
mu     58.001  0.766    56.415     59.504  ...    0.020    1570.0    1493.0    1.0
sigma   2.348  0.677     1.307      3.706  ...    0.018    1883.0    2166.0    1.0

[2 rows x 9 columns]

Frequentist estimates:
Mean: 58.00
SD:   2.00
Initializing NUTS using jitter+adapt_diag...
Sequential sampling (2 chains in 1 job)
NUTS: [mu, sigma]

'''