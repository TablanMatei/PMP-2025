'''
REZUMAT STATISTICI:

Y=0, θ=0.2:
  Posterior n: mean=8.23, median=8.00, std=2.93
  Predictive Y*: mean=1.64, median=1.00, std=1.27

Y=0, θ=0.5:
  Posterior n: mean=5.11, median=5.00, std=2.28
  Predictive Y*: mean=2.56, median=2.00, std=1.61

Y=5, θ=0.2:
  Posterior n: mean=13.20, median=13.00, std=2.87
  Predictive Y*: mean=2.64, median=3.00, std=1.54

Y=5, θ=0.5:
  Posterior n: mean=10.11, median=10.00, std=2.28
  Predictive Y*: mean=5.05, median=5.00, std=1.95

Y=10, θ=0.2:
  Posterior n: mean=18.23, median=18.00, std=2.93
  Predictive Y*: mean=3.65, median=4.00, std=1.78

Y=10, θ=0.5:
  Posterior n: mean=15.11, median=15.00, std=2.28
  Predictive Y*: mean=7.56, median=8.00, std=2.23

Process finished with exit code 0

'''

'''
PUNCTUL B)
Cand Y creste, posterior mean of n creste proportional
Este intuitiv, mai multi cumparatori observati -> mai mullti clienti au vizitat magazinul
θ are efect invers asupra distributiei posteriaore a lui n
Cand θ este mic n este mai mare pentru acelasi Y
Cand θ este mare n este mai mic pentru acelasi Y
Sunt necesari mai putini cumparatori
'''

'''
PUNCTUL D)
Distributia posterioară descrie credinta noastra despre numarul 
de clienti n care au vizitat magazinul,pe baza datelor observate Y. 
Raspunde la intrebarea: Ctti clienți au vizitat?

Distribuția predictiva posterioară descrie predictia pentru numarul
viitor de cumparatori Y* intr-o noua perioada de observație. 
Răspunde la intrebarea: Câti cumparatori vom avea data viitoare?
 
Predictivul are mai multă incertitudine deoarece include atat 
incertitudinea despre n, cat si variabilitatea aleatorie binomiala.
'''

import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

import os
import pytensor

pytensor.config.cxx = ""
pytensor.config.mode = 'FAST_COMPILE'
os.environ['PYTENSOR_FLAGS'] = 'mode=FAST_COMPILE,device=cpu,floatX=float64'

np.random.seed(42)
Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

traces = {}
posterior_predictive = {}

print("PARTEA A: Calcularea distributiilor posterioare")

# a)
for Y in Y_values:
    for theta in theta_values:
        scenario_name = f"Y={Y}, θ={theta}"
        print(f"\nScenario: {scenario_name}")

        with pm.Model() as model:
            n = pm.Poisson('n', mu=10)
            y_obs = pm.Binomial('y_obs', n=n, p=theta, observed=Y)
            trace = pm.sample(2000, tune=1000, return_inferencedata=True,
                              random_seed=42, progressbar=True, cores=1)
            traces[scenario_name] = trace

            print(f"  Mean n posterior: {trace.posterior['n'].mean().values:.2f}")
            print(f"  Std n posterior: {trace.posterior['n'].std().values:.2f}")

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Distributii Posterioare pentru n', fontsize=16, fontweight='bold')

for idx, (scenario_name, trace) in enumerate(traces.items()):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    az.plot_posterior(trace, var_names=['n'], ax=ax,
                      hdi_prob=0.94, point_estimate='mean')
    ax.set_title(scenario_name, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('posterior_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# c) Distribuția predictiva posterioara pentru Y*
print("\nPARTEA C: ")

fig2, axes2 = plt.subplots(3, 2, figsize=(14, 12))
fig2.suptitle('Distributii Predictive Posterioare pentru Y*', fontsize=16, fontweight='bold')

for idx, (scenario_name, trace) in enumerate(traces.items()):
    Y = int(scenario_name.split(',')[0].split('=')[1])
    theta = float(scenario_name.split('=')[2])

    print(f"\nScenario: {scenario_name}")

    # Recream modelul si generam posterior predictive
    with pm.Model() as model:
        n = pm.Poisson('n', mu=10)
        y_obs = pm.Binomial('y_obs', n=n, p=theta, observed=Y)
        y_star = pm.Binomial('y_star', n=n, p=theta)

        ppc = pm.sample_posterior_predictive(
            trace,
            var_names=['y_star'],
            random_seed=42,
            progressbar=False
        )

    posterior_predictive[scenario_name] = ppc

    # Vizualizare
    row = idx // 2
    col = idx % 2
    ax = axes2[row, col]

    y_star_samples = ppc.posterior_predictive['y_star'].values.flatten()

    az.plot_dist(y_star_samples, ax=ax, kind='hist',
                 color='steelblue', hist_kwargs={'bins': 30, 'alpha': 0.7})

    ax.set_title(scenario_name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Y* (numar viitor de cumparători)', fontsize=10)
    ax.set_ylabel('Densitate', fontsize=10)
    ax.axvline(Y, color='red', linestyle='--', linewidth=2,
               label=f'Y observat = {Y}')
    ax.legend()

    print(f"  Mean Y*: {y_star_samples.mean():.2f}")
    print(f"  Std Y*: {y_star_samples.std():.2f}")

plt.tight_layout()
plt.savefig('posterior_predictive_distributions.png', dpi=300, bbox_inches='tight')
plt.show()




print("REZUMAT STATISTICI:")

for scenario_name, trace in traces.items():
    print(f"\n{scenario_name}:")
    print(f"  Posterior n: mean={trace.posterior['n'].mean().values:.2f}, "
          f"median={np.median(trace.posterior['n'].values):.2f}, "
          f"std={trace.posterior['n'].std().values:.2f}")

    y_star_samples = posterior_predictive[scenario_name].posterior_predictive['y_star'].values.flatten()
    print(f"  Predictive Y*: mean={y_star_samples.mean():.2f}, "
          f"median={np.median(y_star_samples):.2f}, "
          f"std={y_star_samples.std():.2f}")
