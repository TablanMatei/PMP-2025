import numpy as np
import pymc as pm
import pandas as pd
import seaborn as sns
import arviz as az
import matplotlib.pyplot as plt
import os
import pytensor

# Ca si la laboratoarele precedente, trebuei configurat PyTensor
pytensor.config.cxx = ""
pytensor.config.mode = 'FAST_COMPILE'
os.environ['PYTENSOR_FLAGS'] = 'mode=FAST_COMPILE,device=cpu,floatX=float64'

def main():
    # Extragere date
    date = pd.read_csv('../../date_promovare_examen.csv')
    date.head(10)
    col_names = date.columns.tolist()
    ore_studiu_col = col_names[0]
    ore_somn_col = col_names[1]
    promovat_col = col_names[2]



    # Vizualizare date
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    # Stripplot pentru ore studiu
    ax1 = axes[0]
    sns.stripplot(x=promovat_col, y=ore_studiu_col, data=date, jitter=True,
                  hue=promovat_col, ax=ax1, legend=False, palette=['#e74c3c', '#2ecc71'])
    ax1.set_title('Ore studiu vs Promovare', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Promovat (0=Nu, 1=Da)')
    ax1.set_ylabel('Ore studiu/săptămână')

    # Stripplot pentru ore somn
    ax2 = axes[1]
    sns.stripplot(x=promovat_col, y=ore_somn_col, data=date, jitter=True,
                  hue=promovat_col, ax=ax2, legend=False, palette=['#e74c3c', '#2ecc71'])
    ax2.set_title('Ore somn vs Promovare', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Promovat (0=Nu, 1=Da)')
    ax2.set_ylabel('Ore somn/zi')

    plt.tight_layout()
    plt.savefig('vizualizare_initiala.png', dpi=300, bbox_inches='tight')
    plt.show()


    # Subpunctul a)
    print(f"\nDistribuția clasei '{promovat_col}':")
    print(date[promovat_col].value_counts().sort_index())
    print(f"\nProportii:")
    proportii = date[promovat_col].value_counts(normalize=True).sort_index()
    print(proportii)

    diff = abs(proportii.iloc[0] - 0.5)
    if diff < 0.1:
        print(f"\nDatele sunt BALANSATE (diferenta fata de 50%: {diff:.2%})")
    else:
        print(f"\nDatele sunt NEBALANSATE (diferenta fata de 50%: {diff:.2%})")


    # Subpunctul b):
    # Extragere date
    y = date[promovat_col].values
    x_1 = date[ore_studiu_col].values
    x_2 = date[ore_somn_col].values
    x_1_std = (x_1 - x_1.mean()) / x_1.std() #date standardizate
    x_2_std = (x_2 - x_2.mean()) / x_2.std()

    #Construire model
    with pm.Model() as model_logistic:
        alpha = pm.Normal('alpha', mu=0, sigma=5)
        beta_1 = pm.Normal('beta_1', mu=0, sigma=5)  # Coeficient ore studiu
        beta_2 = pm.Normal('beta_2', mu=0, sigma=5)  # Coeficient ore somn

        # Combinatie liniara
        miu = alpha + beta_1 * x_1_std + beta_2 * x_2_std

        # Functia logistica (sigmoid)
        teta = pm.Deterministic('teta', pm.math.sigmoid(miu))

        # Likelihood (distributie Bernoulli)
        y_obs = pm.Bernoulli('y_obs', p=teta, observed=y)

        # Sampling
        print("\nSampling din posterior")
        idata = pm.sample(2000, tune=1000, return_inferencedata=True,
                          random_seed=42, progressbar=True, target_accept=0.95, cores=1)

    print("Model construit si antrenat")
    summary = az.summary(idata, var_names=['alpha', 'beta_1', 'beta_2'],
                         hdi_prob=0.94)
    print(summary)

    posterior = idata.posterior
    alpha_mean = posterior['alpha'].mean().values
    beta_1_mean = posterior['beta_1'].mean().values
    beta_2_mean = posterior['beta_2'].mean().values

    #Frontiera de decizie
    bd_intercept_mean = -alpha_mean / beta_2_mean
    bd_slope_mean = -beta_1_mean / beta_2_mean

    print(f"\nFrontiera de decizie (date standardizate):")
    print(f"x_2_std = {bd_intercept_mean:.4f} + ({bd_slope_mean:.4f}) * x_1_std")

    # Evaluare separabilitate
    teta_samples = posterior['teta'].values
    teta_mean = teta_samples.mean(axis=(0, 1))

    y_pred = (teta_mean > 0.5).astype(int)
    accuracy = (y_pred == y).mean()

    print(f"\nAcuratețe: {accuracy:.2%}")
    if accuracy > 0.85:
        print("Datele sunt bine separate")
    elif accuracy > 0.70:
        print("Datele sunt moderat separate")
    else:
        print("Datele sunt slab separate")


    #Subpunctul c)
    print(f"\n|beta_1| = {abs(beta_1_mean):.4f}, |beta_2| = {abs(beta_2_mean):.4f}")
    if abs(beta_1_mean) > abs(beta_2_mean):
        print(f"Orele de studiu influenteaza mai mult promovabilitatea")
    else:
        print(f"Orele de somn influenteaza mai mult promovabilitatea")

if __name__ == '__main__':
    main()