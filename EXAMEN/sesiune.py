import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import os
import pytensor


pytensor.config.cxx = ""
pytensor.config.mode = 'FAST_COMPILE'
os.environ['PYTENSOR_FLAGS'] = 'mode=FAST_COMPILE,device=cpu,floatX=float64'

def main():

    df = pd.read_csv('../../bike_daily.csv')
    y = df[['temp_c', 'humidity', 'wind_kph', 'is_holiday', 'season']].values
    X = df[['rentals']].values

    def function(K):

        with pm.Model() as model:
            alpha = pm.Normal('alpha', mu=0, sigma=10)
            beta = pm.Normal('beta', mu=0, sigma=10, shape=3)
            sigma = pm.HalfNormal('sigma', sigma=10)
            mu = alpha + beta * X
            #medv_obs = pm.Normal('medv_obs', mu=mu, sigma=sigma, observed=y)

            trace = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=42, return_inferencedata=True,  idata_kwargs={"log_likelihood": True})

            # mean values
            alpha_mean = trace.posterior["alpha"].mean().values
            beta_mean = trace.posterior["beta"].mean().values
            sigma_mean = trace.posterior["sigma"].mean().values

            print(f"- Intercept (alpha): {alpha_mean:.3f} ")
            print(f"- Slope (beta): {beta_mean:.3f}")
            print(f"- Noise (sigma): {sigma_mean:.3f}")

        return model, trace

    nothing, tracee = function(0)
    beta_means = tracee.posterior['beta'].mean(dim=("chain", "draw")).values
    most_influential_index = np.argmax(np.abs(beta_means))
    most_influential_var = ['temp_c', 'humidity', 'wind_kph', 'is_holiday', 'season'][most_influential_index]
    print("Variabila cu cea mai mare influență asupra medv este:", most_influential_var)



    models = {}
    idatas = {}

    for K in [3, 4, 5]:
        model, idata = function(K)
        models[K] = model
        idatas[K] = idata


    for K in [3, 4, 5]:
        print(az.summary(idatas[K], var_names=["alpha", "beta", "sigma"]))


    print("\nWAIC comparison")
    for K in [3, 4, 5]:
        print(f"K={K}:", az.waic(idatas[K]))

    print("\nLOO comparison")
    for K in [3, 4, 5]:
        print(f"K={K}:", az.loo(idatas[K]))


def crate_new_column(rentals_col):
    global y, x
    Q = 75 * 501 / 100 # hardcodat - fisierul are 500 de linii cu date
    for i in reantals_col:
        if renatals>Q:
            x=1
            y=1
        x=0
        y=0

    data = np.vstack([x, y]).T
    np.savetxt("demand.csv", data, delimiter=",")

def main2():
    df2 = pd.read_csv('../../bike_daily.csv')
    date.head(10)
    col_names = date.columns.tolist()
    rentals_col = col_names[0]
    temp_c_col = col_names[1]
    humidity_col = col_names[2]
    wind_kph_col = col_names[3]
    is_holiday_col = col_names[4]
    season_col = col_names[5]
    crate_new_column(rentals_col)
    df2 = pd.read_csv('../../demand.csv')
    date2.head(10)
    col_names2 = date2.columns.tolist()
    is_high_demand = col_names2[0]

    x_1 = date[rentals_col].values
    x_2 = date[temp_c_col].values
    x_3 = date[humidity_col].values
    x_4 = date[wind_kph_col].values
    x_5 = date[is_holiday_col].values
    x_6 = date[season_col].values
    x_7 = date[is_high_demand].values

    with pm.Model() as model_logistic:
        alpha = pm.Normal('alpha', mu=0, sigma=5)
        beta_1 = pm.Normal('beta_1', mu=0, sigma=5)
        beta_2 = pm.Normal('beta_2', mu=0, sigma=5)
        beta_3 = pm.Normal('beta_3', mu=0, sigma=5)
        beta_4 = pm.Normal('beta_4', mu=0, sigma=5)
        beta_5 = pm.Normal('beta_5', mu=0, sigma=5)
        beta_6 = pm.Normal('beta_6', mu=0, sigma=5)
        beta_7 = pm.Normal('beta_7', mu=0, sigma=5)

        miu = alpha + beta_1 * x_1 + beta_2 * x_2 + beta_3 * x_3 + beta_4 * x_4 + beta_5 * x_5 + beta_6 * x_6 + beta_7 * x_7

        teta = pm.Deterministic('teta', pm.math.sigmoid(miu))

        y_obs = pm.Bernoulli('y_obs', p=teta, observed=y)

        print("\nSampling din posterior")
        idata = pm.sample(2000, tune=1000, return_inferencedata=True,
                          random_seed=42, progressbar=True, target_accept=0.95, cores=1)

    print("Model construit si antrenat")
    summary = az.summary(idata, var_names=['alpha', 'beta_1', 'beta_2'],
                         hdi_prob=0.94)
    print(summary)



    hdi_95 = az.hdi(trace, hdi_prob=0.95)
    print(hdi_95)


    print(f"\n|beta_1| = {abs(beta_1_mean):.4f}, |beta_2| = {abs(beta_2_mean):.4f}")
    if abs(beta_1_mean) > abs(beta_2_mean):
        print(f"beta_1 influenteaza mai mult outcome-ul")
    else:
        print(f"beta_2 influenteaza mai mult outcome-ul")


''''
In laboratorul 13 daca va uitati, am creat un fisier nou cu date. Asta am incercat sa fac si aici, practic pun noua coloana ceruta
intr-un fisier nou creat si o extrag de acolo.  Acela este binary target-ul. Am folosit mult din ce am facut la Lab 12, de asemenea
'''

if __name__ == '__main__':
    main()
    main2()