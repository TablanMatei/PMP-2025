"""
Rezultate obtinute la executie:
a) (0.5p) estimates the regression coefficients (intercept and slope);
- Intercept (alpha): 3.392
- Slope (beta): 1.704
- Noise (sigma): 0.403
b) 94% CREDIBLE INTERVALS (HDI) FOR COEFFICIENTS

Highest Density Intervals (94%):
alpha: [2.960, 3.784]
beta:  [1.642, 1.761]
sigma: [0.273, 0.534]
c) (0.5p) predicts future revenues for new levels of advertising expenses.

Predicting sales for new advertising expenses: [ 3.5  7.  12. ]
Sampling: [sales_pred]
Sampling ... ---------------------------------------- 100% 0:00:00 / 0:00:00
Advertising = 3.5 -> Sales: from 8.543767298152718 to $10.171471411013345 (mean: $9.350098429108009)
Advertising = 7.0 -> Sales: from 14.520446347279432 to $16.09528859934049 (mean: $15.311564665070625)
Advertising = 12.0 -> Sales: from 22.947814797423543 to $24.679809110213437 (mean: $23.828670034955852)
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os
import pytensor

# Ca si la laboratoarele precedente, trebuei configurat PyTensor
pytensor.config.cxx = ""
pytensor.config.mode = 'FAST_COMPILE'
os.environ['PYTENSOR_FLAGS'] = 'mode=FAST_COMPILE,device=cpu,floatX=float64'


def main():
    # Datele luate din enuntul problemei:
    publicity = np.array([1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
                          6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])
    sales = np.array([5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0,
                      15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0])

    new_publicity = np.array([3.5, 7.0, 12.0])


    with pm.Model() as model:
        x = pm.Data("publicity", publicity)

        alpha = pm.Normal("alpha", mu=10, sigma=10)
        beta = pm.Normal("beta", mu=2, sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=5)

        mu = alpha + beta * x
        pm.Normal("sales", mu=mu, sigma=sigma, observed=sales)

        print("\nSampling from posterior distribution...")
        idata = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=42)


        print("a) (0.5p) estimates the regression coefficients (intercept and slope);")
        # mean values
        alpha_mean = idata.posterior["alpha"].mean().values
        beta_mean = idata.posterior["beta"].mean().values
        sigma_mean = idata.posterior["sigma"].mean().values

        print(f"- Intercept (alpha): {alpha_mean:.3f} ")
        print(f"- Slope (beta): {beta_mean:.3f}")
        print(f"- Noise (sigma): {sigma_mean:.3f}")


        print("b) 94% CREDIBLE INTERVALS (HDI) FOR COEFFICIENTS")

        # HDI l-am setat 94%, la fel ca in exemplul dat pe suportul de lab
        coef_hdi = az.hdi(idata, var_names=["alpha", "beta", "sigma"], hdi_prob=0.94)
        print("\nHighest Density Intervals (94%):")

        alpha_hdi = coef_hdi["alpha"].values
        beta_hdi = coef_hdi["beta"].values
        sigma_hdi = coef_hdi["sigma"].values

        print(f"alpha: [{alpha_hdi[0]:.3f}, {alpha_hdi[1]:.3f}]")
        print(f"beta:  [{beta_hdi[0]:.3f}, {beta_hdi[1]:.3f}]")
        print(f"sigma: [{sigma_hdi[0]:.3f}, {sigma_hdi[1]:.3f}]")


    print("c) (0.5p) predicts future revenues for new levels of advertising expenses.")

    # Model nou de predictie fara datele observate
    with pm.Model() as prediction_model:
        x_new = pm.Data("publicity_new", new_publicity)

        alpha = pm.Normal("alpha", mu=10, sigma=10)
        beta = pm.Normal("beta", mu=2, sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=5)

        mu_new = alpha + beta * x_new
        sales_pred = pm.Normal("sales_pred", mu=mu_new, sigma=sigma)

        # SSampling nou
        print(f"\nPredicting sales for new advertising expenses: {new_publicity}")
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["sales_pred"],
            random_seed=42
        )

                                        # 94% se obtine excluzand 3% din stg si dreapta pe o scara de la [0,100]
    pred_int = np.percentile(ppc.posterior_predictive["sales_pred"], [3, 97], axis=(0, 1))
    for i, adv in enumerate(new_publicity):
        mean_pred = ppc.posterior_predictive["sales_pred"].isel(sales_pred_dim_0=i).mean().values
        lo = pred_int[0, i]
        hi = pred_int[1, i]
        print(f"Advertising = {adv} -> Sales: from {lo} to ${hi} (mean: ${mean_pred})")


if __name__ == '__main__':
    main()