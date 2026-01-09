import os
os.environ["PYTENSOR_FLAGS"] = "linker=py,optimizer=fast_compile,cxx="
import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pytensor
import os


az.style.use('arviz-darkgrid')


def create_new_file():  #Subcpunctul 2
    N = 500
    x = np.linspace(-2, 2, N)
    y = np.sin(x) + np.random.normal(0, 0.2, N)

    data = np.vstack([x, y]).T
    np.savetxt("dummy_500.csv", data, delimiter=",")

def main():

    #dummy_data = np.loadtxt('../../dummy.csv')  #Subcpunctul 1
    dummy_data = np.loadtxt('../../dummy_500.csv', delimiter=',')   #Subcpunctul 2

    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]

    #order = 2
    order = 5               #SUBPUNCTUL 0) 0.5p

    x_1p = np.vstack([x_1**i for i in range(1, order+1)])

    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))/x_1p.std(axis=1, keepdims=True)

    y_1s = (y_1 - y_1.mean()) / y_1.std()

    plt.scatter(x_1s[0], y_1s)


    plt.xlabel('x')
    plt.ylabel('y')

    #plt.show()


    #SUBPUNCTUL a) 0.5p
    with pm.Model() as model_p:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        #beta = pm.Normal('beta', mu=0, sigma=10, shape=order) #pentru a
        #beta = pm.Normal('beta', mu=0, sigma=100, shape=order)     #SUBPUNCTUL b) 1p
        beta = pm.Normal('beta', mu=0,  sigma = np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order) #SUBPUNCTUL b)

        epsilon = pm.HalfNormal('epsilon', 5)
        miu = alpha + pm.math.dot(beta, x_1s)
        #y_pred = pm.Normal('y_pred', mu=miu, sigma=epsilon, observed=y_1s)
        idata_p = pm.sample(
            1000,
            tune=1000,
            chains=2,
            cores=2,
            target_accept=0.9
        )

    alpha_p_post = idata_p.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_post = idata_p.posterior['beta'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = alpha_p_post + beta_p_post @ x_1s

    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #create_new_file()    #Subcpunctul 2 0.5p
    main()