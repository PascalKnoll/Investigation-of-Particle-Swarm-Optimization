# %%
# %pip install pandas
# %pip install numpy
# %pip install sklearn
# %pip install matplotlib
# %pip install pyswarms

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pyswarms as ps

# %%

target_func = lambda X: (20 + X**2 - 10 * np.cos(2*np.pi*X)).sum(axis=0)

def generate_sample(n, n_dims, lower, upper, target_func):
    """
    Generates data sample 1.
    """
    np.random.seed(42)
    X = np.random.rand(n_dims, n)
    for i in range(n_dims):
        X[i] = X[i]*(upper[i]-lower[i]) + lower[i]
    y = target_func(X)
    
    return (X,y)

# %%
N = 70
sample1 = generate_sample(N, 2, (-2.5, -1.5), (1.5, 2.5), target_func)

# %%
plt.scatter(sample1[0][0], sample1[0][1], c=sample1[1])

# %%
sample_test = generate_sample(1000000, 2, (-2.5, -1.5), (1.5, 2.5), target_func)
plt.scatter(sample_test[0][0], sample_test[0][1], c=sample_test[1])

# %%
m = np.arange(-2.5,1.5,0.001)
p = np.arange(-1.5,2.5,0.001)

X = np.array(np.meshgrid(m, p))
Z = target_func(X)

plt.pcolormesh(m,p,Z)
plt.colorbar()

# %%
topo = ps.backend.topology.Star()
n_dims = 2

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
def optim(obj_func, init_theta, bounds):
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=len(init_theta), dimensions=2, options=options, bounds=bounds)
        print("IM HERE")
        f_opt, theta_opt = optimizer.optimize(obj_func, iters=1, verbose=True)
        
        return f_opt, theta_opt

# Q: I get this error: File <__array_function__ internals>:180, in copyto(*args, **kwargs) ValueError: could not broadcast input array from shape (2,) into shape (70,70)
# Q: What is the problem and how can i fix this?
# A: The problem is that the optimizer is trying to fit a 2D function to a 1D data set.
# A: The solution is to use the following:

# %%
def gaussian_process_regressor(x, y):
    """
    This function is used to test the gaussian process regressor.
    """
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, optimizer=optim)
    gpr.fit(x, y)

    return gpr

# %%
X, y = sample1
X = X.T
y = y[..., None]
gpr = gaussian_process_regressor(X, y)

# %%
X_test, y_test = sample_test
X_test = X_test.T
y_test = y_test[..., None]

# %%
# predict and plot the prediction of gpr
X_test = X_test[:10000,:]
y_pred = gpr.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred[None,...])
plt.colorbar()

# %%
    # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    # gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    # gpr.fit(x[:, np.newaxis], y)

    # # Predict
    # x_new = np.linspace(0, 1, 1000)
    # y_new = gpr.predict(x_new[:, np.newaxis])

    # # Plot the results
    # plt.figure(figsize=(10, 5))
    # plt.scatter(x, y, label="Training data")
    # plt.plot(x_new, y_new, label="Prediction")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend()
    # plt.show()
    # return gpr

# %%


# %%



