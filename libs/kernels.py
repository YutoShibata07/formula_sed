import GPy

kernels_list = [
    ["RBF", GPy.kern.RBF],  # variance, lengthscale
    ["Exponential", GPy.kern.Exponential],  # variance, lengthscale
    ["Cosine", GPy.kern.Cosine],  # variance, lengthscale
    ["Linear", GPy.kern.Linear],  # variances
    ["sde_Brownian", GPy.kern.sde_Brownian],  # variance
    ["Poly", GPy.kern.Poly],  # variance, scale, bias, order
    ["StdPeriodic", GPy.kern.StdPeriodic],  # variance, period, lengthscale
    ["PeriodicExponential", GPy.kern.PeriodicExponential],  # variance, lengthscale, period=6.283185307179586, n_freq=10, lower=0.0, upper=12.566370614359172
]


n_kernel_kinds = len(kernels_list)
def kernel_sampler(rng=None, variance=None, lengthscale=None, period=None):
    kernels = [
        ["RBF", GPy.kern.RBF(1, variance, lengthscale)],  # variance, lengthscale
        ["Exponential", GPy.kern.Exponential(1, variance, lengthscale)],  # variance, lengthscale
        ["Cosine", GPy.kern.Cosine(1, variance, lengthscale)],  # variance, lengthscale
        ["Linear", GPy.kern.Linear(1, variance)],  # variances
        ["sde_Brownian", GPy.kern.sde_Brownian(1, variance)],  # variance
        ["Poly", GPy.kern.Poly(1, variance, lengthscale, bias=1., order=3.)],  # variance, scale, bias, order
        ["StdPeriodic", GPy.kern.StdPeriodic(1, variance, period, lengthscale)],  # variance, period, lengthscale
        ["PeriodicExponential", GPy.kern.PeriodicExponential(1, variance, lengthscale, period)],  # variance, lengthscale, period
    ]
    k = rng.integers(0, n_kernel_kinds)
    return kernels[k]