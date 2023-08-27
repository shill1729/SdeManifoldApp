import sympy as sp


def metric_tensor(xi, x):
    j = sp.simplify(xi.jacobian(x))
    g = j.T * j
    return sp.simplify(g)


def matrix_divergence(A : sp.Matrix, x):
    n,m = A.shape
    d = sp.zeros(n,1)
    for i in range(n):
        for j in range(m):
            d[i] += sp.simplify(sp.diff(A[i, j], x[j]))
    return sp.simplify(d)


def coefficients(g: sp.Matrix, x):
    ginv = g.inv()
    ginv = sp.simplify(ginv)
    diffusion = ginv.cholesky(hermitian=False)
    diffusion = sp.simplify(diffusion)
    detg = g.det()
    sqrt_detg = sp.sqrt(detg)
    drift = 0.5*(1./sqrt_detg)*matrix_divergence(sp.simplify(sqrt_detg*ginv), x)
    drift = sp.simplify(drift)
    return drift, diffusion


if __name__ == "__main__":
    theta, phi = sp.symbols("theta phi", real=True)
    x = sp.sin(theta) * sp.cos(phi)
    y = sp.sin(theta) * sp.sin(phi)
    z = sp.cos(theta)
    xi = sp.Matrix([x, y, z])
    coord = sp.Matrix([theta, phi])
    g = metric_tensor(xi, coord)
    g = sp.simplify(g)
    mu, Sigma = coefficients(g, coord)

    print(g)
    print(mu)
    print(Sigma)
