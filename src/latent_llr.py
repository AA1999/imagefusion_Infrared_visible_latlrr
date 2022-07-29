from numpy import array, copy, transpose, eye, zeros, diag, sum, maximum, minimum, absolute
from numpy.linalg import inv, svd
from dataclasses import dataclass

def latent_llr(X: array, lambda_value: float) -> tuple:
    A = copy(X)
    tol = pow(10, -6)
    rho = 1.1
    max_mu = pow(10, 6)
    mu = pow(10, -6)
    maxIter = pow(10, 6)
    d, n = X.shape
    m = A.shape[1]
    atx = transpose(X).dot(X)
    inv_a = inv(transpose(A).dot(A) + eye(m))
    inv_b = inv(A.dot(transpose(A)) + eye(d))
    J = zeros((m, n))
    Z = zeros((m, n))
    L = zeros((d, d))
    S = zeros((d, d))

    E = zeros((d, n))

    Y1 = zeros((d, n))
    Y2 = zeros((m, n))
    Y3 = zeros((d, d))

    iter = 0
    print('initial')

    while iter < maxIter:
        iter += 1
        temp_J = Z + Y2 / mu
        U_J, sigma_J, V_J = svd(temp_J, full_matrices=False)
        sigma_J = diag(sigma_J)
        svp_J = sum(sigma_J > 1 / mu)
        if svp_J >= 1:
            sigma_J = sigma_J[0:svp_J] - 1 / mu
        else:
            svp_J = 1
            sigma_J = array([0])
        J = U_J[:, 0:svp_J].dot(diag(sigma_J)).dot(transpose(V_J[:, 0:svp_J]))

        temp_S = L + Y3 / mu
        U_S, sigma_S, V_S = svd(temp_S, full_matrices=False)
        sigma_S = diag(sigma_S)
        svp_S = sum(sigma_S > 1 / mu)
        if svp_S >= 1:
            sigma_S = sigma_S[0:svp_S] - 1 / mu
        else:
            svp_S = 1
            sigma_S = array([0])
        S = U_S[:, 0:svp_S].dot(diag(sigma_S).dot(transpose(V_S[:, 0:svp_S])))

        #  Z = inv_a*(atx-X'*L*X-X'*E+J+(X'*Y1-Y2)/mu);
        #  L = ((X-X*Z-E)*X'+S+(Y1*X'-Y3)/mu)*inv_b;
        Z = inv_a.dot((atx - transpose(X).dot(L).dot(X) - transpose(X).dot(E) + J + (transpose(X).dot(Y1) - Y2) / mu))
        L = ((X - X.dot(Z) - E).dot(transpose(X)) + S + (Y1.dot(transpose(X)) - Y3) / mu).dot(inv_b)

        xmaz = X - X.dot(Z) - L.dot(X)
        temp = xmaz + Y1 / mu
        temp2 = temp - lambda_value / mu
        temp3 = temp + lambda_value / mu
        E = maximum(0, temp - lambda_value / mu) + minimum(0, temp + lambda_value / mu)

        leq1 = xmaz - E
        leq2 = Z - J
        leq3 = L - S
        max_l1 = absolute(leq1).max()
        max_l2 = absolute(leq2).max()
        max_l3 = absolute(leq2).max()

        stopC1 = max(max_l1, max_l2)
        stopC = max(stopC1, max_l3)

        if stopC < tol:
            print('LLR done.')
            return Z, L, E
        else:
            Y1 = Y1 + mu * leq1
            Y2 = Y2 + mu * leq2
            Y3 = Y3 + mu * leq3
            mu = min(max_mu, mu * rho)
            return Z, L, E