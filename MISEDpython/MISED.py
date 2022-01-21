import numpy as np
from .density_derivative import DensityDerivative, KernelInfo


def MISED_full(X, bnum=100, sigma_list=None, lda_list=None, cvfold=5):
    '''
    Mean integrated squared error for derivatives
    i.e. density-derivative estimation

    Estimates dp(x) / dx_j for all dimensions j from samples {x_i}_{i=1}^{n}
    '''

    # Convert array format to correct shape
    if len(X.shape) == 1:
        X = X[np.newaxis, :]
    else:
        X = X.transpose()

    (dim, samples) = X.shape

    bnum = min(samples, 100 if bnum is None else bnum)
    if sigma_list is None:
        sigma_list = np.logspace(-0.3, 1, 9)
    if lda_list is None:
        lda_list = np.logspace(-1, 1, 9)

    # We only work with a number of bnum centers instead of samples
    cind = np.random.permutation(samples)[0:bnum]
    C = X[:, cind]

    # Difference from centers, i.e. C-X (size: bnum x samples x dim)
    CX_diff = np.tile(C.T[:, np.newaxis], (1, samples, 1)) - np.tile(X.T[np.newaxis, :], (bnum, 1, 1))

    # Distance from centers (size: bnum x samples)
    CX_dist = np.sum(CX_diff ** 2, axis=2)

    # Difference and distance from centers to centers (size: bnum x bnum (x dim))
    CC_diff = np.tile(C.T[:, np.newaxis], (1, bnum, 1)) - np.tile(C.T[np.newaxis, :], (bnum, 1, 1))
    CC_dist = np.sum(CC_diff ** 2, axis=2)

    # Cross-validation parameters
    cv_fold = np.arange(cvfold)
    cv_split = np.floor(np.arange(samples) * cvfold / samples)
    cv_index = cv_split[np.random.permutation(samples)]

    # String the learned parameters
    sigma = np.empty(dim)
    lda = np.empty(dim)

    for dd in range(dim):
        score_cv = np.empty((len(sigma_list), len(lda_list), len(cv_fold)))

        for sigma_index, sigma_tmp in enumerate(sigma_list):
            # Calculate system matrix
            H = (np.sqrt(np.pi) * sigma_tmp)**dim * np.exp(-CC_dist / (4 * sigma_tmp**2))

            # Create kernel
            GauKer = np.exp(-CX_dist / (2. * sigma_tmp**2))

            for kk in cv_fold:
                # Calculate vectors for traning
                phi_train = CX_diff[:, cv_index != kk, dd] * GauKer[:, cv_index != kk] / sigma_tmp**2
                h_train = np.mean(phi_train, axis=1)

                # Calculate vectors for testing
                phi_test = CX_diff[:, cv_index == kk, dd] * GauKer[:, cv_index == kk] / sigma_tmp**2
                h_test = np.mean(phi_test, axis=1)

                for lda_index, lda_tmp in enumerate(lda_list):
                    # Solve system for train data; we ignore here the (-1)^|j| in the rhs!
                    thetah = np.linalg.solve(H + lda_tmp * np.eye(len(H)), h_train)

                    # Compute CV score
                    term1 = np.dot(thetah, np.dot(H, thetah))
                    term2 = np.dot(thetah, h_test)

                    if np.any(np.isnan(term1)) or np.any(np.isnan(term2)):
                        raise Exception("Score is nan!")

                    # We ignore here the (-1)^|j| in term2!
                    score_cv[sigma_index, lda_index, kk] = term1 - 2 * term2
                # end for lda
            # end for kk
        # end for sigma

        score_cv_mu = np.mean(score_cv, axis=2)
        (sigma_idx_chosen, lda_idx_chosen) = np.unravel_index(
            np.argmin(score_cv_mu), score_cv_mu.shape)
        sigma[dd] = sigma_list[sigma_idx_chosen]
        lda[dd] = lda_list[lda_idx_chosen]
        print("dd={:g}, sigma={:g}, lambda={:g}".format(dd, sigma[dd], lda[dd]))
    # end for dd

    # Save final theta vectors
    theta = np.empty((dim, bnum))

    for dd in range(dim):
        H = (np.sqrt(np.pi) * sigma[dd])**dim * np.exp(-CC_dist / (4 * sigma[dd]**2))
        GauKer = np.exp(-CX_dist / (2. * sigma[dd]**2))
        phi = CX_diff[:, :, dd] * GauKer / sigma[dd]**2
        h = np.mean(phi, axis=1)

        # Compute final theta; remember the (-1)^|j| in the rhs!
        theta[dd] = np.linalg.solve(H + lda[dd] * np.eye(len(H)), -h)
    # end for dd

    def density_deriv(T):
        # Check shape of T
        if len(T.shape) == 1:
            # Convert array format to correct shape
            T = T[np.newaxis, :]
        else:
            T = T.transpose()

        (__, Tsamples) = T.shape

        # Compute required matrices
        CT_diff = np.tile(C.T[:, np.newaxis], (1, Tsamples, 1)) - np.tile(T.T[np.newaxis, :], (bnum, 1, 1))
        CT_dist = np.sum(CT_diff ** 2, axis=2)

        # Compute density derivative
        dderivh = np.empty((dim, Tsamples))

        for dd in range(dim):
            # Eval T
            dderivh[dd] = np.dot(theta[dd], np.exp(-CT_dist / (2. * sigma[dd]**2)))
        # end for dd
        return dderivh  # (dim, n)

    kernel_info = KernelInfo(
        kernel_type="Gaussian", kernel_num=bnum, sigma=sigma, centers=C)
    result = DensityDerivative(
        method="MISED", theta=theta, lambda_=lda, kernel_info=kernel_info,
        compute_density_deriv=density_deriv)

    return result
