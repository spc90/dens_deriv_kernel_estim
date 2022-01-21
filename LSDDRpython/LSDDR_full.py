import numpy as np
from .density_derivative_ratio import DensityDerivativeRatio, KernelInfo


def LSDDR_full(X, bnum=100, sigma_list=None, lda_list=None, cvfold=5):
    '''
    Least-squares density-derivative ratio

    Estimates (dp(x) / dx_j) / p(x) for all dimensions j
    from samples {x_i}_{i=1}^{n}
    '''

    # Convert array format to correct shape
    if len(X.shape) == 1:
        X = X[np.newaxis, :]
    else:
        X = X.transpose()

    (dim, samples) = X.shape

    bnum = min(samples, 100 if bnum is None else bnum)
    if sigma_list is None:
        sigma_list = np.linspace(0.5, 5, 10)
    if lda_list is None:
        lda_list = np.logspace(-3, 0, 10)

    # We only work with a number of bnum centers instead of samples
    cind = np.random.permutation(samples)[0:bnum]
    C = X[:, cind]

    # We scale the sigma values with the median inter-value distance
    MedX = MedianDiffDim(X)

    # Difference to centers (size: bnum x samples x dim)
    CX_diff = np.tile(C.T[:, np.newaxis], (1, samples, 1)) - np.tile(X.T[np.newaxis, :], (bnum, 1, 1))

    # Distance from centers (size: bnum x samples)
    CX_dist = np.sum(CX_diff ** 2, axis=2)

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
            # Select sigma and create kernel
            sigma_tmp *= MedX[dd]
            GauKer = np.exp(-CX_dist / (2. * sigma_tmp**2))

            for kk in cv_fold:
                # Create the parts of the (2*bnum x 2*bnum) linear system
                psi_train_a = GauKer[:, cv_index != kk]
                psi_train_b = CX_diff[:, cv_index != kk, dd] * GauKer[:, cv_index != kk] / sigma_tmp**2

                # Create the parts of the 2*bnum rhs; multiplied already with the (-1)^|j|
                phi_train_a = -CX_diff[:, cv_index != kk, dd] * GauKer[:, cv_index != kk] / sigma_tmp**2
                phi_train_b = (1. / sigma_tmp**2 - CX_diff[:, cv_index != kk, dd]**2 / sigma_tmp**4) * GauKer[:, cv_index != kk]

                # Create the system matrix blocks (each of size: bnum_t x bnum_t)
                K_train_aa = np.matmul(psi_train_a, psi_train_a.T) / psi_train_a.shape[1]
                K_train_ab = np.matmul(psi_train_a, psi_train_b.T) / psi_train_a.shape[1]
                K_train_ba = np.matmul(psi_train_b, psi_train_a.T) / psi_train_b.shape[1]
                K_train_bb = np.matmul(psi_train_b, psi_train_b.T) / psi_train_b.shape[1]

                # Create the rhs matrix blocks (each of size: bnum_t x 1)
                h_train_a = np.mean(phi_train_a, axis=1)
                h_train_b = np.mean(phi_train_b, axis=1)

                # Stitch the blocks together
                K_train = np.block([[K_train_aa, K_train_ab], [K_train_ba, K_train_bb]])
                h_train = np.block([h_train_a, h_train_b])  # 1D vector

                # Repeat procedure for test data
                psi_test_a = GauKer[:, cv_index == kk]
                psi_test_b = CX_diff[:, cv_index == kk, dd] * GauKer[:, cv_index == kk] / sigma_tmp**2

                phi_test_a = -CX_diff[:, cv_index == kk, dd] * GauKer[:, cv_index == kk] / sigma_tmp**2
                phi_test_b = (1. / sigma_tmp**2 - CX_diff[:, cv_index == kk, dd]**2 / sigma_tmp**4) * GauKer[:, cv_index == kk]

                K_test_aa = np.matmul(psi_test_a, psi_test_a.T) / psi_test_a.shape[1]
                K_test_ab = np.matmul(psi_test_a, psi_test_b.T) / psi_test_a.shape[1]
                K_test_ba = np.matmul(psi_test_b, psi_test_a.T) / psi_test_b.shape[1]
                K_test_bb = np.matmul(psi_test_b, psi_test_b.T) / psi_test_b.shape[1]

                h_test_a = np.mean(phi_test_a, axis=1)
                h_test_b = np.mean(phi_test_b, axis=1)

                K_test = np.block([[K_test_aa, K_test_ab], [K_test_ba, K_test_bb]])
                h_test = np.block([h_test_a, h_test_b])  # 1D vector

                for lda_index, lda_tmp in enumerate(lda_list):
                    # Solve system for train data; we ignore here the (-1)^|j| in the rhs!
                    thetah = np.linalg.solve(K_train + lda_tmp * np.eye(len(K_train)), h_train)

                    # Compute CV score
                    term1 = np.dot(thetah, np.dot(K_test, thetah))
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
        sigma[dd] = MedX[dd] * sigma_list[sigma_idx_chosen]
        lda[dd] = lda_list[lda_idx_chosen]
        print("dd={:g}, sigma={:g}, lambda={:g}".format(dd, sigma[dd], lda[dd]))
    # end for dd

    # Save final theta vectors & save T eval values
    theta = np.empty((dim, 2*bnum))

    for dd in range(dim):
        GauKer = np.exp(-CX_dist / (2. * sigma[dd]**2))

        # psi sub-vectors
        psi_a = GauKer  # .copy() is not necessary
        psi_b = CX_diff[:, :, dd] * GauKer / sigma[dd]**2

        # phi sub-vectors; multiplied already with the (-1)^|j|
        phi_a = -CX_diff[:, :, dd] * GauKer / sigma[dd]**2
        phi_b = (1. / sigma[dd]**2 - CX_diff[:, :, dd]**2 / sigma[dd]**4) * GauKer

        K_aa = np.matmul(psi_a, psi_a.T) / psi_a.shape[1]
        K_ab = np.matmul(psi_a, psi_b.T) / psi_a.shape[1]
        K_ba = np.matmul(psi_b, psi_a.T) / psi_b.shape[1]
        K_bb = np.matmul(psi_b, psi_b.T) / psi_b.shape[1]

        h_a = np.mean(phi_a, axis=1)
        h_b = np.mean(phi_b, axis=1)

        K = np.block([[K_aa, K_ab], [K_ba, K_bb]])
        h = np.block([h_a, h_b])  # 1D vector

        # Compute final theta; remember the (-1)^|j| in the rhs is already included!
        theta[dd] = np.linalg.solve(K + lda[dd] * np.eye(len(K)), h)
    # end for dd

    def density_deriv_ratio(T):
        # Check if T is given; if not, evaluate at X
        if T is None:
            CT_diff = CX_diff
            CT_dist = CX_dist
        else:
            if len(T.shape) == 1:
                # Convert array format to correct shape
                T = T[np.newaxis, :]
            else:
                T = T.transpose()

            (_, Tsamples) = T.shape

            # Compute required matrices
            CT_diff = np.tile(
                C.T[:, np.newaxis], (1, Tsamples, 1)) - np.tile(
                T.T[np.newaxis, :], (bnum, 1, 1))
            CT_dist = np.sum(CT_diff ** 2, axis=2)

        # Compute density derivative ratio
        ddrh = np.empty((dim, Tsamples))

        for dd in range(dim):
            # Prepare eval
            TGauKer = np.exp(-CT_dist / (2. * sigma[dd]**2))

            psi_a = TGauKer  # .copy() is not necessary
            psi_b = CT_diff[:, :, dd] * TGauKer / sigma[dd]**2

            ddrh[dd] = np.dot(theta[dd], np.block([[psi_a], [psi_b]]))
        # end for dd
        return ddrh  # (dim, n)

    kernel_info = KernelInfo(
        kernel_type="Gaussian", kernel_num=bnum, sigma=sigma, centers=C)
    result = DensityDerivativeRatio(
        method="LSDDR", theta=theta, lambda_=lda, kernel_info=kernel_info,
        compute_density_deriv_ratio=density_deriv_ratio)

    return result


def MedianDiffDim(X):
    (dim, n) = X.shape
    XX_diff = np.tile(
        X.T[:, np.newaxis], (1, n, 1)) - np.tile(
        X.T[np.newaxis, :], (n, 1, 1))
    return np.median(np.reshape(np.abs(XX_diff), (n**2, dim)), axis=0)
