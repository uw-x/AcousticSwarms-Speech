import numpy as np 




class MUSIC(object):
    def __init__(self, freq_bins, mod_vecotr):
        self.num_freq = freq_bins.shape[0]
        self.freq_bins = freq_bins
        self.mode_vec = mod_vecotr
        assert self.mode_vec.shape[0] == self.num_freq
        self.n_points = mod_vecotr.shape[-1]
        self.num_src = 3
        self.frequency_normalization = True

    def MUSIC_process(self, X):
        """
        Perform MUSIC for given frame in order to estimate steered response
        spectrum.
        """
        # compute steered response
        # X -  (M, F, Frames)
        self.M = X.shape[0]


        self.Pssl = np.zeros((self.num_freq, self.n_points))
        C_hat = self._compute_correlation_matricesvec(X)
        # subspace decomposition
        Es, En, ws, wn = self._subspace_decomposition(C_hat[None, ...])
        # compute spatial spectrum
        identity = np.zeros((self.num_freq, self.M, self.M))
        identity[:, list(np.arange(self.M)), list(np.arange(self.M))] = 1 # (F_selected, M) 


        # ES - (1, F_selected, M, num_src), after moveaxis is (1, F_selected, num_src, M)
        cross = identity - np.matmul(Es, np.moveaxis(np.conjugate(Es), -1, -2)) # after matual is  (1, F_selected, M, M)
        
        self.Pssl = self._compute_spatial_spectrumvec(cross)
        # out: Pssl : num_grids, F_selected

        if self.frequency_normalization:
            self._apply_frequency_normalization()


        result = np.squeeze(np.sum(self.Pssl, axis=1) / self.num_freq)
        return result

    def _apply_frequency_normalization(self):
        """
        Normalize the MUSIC pseudo-spectrum per frequency bin
        """
        self.Pssl = self.Pssl / np.max(self.Pssl, axis=0, keepdims=True)

    def _compute_spatial_spectrumvec(self, cross):
        ## In:  (1, F_selected, M, M)

        mod_vec = np.transpose(
            self.mode_vec, axes=[2, 0, 1]
        )

        # mode_vec - (num_grids, F_selected, M)

        # timeframe, frequ, no idea
        denom = np.matmul(
            np.conjugate(mod_vec[..., None, :]), np.matmul(cross, mod_vec[..., None])
        )

        ### matual: (num_grids, F_selected, 1, M), (1, F_selected, M, M), (num_grids, F_selected, M, 1)
        ## result - (num_grids, F_selected, 1, 1)
        return 1.0 / abs(denom[..., 0, 0])

    # def _compute_spatial_spectrum(self, cross, k):

    #     P = np.zeros(self.n_points)

    #     for n in range(self.n_points):
    #         Dc = np.array(self.mode_vec[k, :, n], ndmin=2).T
    #         Dc_H = np.conjugate(np.array(self.mode_vec[k, :, n], ndmin=2))
    #         denom = np.dot(np.dot(Dc_H, cross), Dc)
    #         P[n] = 1 / abs(denom)

    #     return P

    # # non-vectorized version
    def _compute_correlation_matrices(self, X, num_snap):
        C_hat = np.zeros([self.num_freq, self.M, self.M], dtype=complex)
        for i in range(self.num_freq):
            k = self.freq_bins[i]
            for s in range(num_snap):
                C_hat[i, :, :] = C_hat[i, :, :] + np.outer(
                    X[:, k, s], np.conjugate(X[:, k, s])
                )
        return C_hat / num_snap

    # vectorized version
    def _compute_correlation_matricesvec(self, X):
        # change X such that time frames, frequency microphones is the result
        # In: X -  (M, F_all, Frames)


        X = np.transpose(X, axes=[2, 1, 0]) # X = Frames, F_all, M
        # select frequency bins
        X = X[..., list(self.freq_bins), :] # X = Frames, F_selected, M
        # Compute PSD and average over time frame
        C_hat = np.matmul(X[..., None], np.conjugate(X[..., None, :]))
        # Average over time-frames
        C_hat = np.mean(C_hat, axis=0)

        ## OUt: C_hat - (F_selected, M, M)
        return C_hat

    # vectorized versino
    def _subspace_decomposition(self, R):
        # eigenvalue decomposition!
        # This method is specialized for Hermitian symmetric matrices,
        # which is the case since R is a covariance matrix
        #In: R - (1, F_selected, M, M)

        w, v = np.linalg.eigh(R)

        # This method (numpy.linalg.eigh) returns the eigenvalues (and
        # eigenvectors) in ascending order, so there is no need to sort Signal
        # comprises the leading eigenvalues Noise takes the rest
 
        Es = v[..., -self.num_src :] ## signal space (1, F_selected, M, self.num_src)
        ws = w[..., -self.num_src :] ## signal space (1, F_selected, self.num_src)
        En = v[..., : -self.num_src] ## noise space (1, F_selected, M, M - self.num_src)
        wn = w[..., : -self.num_src] ## noise space (1, F_selected, M - self.num_src)

        return (Es, En, ws, wn)