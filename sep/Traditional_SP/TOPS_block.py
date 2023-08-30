from .MUSIC_block import MUSIC
import numpy as np
from scipy.linalg import svdvals

from scipy import linalg

class TOPS(MUSIC):
    """
    Class to apply Test of Orthogonality of Projected Subspaces [TOPS]_ for
    Direction of Arrival (DoA) estimation.
    .. note:: Run locate_source() to apply the TOPS algorithm.
    Parameters
    ----------
    L: numpy array
        Microphone array positions. Each column should correspond to the
        cartesian coordinates of a single microphone.
    fs: float
        Sampling frequency.
    nfft: int
        FFT length.
    c: float
        Speed of sound. Default: 343 m/s
    num_src: int
        Number of sources to detect. Default: 1
    mode: str
        'far' or 'near' for far-field or near-field detection
        respectively. Default: 'far'
    r: numpy array
        Candidate distances from the origin. Default: np.ones(1)
    azimuth: numpy array
        Candidate azimuth angles (in radians) with respect to x-axis.
        Default: np.linspace(-180.,180.,30)*np.pi/180
    colatitude: numpy array
        Candidate elevation angles (in radians) with respect to z-axis.
        Default is x-y plane search: np.pi/2*np.ones(1)
    References
    ----------
    .. [TOPS] Y. Yeo-Sun, L. M. Kaplan, J. H. McClellan, *TOPS: New DOA estimator for wideband
        signals*, IEEE Trans. Signal Process., Vol. 54, Num 6., pp 1977--1989, 2006
    """

    def __init__(
        self, mic_pos, grids_pos, freq_bins, mod_vecotr, nfft, c, fs
    ):

        MUSIC.__init__(
            self, freq_bins, mod_vecotr
        )

        self.mic_pos = np.copy(mic_pos)
        mic_center = np.mean(self.mic_pos, axis = 0)
        self.mic_pos = self.mic_pos - mic_center
        self.grids_pos = np.copy(grids_pos) - mic_center


        self.nfft = nfft
        self.c = c
        self.fs = fs
        self.n_points = self.mode_vec.shape[-1]
        print(self.mode_vec.shape)

    def _process(self, X):
        """
        Perform TOPS for a given frame in order to estimate steered response
        spectrum.
        """
        # In: X -  (M, F, Frames)
        # need more than 1 frequency band
        self.num_snap = X.shape[2] #Frames
        self.M = X.shape[0]

        if self.num_freq < 2:
            raise ValueError("Need more than one frequency band!")

        # select reference frequency (largest power)
        max_bin = np.argmax(
            np.sum(np.sum(abs(X[:, self.freq_bins, :]), axis=0), axis=1)
        )
        f0 = self.freq_bins[max_bin]
        freq = list(self.freq_bins)
        freq.remove(f0)

        # compute empirical cross correlation matrices
        C_hat = self._compute_correlation_matrices(X, num_snap = self.num_snap)

        # compute signal and noise subspace for each frequency band
        F = np.zeros((self.num_freq, self.M, self.num_src), dtype="complex64")
        W = np.zeros((self.num_freq, self.M, self.M - self.num_src), dtype="complex64")
        for k in range(self.num_freq):
            # subspace decomposition
            F[k, :, :], W[k, :, :], ws, wn = self._subspace_decomposition(
                C_hat[k, :, :]
            )

        # create transformation matrix
        f = (
            1.0
            / self.nfft
            / self.c
            * 1j
            * 2
            * np.pi
            * self.fs
            * (np.linspace(0, self.nfft // 2, self.nfft // 2 + 1) - f0)
        )

        Phi = np.zeros((len(f), self.M, self.n_points), dtype="complex64")

        for n in range(self.n_points):
            p_s = self.grids_pos[n, :]
            path_s = np.linalg.norm(p_s)

            for m in range(self.M):
                pos_m = self.mic_pos[m, :]
                delta_path = path_s - np.linalg.norm(p_s - pos_m)
                Phi[:, m, n] = np.exp(f * delta_path)

        # determine direction using rank test
        values = np.zeros((self.n_points, ))
        for n in range(self.n_points):
            # form D matrix
            D = np.zeros(
                (self.num_src, (self.M - self.num_src) * (self.num_freq - 1)),
                dtype="complex64",
            )
            for k in range(self.num_freq - 1):
                Uk = np.conjugate(np.dot(np.diag(Phi[k, :, n]), F[max_bin, :, :])).T
                # F[max_bin,:,:])).T
                idx = range(
                    k * (self.M - self.num_src), (k + 1) * (self.M - self.num_src)
                )
                D[:, idx] = np.dot(Uk, W[k, :, :])
            s = svdvals(D)
            values[n] =  1.0 / s[-1]

        return values
