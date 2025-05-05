import numpy as np
import ruptures as rpt
from scipy import stats

def offline_changepoint_detection_pelt(signal_series, penalty=10):
    """
    Offline changepoint detection using the PELT algorithm from ruptures.

    Parameters:
        signal_series (list or np.array): 1D array of observed values (e.g., bandwidth)
        penalty_value (int or float): Penalty value for number of changepoints

    Returns:
        Tuple (bool, int):
            bool: Whether a changepoint was detected
            int: Index of the last changepoint (or -1 if none)
    """
    if len(signal_series) < 5:
        return False, -1  # Too short to detect changepoints

    detection_model = "l2"
    pelt_algo = rpt.Pelt(model=detection_model).fit(signal_series)
    change_predictions = pelt_algo.predict(pen=penalty_value)

    changepoint_indices = [cp for cp in change_predictions if cp != len(signal_series)]

    if not changepoint_indices:
        return False, -1

    return True, changepoint_indices[-1]


# -------------------------------
# Legacy Bayesian interface for reference only
# -------------------------------

def online_changepoint_detection(time_series, hazard_fn, obs_likelihood):
    changepoint_trace = np.zeros(len(time_series) + 1)
    run_matrix = np.zeros((len(time_series) + 1, len(time_series) + 1))
    run_matrix[0, 0] = 1

    for t_idx, value in enumerate(time_series):
        predictive_prob = obs_likelihood.pdf(value)
        hazard_vals = hazard_fn(np.arange(t_idx + 1))

        run_matrix[1:t_idx + 2, t_idx + 1] = run_matrix[0:t_idx + 1, t_idx] * predictive_prob * (1 - hazard_vals)
        run_matrix[0, t_idx + 1] = np.sum(run_matrix[0:t_idx + 1, t_idx] * predictive_prob * hazard_vals)
        run_matrix[:, t_idx + 1] /= np.sum(run_matrix[:, t_idx + 1])

        obs_likelihood.update_theta(value)
        changepoint_trace[t_idx] = run_matrix[:, t_idx].argmax()

    return run_matrix, changepoint_trace


def constant_hazard(lam_val, idx_array):
    return 1 / lam_val * np.ones(idx_array.shape)


class StudentT:
    def __init__(self, alpha_init, beta_init, kappa_init, mu_init):
        self.alpha0 = self.alpha = np.array([alpha_init])
        self.beta0 = self.beta = np.array([beta_init])
        self.kappa0 = self.kappa = np.array([kappa_init])
        self.mu0 = self.mu = np.array([mu_init])

    def pdf(self, observation):
        return stats.t.pdf(
            x=observation,
            df=2 * self.alpha,
            loc=self.mu,
            scale=np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        )

    def update_theta(self, observation):
        updated_mu = np.concatenate((self.mu0, (self.kappa * self.mu + observation) / (self.kappa + 1)))
        updated_kappa = np.concatenate((self.kappa0, self.kappa + 1.))
        updated_alpha = np.concatenate((self.alpha0, self.alpha + 0.5))
        updated_beta = np.concatenate((
            self.beta0,
            self.beta + (self.kappa * (observation - self.mu) ** 2) / (2. * (self.kappa + 1.))
        ))

        self.mu = updated_mu
        self.kappa = updated_kappa
        self.alpha = updated_alpha
        self.beta = updated_beta
