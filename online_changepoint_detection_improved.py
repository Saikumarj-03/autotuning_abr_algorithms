import numpy as np
import ruptures as rpt
from scipy import stats

def offline_changepoint_detection_pelt(data, penalty=10):
    """
    Offline changepoint detection using the PELT algorithm from ruptures.
    
    Parameters:
        data (list or np.array): 1D array of observed values (e.g., bandwidth)
        penalty (int or float): Penalty value for number of changepoints
    
    Returns:
        Tuple (bool, int):
            bool: Whether a changepoint was detected
            int: Index of the last changepoint (or -1 if none)
    """
    if len(data) < 5:
        return False, -1  # Too short to detect changepoints

    model = "l2"  # Suitable for numeric sequences like bandwidth
    algo = rpt.Pelt(model=model).fit(data)
    result = algo.predict(pen=penalty)

    # Remove last changepoint if it's at the end
    changepoints = [r for r in result if r != len(data)]

    if not changepoints:
        return False, -1

    return True, changepoints[-1]


# -------------------------------
# Legacy Bayesian interface for reference only
# -------------------------------

def online_changepoint_detection(data, hazard_func, observation_likelihood):
    maxes = np.zeros(len(data) + 1)
    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1
    
    for t, x in enumerate(data):
        predprobs = observation_likelihood.pdf(x)
        H = hazard_func(np.array(list(range(t+1))))
        
        R[1:t+2, t+1] = R[0:t+1, t] * predprobs * (1-H)
        R[0, t+1] = np.sum(R[0:t+1, t] * predprobs * H)
        R[:, t+1] = R[:, t+1] / np.sum(R[:, t+1])
        
        observation_likelihood.update_theta(x)
        maxes[t] = R[:, t].argmax()
        
    return R, maxes


def constant_hazard(lam, r):
    return 1/lam * np.ones(r.shape)


class StudentT:
    def __init__(self, alpha, beta, kappa, mu):
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data):
        return stats.t.pdf(
            x=data,
            df=2 * self.alpha,
            loc=self.mu,
            scale=np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        )

    def update_theta(self, data):
        muT0 = np.concatenate((self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate((
            self.beta0,
            self.beta + (self.kappa * (data - self.mu) ** 2) / (2. * (self.kappa + 1.))
        ))

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0
