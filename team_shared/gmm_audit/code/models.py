"""Small forecasting baselines. No anomaly labels are supplied to these models."""
import numpy as np


class Persistence:
    def fit(self, x, y, vx, vy):
        return {"parameters": "predict the last observed three-channel value"}

    def predict(self, x):
        return x[:, -3:]


class Ridge:
    def fit(self, x, y, vx, vy):
        candidates = []
        a = np.c_[x, np.ones(len(x))]
        penalty = np.eye(a.shape[1]); penalty[-1, -1] = 0
        for strength in (0.01, 0.1, 1., 10., 100.):
            weights = np.linalg.solve(a.T @ a + strength * penalty, a.T @ y)
            mse = float(np.mean((np.c_[vx, np.ones(len(vx))] @ weights - vy) ** 2))
            candidates.append((mse, strength, weights))
        mse, strength, self.weights = min(candidates, key=lambda c: c[0])
        return {"ridge_alpha": strength, "validation_mse": mse}

    def predict(self, x):
        return np.c_[x, np.ones(len(x))] @ self.weights


class MLP:
    """One tanh hidden layer, linear three-channel output, Adam, early stopping."""
    def __init__(self, seed=0, hidden=32, max_epochs=150, patience=15):
        self.seed, self.hidden = seed, hidden
        self.max_epochs, self.patience = max_epochs, patience
        self.l2 = 1e-4

    def loss_and_gradient(self, x, y, weights):
        w1, b1, w2, b2 = weights
        h = np.tanh(x @ w1 + b1)
        pred = h @ w2 + b2
        delta = 2 * (pred - y) / y.size
        hidden_delta = (delta @ w2.T) * (1 - h ** 2)
        loss = np.mean((pred-y)**2) + self.l2 * (np.sum(w1*w1) + np.sum(w2*w2))
        grads = [x.T @ hidden_delta + 2*self.l2*w1, hidden_delta.sum(0),
                 h.T @ delta + 2*self.l2*w2, delta.sum(0)]
        return float(loss), grads

    def fit(self, x, y, vx, vy):
        rng = np.random.default_rng(self.seed)
        self.weights = [rng.normal(0, np.sqrt(2/(x.shape[1]+self.hidden)), (x.shape[1], self.hidden)),
                        np.zeros(self.hidden),
                        rng.normal(0, np.sqrt(2/(self.hidden+3)), (self.hidden, 3)), np.zeros(3)]
        moment = [np.zeros_like(w) for w in self.weights]
        variance = [np.zeros_like(w) for w in self.weights]
        best_mse, best_epoch, stale, step = np.inf, 0, 0, 0
        history = []
        for epoch in range(1, self.max_epochs+1):
            order = rng.permutation(len(x))
            for start in range(0, len(order), 128):
                idx = order[start:start+128]
                _, gradients = self.loss_and_gradient(x[idx], y[idx], self.weights)
                step += 1
                for j, gradient in enumerate(gradients):
                    moment[j] = .9*moment[j] + .1*gradient
                    variance[j] = .999*variance[j] + .001*gradient**2
                    self.weights[j] -= .001 * (moment[j]/(1-.9**step)) / (
                        np.sqrt(variance[j]/(1-.999**step))+1e-8)
            mse = float(np.mean((self.predict(vx)-vy)**2))
            if not np.isfinite(mse):
                raise ValueError("Non-finite MLP validation loss")
            history.append({"epoch": epoch, "validation_mse": mse})
            if mse < best_mse - 1e-6:
                best_mse, best_epoch, stale = mse, epoch, 0
                best = [w.copy() for w in self.weights]
            else:
                stale += 1
            if stale >= self.patience:
                break
        self.weights = best
        return {"seed": self.seed, "hidden_units": self.hidden, "activation": "tanh",
                "learning_rate": .001, "batch_size": 128, "l2": self.l2,
                "max_epochs": self.max_epochs, "early_stopping_patience": self.patience,
                "best_epoch": best_epoch, "epochs_run": epoch,
                "validation_mse": best_mse, "history": history}

    def predict(self, x):
        w1, b1, w2, b2 = self.weights
        return np.tanh(x @ w1 + b1) @ w2 + b2


class AR:
    """Separate autoregression for each sensor; no cross-sensor inputs."""
    def fit(self, x, y, vx, vy):
        self.regressors = []
        self.weights = []
        params = []
        for j in range(3):
            reg = Ridge()
            params.append(reg.fit(x[:, j::3], y[:, j:j+1], vx[:, j::3], vy[:, j:j+1]))
            self.regressors.append(reg)
            self.weights.append(reg.weights)
        return {"structure": "independent per-sensor AR", "channels": params}

    def predict(self, x):
        return np.concatenate([reg.predict(x[:, j::3]) for j, reg in enumerate(self.regressors)], axis=1)


class ARRelation:
    """Own-sensor AR plus causal nonlinear cross-sensor residual regression."""
    def features(self, x):
        p = self.ar.predict(x)
        last = x[:, -3:]
        # Predicted next current is available before the next sample arrives.
        # Quadratic terms permit nonlinear current/vibration relationships.
        z = np.c_[p, last, p[:, 2]**2, last[:, 2]**2,
                  p[:, 2]*last[:, 0], p[:, 2]*last[:, 1],
                  last[:, 0]*last[:, 1]]
        return z

    def fit(self, x, y, vx, vy):
        self.ar = AR()
        ar_params = self.ar.fit(x, y, vx, vy)
        z, vz = self.features(x), self.features(vx)
        self.feature_mean = z.mean(0)
        self.feature_scale = np.maximum(z.std(0), 1e-8)
        self.correction = Ridge()
        params = self.correction.fit((z-self.feature_mean)/self.feature_scale,
                                     y-self.ar.predict(x),
                                     (vz-self.feature_mean)/self.feature_scale,
                                     vy-self.ar.predict(vx))
        self.weights = self.ar.weights + [self.correction.weights,
                                         self.feature_mean, self.feature_scale]
        return {"structure": "AR + cross-sensor quadratic residual regression",
                "AR": ar_params, "correction": params,
                "next_current_input": "AR prediction only, never observed next current"}

    def predict(self, x):
        z = (self.features(x)-self.feature_mean)/self.feature_scale
        return self.ar.predict(x) + self.correction.predict(z)


from recurrent_models import LSTM, GRU

MODELS = {"Persistence": Persistence, "VAR_Ridge": Ridge, "MLP32": MLP,
          "AR": AR, "AR_Relation": ARRelation, "LSTM32": LSTM, "GRU32": GRU}
from ad_models import AD_MODELS
MODELS.update(AD_MODELS)
