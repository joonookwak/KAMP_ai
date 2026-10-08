"""Actual PyTorch LSTM/GRU next-sample forecasters, normal-only fitting."""
import numpy as np


class Recurrent:
    cell = 'LSTM'

    def __init__(self,seed=0):
        self.seed=seed

    def fit(self, x, y, vx, vy):
        import torch
        from torch import nn
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        rng = np.random.default_rng(self.seed)

        class Net(nn.Module):
            def __init__(net, cell):
                super().__init__()
                net.rnn = getattr(nn, cell)(3, 32, num_layers=1, batch_first=True,
                                            bidirectional=False)
                net.head = nn.Linear(32, 3)

            def forward(net, values):
                sequence, _ = net.rnn(values)
                return net.head(sequence[:, -1])

        self.net = Net(self.cell)
        tx = torch.tensor(x.reshape(len(x), -1, 3), dtype=torch.float32)
        ty = torch.tensor(y, dtype=torch.float32)
        tvx = torch.tensor(vx.reshape(len(vx), -1, 3), dtype=torch.float32)
        tvy = torch.tensor(vy, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=.001, weight_decay=1e-4)
        best_mse, stale = float('inf'), 0
        history = []
        for epoch in range(1, 151):
            self.net.train()
            order = rng.permutation(len(x))
            for start in range(0, len(order), 128):
                ids = order[start:start+128]
                optimizer.zero_grad()
                loss = ((self.net(tx[ids])-ty[ids])**2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                optimizer.step()
            self.net.eval()
            with torch.no_grad():
                mse = float(((self.net(tvx)-tvy)**2).mean())
            if not np.isfinite(mse):
                raise ValueError('Nonfinite recurrent validation loss')
            history.append({'epoch': epoch, 'validation_mse': mse})
            if mse < best_mse-1e-6:
                best_mse, best_epoch, stale = mse, epoch, 0
                best = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
            if stale >= 15:
                break
        self.net.load_state_dict(best)
        self.net.eval()
        self.weights = [v.detach().numpy().copy() for v in self.net.state_dict().values()]
        return {'cell': self.cell, 'torch_version': torch.__version__, 'device': 'cpu',
                'seed': self.seed, 'hidden_units': 32, 'layers': 1, 'bidirectional': False,
                'learning_rate': .001, 'weight_decay': 1e-4, 'batch_size': 128,
                'gradient_clip': 1., 'max_epochs': 150, 'early_stopping_patience': 15,
                'best_epoch': best_epoch, 'epochs_run': epoch, 'validation_mse': best_mse,
                'state_dict_keys': list(self.net.state_dict()), 'history': history,
                'state_policy': 'zero hidden state per window; no carry across clips'}

    def predict(self, x):
        import torch
        self.net.eval()
        with torch.no_grad():
            tx = torch.tensor(x.reshape(len(x), -1, 3), dtype=torch.float32)
            return self.net(tx).numpy().astype(float)


class LSTM(Recurrent):
    cell = 'LSTM'


class GRU(Recurrent):
    cell = 'GRU'
