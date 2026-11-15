import numpy as np
import pennylane as qml
import pymatching
import torch

import surface_code as sc

all_syn = sc.idx_to_syn(np.arange(sc.n_syn))


def ml_answers(p):
    return sc.joint_probs(p).argmax(axis=1)


def mwpm_answers():
    z_finder = pymatching.Matching(sc.H_x)
    x_finder = pymatching.Matching(sc.H_z)
    syn_xchk = all_syn[:, :4]
    syn_zchk = all_syn[:, 4:]
    z_fix = z_finder.decode_batch(syn_xchk).astype(np.uint8)
    x_fix = x_finder.decode_batch(syn_zchk).astype(np.uint8)
    return sc.fix_to_cls(all_syn, x_fix, z_fix)


def count_samples(p, n, rng):
    x_err, z_err = sc.sample_depol(p, n, rng)
    syn, cls = sc.label(x_err, z_err)
    counts = np.zeros((sc.n_syn, 4))
    np.add.at(counts, (syn, cls), 1)
    return counts


def train(model, counts, steps, lr):
    seen = counts.sum(axis=1) > 0
    x_seen = torch.tensor(all_syn[seen], dtype=torch.float64)
    wts = torch.tensor(counts[seen] / counts.sum(), dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logp = model.log_probs(x_seen)
        loss = -(wts * logp).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x_all = torch.tensor(all_syn, dtype=torch.float64)
        table = model.log_probs(x_all).argmax(dim=1).numpy()
    return table, loss.item()


class MLP(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        layers, width = [], sc.n_bits
        for h in hidden:
            layers += [torch.nn.Linear(width, h), torch.nn.ReLU()]
            width = h
        layers.append(torch.nn.Linear(width, 4))
        self.net = torch.nn.Sequential(*layers).double()

    def log_probs(self, x):
        return torch.log_softmax(self.net(x), dim=1)


n_wires = sc.n_bits
dev = qml.device("default.qubit", wires=n_wires)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def vqc_circuit(syn, theta):
    for q in range(n_wires):
        qml.RX(np.pi * syn[:, q], wires=q)
    for layer in theta:
        for q in range(n_wires):
            qml.Rot(*layer[q], wires=q)
        for q in range(n_wires):
            qml.CNOT(wires=[q, (q + 1) % n_wires])
    return [qml.expval(qml.PauliZ(q)) for q in range(n_wires)]


class VQC(torch.nn.Module):
    def __init__(self, n_layers, rng):
        super().__init__()
        theta0 = rng.uniform(0, 2 * np.pi, size=(n_layers, n_wires, 3))
        self.theta = torch.nn.Parameter(torch.tensor(theta0, dtype=torch.float64))
        self.head = torch.nn.Linear(n_wires, 4).double()

    def log_probs(self, x):
        z_vals = torch.stack(vqc_circuit(x, self.theta), dim=1)
        return torch.log_softmax(self.head(z_vals), dim=1)


def n_params(model):
    return sum(t.numel() for t in model.parameters())
