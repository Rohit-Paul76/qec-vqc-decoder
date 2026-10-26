"""The four decoders. Each one ends up as a table: syndrome index (0..255) -> class (0..3)."""
import numpy as np
import pennylane as qml
import pymatching
import torch

import surface_code as sc

ALL_SYNDROMES = sc.index_to_bits(np.arange(sc.N_SYNDROMES))   # (256, 8)



def ml_table(p):
    """For each syndrome, the class with the highest probability at noise level p."""
    return sc.joint_probs(p).argmax(axis=1)



def mwpm_table():
    """Minimum-weight matching, run once per syndrome."""
    match_z = pymatching.Matching(sc.HX)   # columns = qubits, rows = X-type checks
    match_x = pymatching.Matching(sc.HZ)
    s_x = ALL_SYNDROMES[:, :4]
    s_z = ALL_SYNDROMES[:, 4:]
    cor_z = match_z.decode_batch(s_x).astype(np.uint8)   # (256, 9)
    cor_x = match_x.decode_batch(s_z).astype(np.uint8)
    return sc.correction_to_class(ALL_SYNDROMES, cor_x, cor_z)



def training_counts(p, n, rng):
    """Sample n errors at noise level p and count them: counts[s, c]. Shape (256, 4)."""
    ex, ez = sc.sample_depolarizing(p, n, rng)
    s_idx, cls = sc.label(ex, ez)
    counts = np.zeros((sc.N_SYNDROMES, 4))
    np.add.at(counts, (s_idx, cls), 1)
    return counts


def train(model, counts, steps, lr):
    """Full-batch Adam on the average cross-entropy over all training samples."""
    seen = counts.sum(axis=1) > 0
    x_seen = torch.tensor(ALL_SYNDROMES[seen], dtype=torch.float64)
    w = torch.tensor(counts[seen] / counts.sum(), dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        log_q = model.log_probs(x_seen)     # (number of seen syndromes, 4)
        loss = -(w * log_q).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x_all = torch.tensor(ALL_SYNDROMES, dtype=torch.float64)
        table = model.log_probs(x_all).argmax(dim=1).numpy()
    return table, loss.item()



class MLP(torch.nn.Module):
    """8 syndrome bits -> hidden ReLU layers -> 4 class scores -> log-probabilities."""

    def __init__(self, hidden):
        super().__init__()
        layers, width = [], sc.N_SYNDROME_BITS
        for h in hidden:
            layers += [torch.nn.Linear(width, h), torch.nn.ReLU()]
            width = h
        layers.append(torch.nn.Linear(width, 4))
        self.net = torch.nn.Sequential(*layers).double()

    def log_probs(self, x):
        return torch.log_softmax(self.net(x), dim=1)



N_WIRES = sc.N_SYNDROME_BITS   # one qubit per syndrome bit
_dev = qml.device("default.qubit", wires=N_WIRES)


@qml.qnode(_dev, interface="torch", diff_method="backprop")
def vqc_circuit(s, weights):
    """Basis encoding, then layers of Rot + a CNOT ring, then <Z> on every qubit."""
    for q in range(N_WIRES):
        qml.RX(np.pi * s[:, q], wires=q)
    for layer in weights:
        for q in range(N_WIRES):
            qml.Rot(*layer[q], wires=q)
        for q in range(N_WIRES):
            qml.CNOT(wires=[q, (q + 1) % N_WIRES])
    return [qml.expval(qml.PauliZ(q)) for q in range(N_WIRES)]


class VQC(torch.nn.Module):
    """Rot + CNOT ring on 8 qubits, then a linear layer from the 8 <Z> values to 4 classes."""

    def __init__(self, n_layers, rng):
        super().__init__()
        init = rng.uniform(0, 2 * np.pi, size=(n_layers, N_WIRES, 3))
        self.weights = torch.nn.Parameter(torch.tensor(init, dtype=torch.float64))
        self.head = torch.nn.Linear(N_WIRES, 4).double()

    def log_probs(self, x):
        z = torch.stack(vqc_circuit(x, self.weights), dim=1)   # (batch, 8)
        return torch.log_softmax(self.head(z), dim=1)


def n_params(model):
    return sum(t.numel() for t in model.parameters())
