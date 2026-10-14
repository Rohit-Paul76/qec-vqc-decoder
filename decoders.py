"""The four decoders. Each one ends up as a table: syndrome index (0..255) -> class (0..3).

- best possible (maximum likelihood): reads the exact P(s, c) and picks the likeliest class;
- MWPM: pymatching, X and Z parts decoded separately;
- MLP: a small PyTorch network trained on sampled (syndrome, class) pairs
  (MLP([]) is logistic regression, used as the "Linear" control);
- VQC: an 8-qubit PennyLane circuit plus a linear layer, trained on the same samples.

There are only 256 syndromes, so a trained network is fully described by its answer on
each of them. That is what lets run.py compute every logical error rate exactly.
"""
import numpy as np
import pennylane as qml
import pymatching
import torch

import surface_code as sc

ALL_SYNDROMES = sc.index_to_bits(np.arange(sc.N_SYNDROMES))   # (256, 8)


# ---- Best possible ---------------------------------------------------------------

def ml_table(p):
    """For each syndrome, the class with the highest probability at noise level p.

    No decoder that sees only the syndrome can do better: on syndrome s it is right with
    probability P(s, guess), and this picks the guess that maximises it.
    """
    return sc.joint_probs(p).argmax(axis=1)


# ---- MWPM (pymatching) -----------------------------------------------------------

def mwpm_table():
    """Minimum-weight matching, run once per syndrome.

    The X-type checks see only Z parts and the Z-type checks only X parts, so pymatching
    gets two separate problems: find the fewest Z flips that explain the X-check bits, and
    the fewest X flips that explain the Z-check bits. A Y error shows up in both and is
    counted as two flips, which is where MWPM loses to the best decoder under
    depolarizing noise.

    All qubits have the same error probability, so equal edge weights; the answer does not
    depend on p.
    """
    match_z = pymatching.Matching(sc.HX)   # columns = qubits, rows = X-type checks
    match_x = pymatching.Matching(sc.HZ)
    s_x = ALL_SYNDROMES[:, :4]
    s_z = ALL_SYNDROMES[:, 4:]
    cor_z = match_z.decode_batch(s_x).astype(np.uint8)   # (256, 9)
    cor_x = match_x.decode_batch(s_z).astype(np.uint8)
    return sc.correction_to_class(ALL_SYNDROMES, cor_x, cor_z)


# ---- Training data for the learned decoders --------------------------------------

def training_counts(p, n, rng):
    """Sample n errors at noise level p and count them: counts[s, c]. Shape (256, 4).

    Many samples share a syndrome, so the training loss over n samples equals a weighted
    loss over the 256 syndromes with these counts as weights (see train()).
    """
    ex, ez = sc.sample_depolarizing(p, n, rng)
    s_idx, cls = sc.label(ex, ez)
    counts = np.zeros((sc.N_SYNDROMES, 4))
    np.add.at(counts, (s_idx, cls), 1)
    return counts


def train(model, counts, steps, lr):
    """Full-batch Adam on the average cross-entropy over all training samples.

    loss = - (1/n) * sum over samples of log q(true class | syndrome)
         = - (1/n) * sum over s, c of counts[s, c] * log q(c | s)
    where q(c | s) is the model's probability for class c. The second line is the same
    number, computed with at most 256 model evaluations instead of n. Syndromes that never
    appear in the sample have zero weight, so they are left out of training (the model
    still gives them an answer at the end).
    """
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


# ---- MLP --------------------------------------------------------------------------

class MLP(torch.nn.Module):
    """8 syndrome bits -> hidden ReLU layers -> 4 class scores -> log-probabilities.

    MLP([]) has no hidden layer: that is logistic regression (the "Linear" decoder).
    """

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


# ---- VQC --------------------------------------------------------------------------

N_WIRES = sc.N_SYNDROME_BITS   # one qubit per syndrome bit
_dev = qml.device("default.qubit", wires=N_WIRES)


@qml.qnode(_dev, interface="torch", diff_method="backprop")
def vqc_circuit(s, weights):
    """Basis encoding, then layers of Rot + a CNOT ring, then <Z> on every qubit.

    s: (batch, 8) syndrome bits. RX(pi) sends |0> to -i|1>, so RX(pi * s_q) on each qubit
    prepares the basis state |s> (up to a global phase that no measurement sees). It is
    used instead of qml.BasisState because it accepts a whole batch at once.
    weights: (layers, 8, 3), the three angles of each Rot.
    Output: 8 expectation values <Z_q>, each in [-1, 1], one tensor of shape (batch,) per qubit.
    """
    for q in range(N_WIRES):
        qml.RX(np.pi * s[:, q], wires=q)
    for layer in weights:
        for q in range(N_WIRES):
            qml.Rot(*layer[q], wires=q)
        for q in range(N_WIRES):
            qml.CNOT(wires=[q, (q + 1) % N_WIRES])
    return [qml.expval(qml.PauliZ(q)) for q in range(N_WIRES)]


class VQC(torch.nn.Module):
    """Same ansatz as the parity classifier (Rot + CNOT ring), on 8 qubits, then a linear
    layer from the 8 <Z> values to 4 class scores.

    The parity classifier added a trainable bias to one <Z>; this is the 4-class version
    of that classical step. If every Rot were the identity, <Z_q> = 1 - 2 s_q and the
    model would be plain logistic regression on the syndrome bits (the "Linear" decoder),
    so anything it does better than Linear comes from the circuit.
    """

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
