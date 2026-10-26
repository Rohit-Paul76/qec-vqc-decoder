"""Distance-3 rotated surface code as bit arrays. Qubit q = 3 * row + col on a 3x3 grid;
an error is (ex, ez), its X and Z parts."""
import itertools

import numpy as np

N_QUBITS = 9

# X-type stabilizers detect Z errors
X_STABS = [
    [0, 1, 3, 4],  # top-left square
    [4, 5, 7, 8],  # bottom-right square
    [1, 2],        # top edge
    [6, 7],        # bottom edge
]
# Z-type stabilizers detect X errors
Z_STABS = [
    [1, 2, 4, 5],  # top-right square
    [3, 4, 6, 7],  # bottom-left square
    [0, 3],        # left edge
    [5, 8],        # right edge
]

LOGICAL_X = [0, 3, 6]
LOGICAL_Z = [0, 1, 2]


def support_matrix(stabs):
    """Rows of 0/1: row i has a 1 on every qubit stabilizer i acts on. Shape (len(stabs), 9)."""
    m = np.zeros((len(stabs), N_QUBITS), dtype=np.uint8)
    for i, qubits in enumerate(stabs):
        m[i, qubits] = 1
    return m


HX = support_matrix(X_STABS)              # (4, 9), detects ez
HZ = support_matrix(Z_STABS)              # (4, 9), detects ex
LX = support_matrix([LOGICAL_X])[0]       # (9,)
LZ = support_matrix([LOGICAL_Z])[0]       # (9,)

N_SYNDROME_BITS = len(X_STABS) + len(Z_STABS)   # 8
N_SYNDROMES = 2 ** N_SYNDROME_BITS              # 256

CLASS_NAMES = ["I", "X", "Z", "Y"]   # class index = lx + 2 * lz


def syndrome(ex, ez):
    """8 syndrome bits for errors given as arrays of shape (..., 9)."""
    s_x = (ez @ HX.T) % 2
    s_z = (ex @ HZ.T) % 2
    return np.concatenate([s_x, s_z], axis=-1).astype(np.uint8)


def syndrome_index(s):
    """Turn 8 syndrome bits into one integer 0..255 (bit 0 is the most significant)."""
    weights = 2 ** np.arange(N_SYNDROME_BITS - 1, -1, -1)
    return (s @ weights).astype(np.int64)


def index_to_bits(idx):
    """Inverse of syndrome_index. Returns shape (..., 8)."""
    idx = np.asarray(idx)
    shifts = np.arange(N_SYNDROME_BITS - 1, -1, -1)
    return ((idx[..., None] >> shifts) & 1).astype(np.uint8)


def logical_class(rx, rz):
    """Which logical operator a syndrome-free Pauli (rx, rz) applies to the stored qubit."""
    lx = (rx @ LZ) % 2
    lz = (rz @ LX) % 2
    return (lx + 2 * lz).astype(np.int64)



def _pure_errors():
    """For each of the 8 syndrome bits, the lightest single-type error that sets only it."""
    patterns = np.array(list(itertools.product([0, 1], repeat=N_QUBITS)), dtype=np.uint8)
    patterns = patterns[np.argsort(patterns.sum(axis=1), kind="stable")]   # lightest first
    px = np.zeros((N_SYNDROME_BITS, N_QUBITS), dtype=np.uint8)
    pz = np.zeros((N_SYNDROME_BITS, N_QUBITS), dtype=np.uint8)
    for i in range(len(X_STABS)):
        target = np.eye(len(X_STABS), dtype=np.uint8)[i]
        hit = np.all((patterns @ HX.T) % 2 == target, axis=1)
        pz[i] = patterns[np.argmax(hit)]
    for i in range(len(Z_STABS)):
        target = np.eye(len(Z_STABS), dtype=np.uint8)[i]
        hit = np.all((patterns @ HZ.T) % 2 == target, axis=1)
        px[len(X_STABS) + i] = patterns[np.argmax(hit)]
    return px, pz


PURE_X, PURE_Z = _pure_errors()


def reference_error(s):
    """C(s): XOR of the pure errors for every bit that is on (Varsamopoulos et al. 2017, arXiv:1705.00857)."""
    cx = (s @ PURE_X) % 2
    cz = (s @ PURE_Z) % 2
    return cx.astype(np.uint8), cz.astype(np.uint8)


def label(ex, ez):
    """The decoding target: syndrome index and the logical class of E * C(s)."""
    s = syndrome(ex, ez)
    cx, cz = reference_error(s)
    return syndrome_index(s), logical_class(ex ^ cx, ez ^ cz)


def correction_to_class(s, cor_x, cor_z):
    """Express any correction (e.g. from MWPM) as a class relative to C(s)."""
    cx, cz = reference_error(s)
    return logical_class(cor_x ^ cx, cor_z ^ cz)



def sample_depolarizing(p, n, rng):
    """n errors: each qubit independently gets X, Y or Z with probability p/3 each."""
    u = rng.random((n, N_QUBITS))
    x = u < p / 3
    y = (u >= p / 3) & (u < 2 * p / 3)
    z = (u >= 2 * p / 3) & (u < p)
    ex = (x | y).astype(np.uint8)
    ez = (z | y).astype(np.uint8)
    return ex, ez



def enumerate_all_errors():
    """All 4^9 = 262,144 Pauli errors, as (ex, ez), each of shape (262144, 9)."""
    codes = np.arange(4 ** N_QUBITS)
    per_qubit = (codes[:, None] // 4 ** np.arange(N_QUBITS)) % 4   # 0=I, 1=X, 2=Z, 3=Y
    ex = (per_qubit % 2).astype(np.uint8)
    ez = (per_qubit // 2).astype(np.uint8)
    return ex, ez


def weight_counts():
    """counts[s, c, w] = number of errors of weight w with syndrome s and class c."""
    ex, ez = enumerate_all_errors()
    s_idx, cls = label(ex, ez)
    w = (ex | ez).sum(axis=1)
    counts = np.zeros((N_SYNDROMES, 4, N_QUBITS + 1), dtype=np.int64)
    np.add.at(counts, (s_idx, cls, w), 1)
    return counts


COUNTS = weight_counts()


def joint_probs(p):
    """P(s, c) at physical error rate p. Shape (256, 4), sums to 1."""
    w = np.arange(N_QUBITS + 1)
    p_weight = (1 - p) ** (N_QUBITS - w) * (p / 3) ** w
    return COUNTS @ p_weight


def exact_logical_error(table, p):
    """Logical error rate of a decoder given as a table: syndrome index -> class."""
    P = joint_probs(p)
    return 1.0 - P[np.arange(N_SYNDROMES), table].sum()
