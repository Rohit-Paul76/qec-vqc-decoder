"""Distance-3 rotated surface code, written as bit arrays (no circuits needed).

Nine data qubits sit on a 3 x 3 grid, numbered q = 3 * row + col:

    0 - 1 - 2
    |   |   |
    3 - 4 - 5
    |   |   |
    6 - 7 - 8

A Pauli error on the 9 qubits is stored as two bit vectors of length 9:
ex[q] = 1 if qubit q has an X part (X or Y), ez[q] = 1 if it has a Z part (Z or Y).
So I = (0, 0), X = (1, 0), Z = (0, 1), Y = (1, 1) on each qubit.

Two Paulis commute exactly when the number of places where one has X and the other
has Z is even. Everything below (syndromes, logical classes, checks) uses only that rule.
"""
import itertools

import numpy as np

N_QUBITS = 9

# Stabilizers, as lists of the qubits they act on.
# X-type: products of X. They anticommute with (so they detect) Z parts of an error.
X_STABS = [
    [0, 1, 3, 4],  # top-left square
    [4, 5, 7, 8],  # bottom-right square
    [1, 2],        # top edge
    [6, 7],        # bottom edge
]
# Z-type: products of Z. They detect X parts of an error.
Z_STABS = [
    [1, 2, 4, 5],  # top-right square
    [3, 4, 6, 7],  # bottom-left square
    [0, 3],        # left edge
    [5, 8],        # right edge
]

# Logical operators: X on the left column, Z on the top row.
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
    """8 syndrome bits for errors given as arrays of shape (..., 9).

    The first 4 bits are the X-type stabilizers (they see ez), the last 4 the Z-type
    (they see ex). Bit i is 1 when the error anticommutes with stabilizer i.
    """
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
    """Which logical operator a syndrome-free Pauli (rx, rz) applies to the stored qubit.

    It flips the logical Z value (has a logical X part) when it anticommutes with Z_L,
    i.e. its X part overlaps the top row an odd number of times. Same for the Z part
    and X_L. Returns lx + 2 * lz: 0 = I, 1 = X, 2 = Z, 3 = Y.
    """
    lx = (rx @ LZ) % 2
    lz = (rz @ LX) % 2
    return (lx + 2 * lz).astype(np.int64)


# ---- The "simple decoder": a fixed error for every syndrome ----------------------
#
# For each syndrome bit we fix one error that turns on that bit and no other (a "pure
# error"). XOR-ing the pure errors of all bits that are on gives an error C(s) with
# exactly syndrome s. Then any error E with syndrome s satisfies E = S * C(s) * L for some
# stabilizer S and logical L. S does nothing to the stored qubit, so decoding reduces to
# naming L, one of 4 classes (Varsamopoulos, Criger, Bertels 2017, arXiv:1705.00857).

def _pure_errors():
    """For each of the 8 syndrome bits, the lightest single-type error that sets only it.

    Bits 0-3 (X-type checks) are set by Z errors, bits 4-7 (Z-type checks) by X errors.
    Found by brute force over the 2^9 bit patterns. Returns arrays (8, 9) for x and z parts.
    """
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
    """C(s): XOR of the pure errors for every bit that is on. s has shape (..., 8)."""
    cx = (s @ PURE_X) % 2
    cz = (s @ PURE_Z) % 2
    return cx.astype(np.uint8), cz.astype(np.uint8)


def label(ex, ez):
    """The decoding target: syndrome index and the logical class of E * C(s).

    A decoder that outputs class L for syndrome s corrects with C(s) * L. That undoes E
    (up to a stabilizer) exactly when L is this label.
    """
    s = syndrome(ex, ez)
    cx, cz = reference_error(s)
    return syndrome_index(s), logical_class(ex ^ cx, ez ^ cz)


def correction_to_class(s, cor_x, cor_z):
    """Express any correction (e.g. from MWPM) as a class relative to C(s)."""
    cx, cz = reference_error(s)
    return logical_class(cor_x ^ cx, cor_z ^ cz)


# ---- Noise ----------------------------------------------------------------------

def sample_depolarizing(p, n, rng):
    """n errors: each qubit independently gets X, Y or Z with probability p/3 each."""
    u = rng.random((n, N_QUBITS))
    x = u < p / 3
    y = (u >= p / 3) & (u < 2 * p / 3)
    z = (u >= 2 * p / 3) & (u < p)
    ex = (x | y).astype(np.uint8)
    ez = (z | y).astype(np.uint8)
    return ex, ez


# ---- Every error at once: the exact joint distribution P(s, class) ----------------

def enumerate_all_errors():
    """All 4^9 = 262,144 Pauli errors, as (ex, ez), each of shape (262144, 9)."""
    codes = np.arange(4 ** N_QUBITS)
    per_qubit = (codes[:, None] // 4 ** np.arange(N_QUBITS)) % 4   # 0=I, 1=X, 2=Z, 3=Y
    ex = (per_qubit % 2).astype(np.uint8)
    ez = (per_qubit // 2).astype(np.uint8)
    return ex, ez


def weight_counts():
    """counts[s, c, w] = number of errors of weight w with syndrome s and class c.

    Under depolarizing noise every weight-w error has probability (1-p)^(9-w) (p/3)^w,
    so these counts give P(s, c) for any p without enumerating again.
    Shape (256, 4, 10).
    """
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
    """Logical error rate of a decoder given as a table: syndrome index -> class.

    It succeeds on syndrome s only when the true class is table[s], so
    P(fail) = 1 - sum over s of P(s, table[s]). No sampling involved.
    """
    P = joint_probs(p)
    return 1.0 - P[np.arange(N_SYNDROMES), table].sum()
