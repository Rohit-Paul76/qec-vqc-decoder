import itertools

import numpy as np

n_q = 9

x_stabs = [
    [0, 1, 3, 4],
    [4, 5, 7, 8],
    [1, 2],
    [6, 7],
]
z_stabs = [
    [1, 2, 4, 5],
    [3, 4, 6, 7],
    [0, 3],
    [5, 8],
]

x_logical = [0, 3, 6]
z_logical = [0, 1, 2]


def to_matrix(stabs):
    m = np.zeros((len(stabs), n_q), dtype=np.uint8)
    for i, qs in enumerate(stabs):
        m[i, qs] = 1
    return m


H_x = to_matrix(x_stabs)
H_z = to_matrix(z_stabs)
L_x = to_matrix([x_logical])[0]
L_z = to_matrix([z_logical])[0]

n_bits = len(x_stabs) + len(z_stabs)
n_syn = 2 ** n_bits

cls_names = ["I", "X", "Z", "Y"]


def get_syn(x_err, z_err):
    syn_xchk = (z_err @ H_x.T) % 2
    syn_zchk = (x_err @ H_z.T) % 2
    return np.concatenate([syn_xchk, syn_zchk], axis=-1).astype(np.uint8)


def syn_to_idx(syn):
    pow2 = 2 ** np.arange(n_bits - 1, -1, -1)
    return (syn @ pow2).astype(np.int64)


def idx_to_syn(idx):
    idx = np.asarray(idx)
    shifts = np.arange(n_bits - 1, -1, -1)
    return ((idx[..., None] >> shifts) & 1).astype(np.uint8)


def get_cls(x_res, z_res):
    flip_x = (x_res @ L_z) % 2
    flip_z = (z_res @ L_x) % 2
    return (flip_x + 2 * flip_z).astype(np.int64)


def pure_errs():
    pats = np.array(list(itertools.product([0, 1], repeat=n_q)), dtype=np.uint8)
    pats = pats[np.argsort(pats.sum(axis=1), kind="stable")]
    x_pure = np.zeros((n_bits, n_q), dtype=np.uint8)
    z_pure = np.zeros((n_bits, n_q), dtype=np.uint8)
    for i in range(len(x_stabs)):
        want = np.eye(len(x_stabs), dtype=np.uint8)[i]
        hit = np.all((pats @ H_x.T) % 2 == want, axis=1)
        z_pure[i] = pats[np.argmax(hit)]
    for i in range(len(z_stabs)):
        want = np.eye(len(z_stabs), dtype=np.uint8)[i]
        hit = np.all((pats @ H_z.T) % 2 == want, axis=1)
        x_pure[len(x_stabs) + i] = pats[np.argmax(hit)]
    return x_pure, z_pure


x_pure, z_pure = pure_errs()


def ref_err(syn):
    x_ref = (syn @ x_pure) % 2
    z_ref = (syn @ z_pure) % 2
    return x_ref.astype(np.uint8), z_ref.astype(np.uint8)


def label(x_err, z_err):
    syn = get_syn(x_err, z_err)
    x_ref, z_ref = ref_err(syn)
    return syn_to_idx(syn), get_cls(x_err ^ x_ref, z_err ^ z_ref)


def fix_to_cls(syn, x_fix, z_fix):
    x_ref, z_ref = ref_err(syn)
    return get_cls(x_fix ^ x_ref, z_fix ^ z_ref)


def sample_depol(p, n, rng):
    u = rng.random((n, n_q))
    is_x = u < p / 3
    is_y = (u >= p / 3) & (u < 2 * p / 3)
    is_z = (u >= 2 * p / 3) & (u < p)
    x_err = (is_x | is_y).astype(np.uint8)
    z_err = (is_z | is_y).astype(np.uint8)
    return x_err, z_err


def all_errs():
    codes = np.arange(4 ** n_q)
    pauli = (codes[:, None] // 4 ** np.arange(n_q)) % 4
    x_err = (pauli % 2).astype(np.uint8)
    z_err = (pauli // 2).astype(np.uint8)
    return x_err, z_err


def weight_counts():
    x_err, z_err = all_errs()
    syn, cls = label(x_err, z_err)
    wt = (x_err | z_err).sum(axis=1)
    counts = np.zeros((n_syn, 4, n_q + 1), dtype=np.int64)
    np.add.at(counts, (syn, cls, wt), 1)
    return counts


counts_by_wt = weight_counts()


def joint_probs(p):
    wt = np.arange(n_q + 1)
    p_wt = (1 - p) ** (n_q - wt) * (p / 3) ** wt
    return counts_by_wt @ p_wt


def exact_err(table, p):
    P = joint_probs(p)
    return 1.0 - P[np.arange(n_syn), table].sum()
