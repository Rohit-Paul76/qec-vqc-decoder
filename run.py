"""Checks, the noise sweep, and the figure.

Run: uv run python run.py            -> figures/logical_error_vs_p.png, results/learned.json
     uv run python run.py --retrain   (train the MLPs and VQCs again instead of loading)

Every logical error rate here is exact: a decoder is a table over the 256 syndromes, and
surface_code.joint_probs gives P(syndrome, class) from all 4^9 errors. The only randomness
is in training (which samples are drawn, and the starting weights), so the learned
decoders are trained with several seeds and their curves carry mean +- standard deviation
over seeds.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import decoders as dec
import surface_code as sc

PS = [0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30]
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN = 10_000              # samples each learned decoder sees, at the p it is tested on
MLP_HIDDEN = [32, 32]
MLP_SMALL_HIDDEN = [12]       # about as many parameters as the VQC
LINEAR_HIDDEN = []            # no hidden layer: logistic regression on the syndrome bits
MLP_STEPS, MLP_LR = 1000, 1e-2
VQC_LAYERS, VQC_STEPS, VQC_LR = 6, 300, 0.05
RESULTS = Path("results/learned.json")
LEARNED = ("MLP", "MLP-small", "Linear", "VQC")


def commutes(ax, az, bx, bz):
    """Two Paulis commute when (X of one meets Z of the other) happens an even number of times."""
    return (ax @ bz + az @ bx) % 2 == 0


def check_code():
    """Check 1: the stabilizers and logicals have the commutation pattern of a code."""
    zero = np.zeros(sc.N_QUBITS, dtype=np.uint8)
    stabs = [(h, zero) for h in sc.HX] + [(zero, h) for h in sc.HZ]
    lx, lz = (sc.LX, zero), (zero, sc.LZ)
    assert all(commutes(*a, *b) for a in stabs for b in stabs), "two stabilizers anticommute"
    assert all(commutes(*s, *lx) and commutes(*s, *lz) for s in stabs), "a logical anticommutes with a stabilizer"
    assert not commutes(*lx, *lz), "X_L and Z_L must anticommute"
    rank = np.linalg.matrix_rank(np.vstack([sc.HX, sc.HZ]).astype(float))
    assert rank == 8, "stabilizers not independent"   # 9 qubits - 8 checks = 1 logical qubit
    # A syndrome-free error that is not a stabilizer must be a logical; the ML check below
    # would fail if labels were wrong, but test the reference errors directly too.
    s = dec.ALL_SYNDROMES
    cx, cz = sc.reference_error(s)
    assert np.array_equal(sc.syndrome(cx, cz), s), "C(s) does not have syndrome s"
    assert np.all(sc.label(cx, cz)[1] == 0), "C(s) should have class I by definition"
    assert sc.COUNTS.sum() == 4 ** 9
    print("check 1: 8 independent commuting stabilizers; X_L, Z_L commute with them and anticommute with each other")


def check_exact_against_sampling(tables, n=1_000_000, ps=(0.05, 0.15)):
    """Check 2: the exact logical error rates agree with plain Monte Carlo.

    Draw n errors, decode each with the table, count failures. The count is binomial, so
    the standard error is sqrt(f (1 - f) / n). Any bug in the enumeration (wrong
    syndrome, wrong class, wrong probabilities) would show up as a gap of many standard errors.
    """
    rng = np.random.default_rng(123)
    for p in ps:
        ex, ez = sc.sample_depolarizing(p, n, rng)
        s_idx, cls = sc.label(ex, ez)
        for name, table in tables(p).items():
            f = np.mean(table[s_idx] != cls)
            se = np.sqrt(f * (1 - f) / n)
            exact = sc.exact_logical_error(table, p)
            z = (f - exact) / se
            assert abs(z) < 4, f"{name} at p={p}: sampled {f:.5f} vs exact {exact:.5f} ({z:.1f} s.e.)"
            print(f"check 2: {name:4s} p={p}: sampled {f:.5f} +- {se:.5f}, exact {exact:.5f} ({z:+.1f} s.e.)")


def train_one(job):
    """Train the three learned decoders for one (p, seed). Returns their exact error rates."""
    p, seed = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    counts = dec.training_counts(p, N_TRAIN, rng)
    out = {"p": p, "seed": seed}
    for name, model, steps, lr in [
        ("MLP", dec.MLP(MLP_HIDDEN), MLP_STEPS, MLP_LR),
        ("MLP-small", dec.MLP(MLP_SMALL_HIDDEN), MLP_STEPS, MLP_LR),
        ("Linear", dec.MLP(LINEAR_HIDDEN), MLP_STEPS, MLP_LR),
        ("VQC", dec.VQC(VQC_LAYERS, rng), VQC_STEPS, VQC_LR),
    ]:
        table, loss = dec.train(model, counts, steps, lr)
        out[name] = {"p_L": sc.exact_logical_error(table, p), "loss": loss,
                     "table": table.tolist(), "params": dec.n_params(model)}
    print(f"  trained p={p} seed={seed}: " + ", ".join(
        f"{k} {out[k]['p_L']:.4f}" for k in LEARNED), flush=True)
    return out


def learned_results():
    if RESULTS.exists() and "--retrain" not in sys.argv:
        print(f"loading trained decoders from {RESULTS} (pass --retrain to train again)")
        return json.loads(RESULTS.read_text())
    jobs = [(p, s) for p in PS for s in SEEDS]
    print(f"training {len(jobs)} (p, seed) jobs ...")
    with Pool(min(len(jobs), 11)) as pool:
        runs = pool.map(train_one, jobs)
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({"settings": {
        "N_TRAIN": N_TRAIN, "MLP_HIDDEN": MLP_HIDDEN, "MLP_SMALL_HIDDEN": MLP_SMALL_HIDDEN, "LINEAR_HIDDEN": LINEAR_HIDDEN,
        "MLP_STEPS": MLP_STEPS, "MLP_LR": MLP_LR, "VQC_LAYERS": VQC_LAYERS,
        "VQC_STEPS": VQC_STEPS, "VQC_LR": VQC_LR}, "runs": runs}))
    return json.loads(RESULTS.read_text())


def main():
    check_code()
    mwpm = dec.mwpm_table()
    check_exact_against_sampling(lambda p: {"ML": dec.ml_table(p), "MWPM": mwpm})

    ps = np.array(PS)
    ml = np.array([sc.exact_logical_error(dec.ml_table(p), p) for p in ps])
    mw = np.array([sc.exact_logical_error(mwpm, p) for p in ps])
    print(f"MWPM and ML disagree on {np.sum(mwpm != dec.ml_table(0.1))} of 256 syndromes")

    runs = learned_results()["runs"]
    learned = {}
    for name in LEARNED:
        vals = np.array([[r[name]["p_L"] for r in runs if r["p"] == p] for p in PS])   # (len(PS), seeds)
        learned[name] = (vals.mean(axis=1), vals.std(axis=1, ddof=1), vals)

    # Check 3: nothing beats the best possible decoder. Every number is exact, so the only
    # slack allowed is floating-point rounding.
    for name, (_, _, vals) in [("MWPM", (None, None, mw[:, None])), *learned.items()]:
        assert np.all(vals >= ml[:, None] - 1e-12), f"{name} beat the best possible decoder: bug"
    print("check 3: no decoder beats the best possible one at any p, for any seed")

    # How much of the MWPM -> best gap each learned decoder closes (1 = all of it, < 0 = worse than MWPM).
    print("\n   p     best    MWPM   " + "   ".join(f"{n:>16s}" for n in learned) + "   gap closed (MLP, MLP-small, Linear, VQC)")
    for i, p in enumerate(ps):
        cells = "   ".join(f"{learned[n][0][i]:.4f} +- {learned[n][1][i]:.4f}" for n in learned)
        closed = ", ".join(f"{(mw[i] - learned[n][0][i]) / (mw[i] - ml[i]):+.2f}" for n in learned)
        print(f"{p:5.2f}  {ml[i]:.4f}  {mw[i]:.4f}   {cells}   {closed}")
    params = {n: runs[0][n]["params"] for n in learned}
    print("parameters:", params)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(ps, ps, "k--", lw=1, label="one bare qubit (= p)")
    ax1.plot(ps, ml, "k-o", ms=4, label="best possible (exact)")
    ax1.plot(ps, mw, "-s", ms=4, label="MWPM (exact)")
    for n, (mean, sd, _) in learned.items():
        ax1.errorbar(ps, mean, yerr=sd, marker="o", ms=3, capsize=3, label=f"{n} ({params[n]} params)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("physical error rate p"); ax1.set_ylabel("logical error rate")
    ax1.set_title("Logical error rate")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, which="both")

    ax2.axhline(0, color="k", lw=1, label="best possible")
    ax2.plot(ps, mw / ml - 1, "-s", ms=4, label="MWPM")
    for n, (mean, sd, _) in learned.items():
        ax2.errorbar(ps, mean / ml - 1, yerr=sd / ml, marker="o", ms=3, capsize=3, label=n)
    ax2.set_xscale("log")
    ax2.set_xlabel("physical error rate p"); ax2.set_ylabel("extra logical errors vs best (fraction)")
    ax2.set_title("How far each decoder is from the best possible")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")
    fig.suptitle(f"Distance-3 surface code, depolarizing noise; learned decoders: {len(SEEDS)} seeds, "
                 f"{N_TRAIN} training samples each, mean +- s.d.", fontsize=10)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/logical_error_vs_p.png", dpi=150)
    print("saved figures/logical_error_vs_p.png")


if __name__ == "__main__":
    main()
