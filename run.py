import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import decoders as dec
import surface_code as sc

p_list = [0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30]
seeds = [0, 1, 2, 3, 4]
n_train = 10_000
mlp_hidden = [32, 32]
mlp_small_hidden = [12]
linear_hidden = []
mlp_steps, mlp_lr = 1000, 1e-2
vqc_layers, vqc_steps, vqc_lr = 6, 300, 0.05
res_file = Path("results/learned.json")
learned_names = ("MLP", "MLP-small", "Linear", "VQC")


def commutes(a_x, a_z, b_x, b_z):
    return (a_x @ b_z + a_z @ b_x) % 2 == 0


def check_code():
    zero = np.zeros(sc.n_q, dtype=np.uint8)
    stabs = [(h, zero) for h in sc.H_x] + [(zero, h) for h in sc.H_z]
    log_x, log_z = (sc.L_x, zero), (zero, sc.L_z)
    assert all(commutes(*a, *b) for a in stabs for b in stabs), "two stabilizers anticommute"
    assert all(commutes(*s, *log_x) and commutes(*s, *log_z) for s in stabs), "a logical anticommutes with a stabilizer"
    assert not commutes(*log_x, *log_z), "X_L and Z_L must anticommute"
    rank = np.linalg.matrix_rank(np.vstack([sc.H_x, sc.H_z]).astype(float))
    assert rank == 8, "stabilizers not independent"
    syn = dec.all_syn
    x_ref, z_ref = sc.ref_err(syn)
    assert np.array_equal(sc.get_syn(x_ref, z_ref), syn), "C(s) does not have syndrome s"
    assert np.all(sc.label(x_ref, z_ref)[1] == 0), "C(s) should have class I"
    assert sc.counts_by_wt.sum() == 4 ** 9
    print("check 1: 8 independent commuting stabilizers; X_L, Z_L commute with them and anticommute with each other")


def check_vs_sampling(tables, n=1_000_000, p_vals=(0.05, 0.15)):
    rng = np.random.default_rng(123)
    for p in p_vals:
        x_err, z_err = sc.sample_depol(p, n, rng)
        syn, cls = sc.label(x_err, z_err)
        for name, table in tables(p).items():
            f = np.mean(table[syn] != cls)
            se = np.sqrt(f * (1 - f) / n)
            exact = sc.exact_err(table, p)
            z = (f - exact) / se
            assert abs(z) < 4, f"{name} at p={p}: sampled {f:.5f} vs exact {exact:.5f} ({z:.1f} s.e.)"
            print(f"check 2: {name:4s} p={p}: sampled {f:.5f} +- {se:.5f}, exact {exact:.5f} ({z:+.1f} s.e.)")


def train_one(job):
    p, seed = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    counts = dec.count_samples(p, n_train, rng)
    out = {"p": p, "seed": seed}
    for name, model, steps, lr in [
        ("MLP", dec.MLP(mlp_hidden), mlp_steps, mlp_lr),
        ("MLP-small", dec.MLP(mlp_small_hidden), mlp_steps, mlp_lr),
        ("Linear", dec.MLP(linear_hidden), mlp_steps, mlp_lr),
        ("VQC", dec.VQC(vqc_layers, rng), vqc_steps, vqc_lr),
    ]:
        table, loss = dec.train(model, counts, steps, lr)
        out[name] = {"p_L": sc.exact_err(table, p), "loss": loss,
                     "table": table.tolist(), "params": dec.n_params(model)}
    print(f"  trained p={p} seed={seed}: " + ", ".join(
        f"{k} {out[k]['p_L']:.4f}" for k in learned_names), flush=True)
    return out


def get_learned():
    if res_file.exists() and "--retrain" not in sys.argv:
        print(f"loading trained decoders from {res_file} (pass --retrain to train again)")
        return json.loads(res_file.read_text())
    jobs = [(p, s) for p in p_list for s in seeds]
    print(f"training {len(jobs)} (p, seed) jobs ...")
    with Pool(min(len(jobs), 11)) as pool:
        runs = pool.map(train_one, jobs)
    res_file.parent.mkdir(exist_ok=True)
    res_file.write_text(json.dumps({"settings": {
        "n_train": n_train, "mlp_hidden": mlp_hidden, "mlp_small_hidden": mlp_small_hidden, "linear_hidden": linear_hidden,
        "mlp_steps": mlp_steps, "mlp_lr": mlp_lr, "vqc_layers": vqc_layers,
        "vqc_steps": vqc_steps, "vqc_lr": vqc_lr}, "runs": runs}))
    return json.loads(res_file.read_text())


def main():
    check_code()
    mwpm = dec.mwpm_answers()
    check_vs_sampling(lambda p: {"ML": dec.ml_answers(p), "MWPM": mwpm})

    ps = np.array(p_list)
    best = np.array([sc.exact_err(dec.ml_answers(p), p) for p in ps])
    mw = np.array([sc.exact_err(mwpm, p) for p in ps])
    print(f"MWPM and ML disagree on {np.sum(mwpm != dec.ml_answers(0.1))} of 256 syndromes")

    runs = get_learned()["runs"]
    learned = {}
    for name in learned_names:
        vals = np.array([[r[name]["p_L"] for r in runs if r["p"] == p] for p in p_list])
        learned[name] = (vals.mean(axis=1), vals.std(axis=1, ddof=1), vals)

    for name, (_, _, vals) in [("MWPM", (None, None, mw[:, None])), *learned.items()]:
        assert np.all(vals >= best[:, None] - 1e-12), f"{name} beat the best possible decoder: bug"
    print("check 3: no decoder beats the best possible one at any p, for any seed")

    print("\n   p     best    MWPM   " + "   ".join(f"{n:>16s}" for n in learned) + "   gap closed (MLP, MLP-small, Linear, VQC)")
    for i, p in enumerate(ps):
        cells = "   ".join(f"{learned[n][0][i]:.4f} +- {learned[n][1][i]:.4f}" for n in learned)
        closed = ", ".join(f"{(mw[i] - learned[n][0][i]) / (mw[i] - best[i]):+.2f}" for n in learned)
        print(f"{p:5.2f}  {best[i]:.4f}  {mw[i]:.4f}   {cells}   {closed}")
    params = {n: runs[0][n]["params"] for n in learned}
    print("parameters:", params)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(ps, ps, "k--", lw=1, label="one bare qubit (= p)")
    ax1.plot(ps, best, "k-o", ms=4, label="best possible (exact)")
    ax1.plot(ps, mw, "-s", ms=4, label="MWPM (exact)")
    for n, (mean, sd, _) in learned.items():
        ax1.errorbar(ps, mean, yerr=sd, marker="o", ms=3, capsize=3, label=f"{n} ({params[n]} params)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("physical error rate p"); ax1.set_ylabel("logical error rate")
    ax1.set_title("Logical error rate")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, which="both")

    ax2.axhline(0, color="k", lw=1, label="best possible")
    ax2.plot(ps, mw / best - 1, "-s", ms=4, label="MWPM")
    for n, (mean, sd, _) in learned.items():
        ax2.errorbar(ps, mean / best - 1, yerr=sd / best, marker="o", ms=3, capsize=3, label=n)
    ax2.set_xscale("log")
    ax2.set_xlabel("physical error rate p"); ax2.set_ylabel("extra logical errors vs best (fraction)")
    ax2.set_title("How far each decoder is from the best possible")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")
    fig.suptitle(f"Distance-3 surface code, depolarizing noise; learned decoders: {len(seeds)} seeds, "
                 f"{n_train} training samples each, mean +- s.d.", fontsize=10)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/logical_error_vs_p.png", dpi=150)
    print("saved figures/logical_error_vs_p.png")


if __name__ == "__main__":
    main()
