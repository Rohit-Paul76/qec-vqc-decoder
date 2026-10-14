# Decisions

Every choice in this project, the other options, why this one was picked, and what would change with a different pick.

## 0. Prior work this rests on (checked 14 Oct 2026)

| Source | What it gives this project |
|---|---|
| Varsamopoulos, Criger, Bertels, *Decoding small surface codes with feedforward neural networks*, [arXiv:1705.00857](https://arxiv.org/abs/1705.00857), QST 3, 015004 (2018) | The framing used here: write the error as E = S · C · L (stabilizer, a fixed "pure error" for the syndrome, a logical), so decoding is a 4-class classification of L. They test distances 3, 5, 7 under depolarizing noise and report that the network does slightly better than Blossom (MWPM) because "Blossom is known to misidentify Y errors, since it performs the decoding for X and Z errors separately". |
| Torlai, Melko, *Neural decoder for topological codes*, [arXiv:1610.04238](https://arxiv.org/abs/1610.04238), PRL 119, 030501 (2017) | First neural decoder for topological codes (a Boltzmann machine, toric code, phase-flip errors only). |
| Krastanov, Jiang, *Deep neural network probabilistic decoder for stabilizer codes*, [arXiv:1705.09334](https://arxiv.org/abs/1705.09334), Sci. Rep. 7, 11003 (2017) | A network that learns P(error given syndrome) on the toric code; credits its gain over other decoders to "taking into account the correlations between errors". |
| P. Zhang, *Correcting a noisy quantum computer using a quantum computer*, [arXiv:2506.08331](https://arxiv.org/abs/2506.08331) (2025) | The closest quantum-circuit decoder found: a trained decoding circuit for surface codes up to distance 7 with circuit-level noise, reported on par with MWPM. A different setting from ours (it decodes coherently on hardware; we classify a classical syndrome with a VQC). |
| PyMatching, [github.com/oscarhiggott/PyMatching](https://github.com/oscarhiggott/PyMatching), Apache-2.0; Higgott and Gidney, Quantum 9, 1600 (2025) | The MWPM decoder. `Matching(H)` takes a check matrix (rows = checks, columns = qubits, each column with one or two 1s); `decode_batch` returns one correction per syndrome. |

No paper found that compares a VQC syndrome classifier with an MLP, MWPM and the exact optimum on the same code. That comparison is this project.

## 1. Which code: distance-3 rotated surface code

**Options:** rotated surface code (chosen, 9 data qubits, 8 checks), unrotated surface code (13 data qubits at distance 3), toric code (periodic, 18 qubits at distance 3), Steane or Shor code.

**Why:** the smallest surface code that corrects any single-qubit error, and the one real hardware runs (Google's and others' memory experiments use the rotated layout). 8 syndrome bits means 256 syndromes, small enough to enumerate every error exactly (decision 4) and to put one qubit per syndrome bit in the VQC.

**Layout** (`surface_code.py`): qubits on a 3 × 3 grid. X-type checks: the top-left and bottom-right squares plus the top and bottom edges. Z-type checks: the other two squares plus the left and right edges. Logical X: X on the left column; logical Z: Z on the top row. `run.py` check 1 confirms the commutation pattern.

**With a larger distance:** distance 5 has 25 qubits and 24 syndrome bits (16.7 million syndromes), so no exact enumeration, and a VQC with one qubit per syndrome bit is out of reach for a laptop simulator.

## 2. Noise model: depolarizing, data qubits only, perfect syndrome

**Options:** bit flips only, independent X and Z (each with its own p), **depolarizing (chosen)**, phenomenological (also noisy syndrome bits), circuit-level (every gate can fail).

**Why depolarizing:** it is the simplest model where X and Z errors are correlated: a Y is an X and a Z on the same qubit, and it happens with probability p/3, not (p/3)². That correlation is the one thing MWPM (which decodes X and Z separately) cannot see, so it is the only place a learned decoder has room to beat MWPM. Under independent X/Z noise MWPM is already close to optimal (Varsamopoulos et al. say so for their "channel capacity" model), and there would be nothing to learn.

**Why no measurement noise:** one round of syndrome is then enough. Measurement noise needs repeated rounds, and the decoder input becomes a 3D block of syndromes, too big to enumerate.

**With circuit-level noise:** realistic, but needs `stim` and repeated rounds; listed in README next steps.

## 3. Decoding as 4-class classification (the "simple decoder" + a class)

**Options:**
- **Predict the logical class L given the syndrome (chosen).** A fixed pure error C(s) turns the syndrome off; the decoder only names which of I, X, Z, Y to add.
- Predict the full 9-qubit correction (18 output bits).

**Why:** many corrections differ only by a stabilizer and do exactly the same thing to the stored qubit, so "which correction" has many right answers while "which class" has exactly one. It also makes every decoder comparable: MWPM's correction is turned into a class too (`correction_to_class`).

**Pure errors:** for each syndrome bit, the lightest single-type error that turns on that bit alone (found by brute force over 2^9 patterns). C(s) is the XOR of those for the bits that are on. Which pure errors are used changes the labels (it relabels classes per syndrome) but not any logical error rate.

## 4. Exact logical error rates instead of Monte Carlo

**Options:**
- Sample errors, decode, count failures (Monte Carlo); error bars from the binomial distribution.
- **Enumerate all 4^9 = 262,144 errors once (chosen).** Store, for each syndrome s, class c and weight w, how many errors there are. A depolarizing error of weight w has probability (1 − p)^(9−w) (p/3)^w, so P(s, c) at any p is a short sum.

**Why:** every decoder here is a function from 256 syndromes to 4 classes, so its failure rate is exactly 1 − Σ_s P(s, decoder(s)). No sampling error at all, and "the best decoder is never beaten" becomes a strict check. The best possible (maximum-likelihood) decoder is then just argmax over c of P(s, c).

**Check:** `run.py` check 2 decodes 10^6 sampled errors with the ML and MWPM tables and confirms the sampled rate is within 4 standard errors of the exact one (it is within about 1).

**What carries error bars then:** the learned decoders, whose tables depend on the random training sample and starting weights. They are trained with 5 seeds; curves show mean ± standard deviation over seeds.

**With Monte Carlo:** needed from distance 5 on. To resolve p_L near 10^-3 to 10% you need about 10^5 samples per point.

## 5. MWPM setup

**Options:** **pymatching with equal weights (chosen)**, weights log((1 − q)/q) with q = 2p/3 (the chance a qubit has an X part), pymatching's correlated matching (v2.3+, needs a detector error model from stim).

**Why:** every qubit has the same error probability, so all weights are equal and the weight value does not change which matching is lightest. The answer does not depend on p, so the MWPM table is built once.

**With correlated matching:** it would reweight edges after a first pass to account for Y errors, closing part of the gap to the optimum. That is a stronger classical baseline, worth a sentence in the README, but it needs stim, which is out of scope.

## 6. Training data: 10,000 samples at the p being tested, 5 seeds

**Options:** train once at one p and test at all p (what Varsamopoulos et al. did, at the p where Blossom fails 25% of the time); **train a fresh decoder at each p (chosen)**; train on exact probabilities instead of samples.

**Why:** it asks the simplest question: "given samples from this noise level, how close does the learner get?" Exact probabilities would hand the learner the answer.

**Why 10,000:** enough to see most syndromes at p ≥ 0.06 (212 of 256 at p = 0.1), small enough that the learned decoders are not just counting. With many more samples the MLP becomes a lookup table of majority votes and reaches the optimum.

**Where 10,000 is too few (seen in the results):** at p = 0.01 the failures come from syndromes that appear only a few dozen times in 10,000 samples, so every learned decoder is worse than MWPM there, and in 2 of 5 seeds the VQC and the small MLP mislabel the syndrome of one single-qubit error. More samples, or training at a higher p and testing lower (the Varsamopoulos et al. approach), would fix this; it is listed in the README's next steps.

**Training trick:** the loss over 10,000 samples equals a weighted loss over the distinct syndromes, weighted by how often each (syndrome, class) pair appeared (`decoders.train`). So training is full-batch on at most 256 inputs. Same answer, far faster for the VQC.

## 7. MLP: 8 → 32 → 32 → 4, plus a small 8 → 12 → 4

**Why the main one:** a plain two-hidden-layer network, enough capacity to represent any 256-entry table. Adam, learning rate 0.01, 1000 full-batch steps. Fixed before looking at results on the true distribution; no tuning against the exact error rates.

**Why the small one:** the VQC has far fewer parameters than the main MLP. The small MLP has about as many as the VQC, so a gap between VQC and MLP can be split into "fewer parameters" and "being a quantum circuit".

## 8. Linear control: logistic regression on the syndrome bits

**Why it is there:** the VQC (decision 9) ends in a linear layer. With every gate set to the identity it *is* logistic regression on the bits. So "Linear" is the floor the circuit must beat to show it adds anything, and it is a plain classical model with 36 parameters (8 × 4 weights + 4 biases).

## 9. VQC: 8 qubits, 6 layers of Rot + CNOT ring, ⟨Z⟩ on every qubit, then a linear layer

**Encoding:** one qubit per syndrome bit, prepared in the basis state |s⟩ (basis encoding, as in the parity classifier). `RX(π s_q)` is used instead of `qml.BasisState` because it takes the whole batch of syndromes at once; the two differ only by a global phase.

**Ansatz:** the parity classifier's layer (a `Rot` on each qubit, then a ring of CNOTs), repeated 6 times: 144 angles.

**Readout (the decision that mattered):**

| Option | Result at p = 0.1 (one tuning run each, same 10,000 samples, exact p_L; best 0.102, MWPM 0.114) |
|---|---|
| Probabilities of 2 of the 8 qubits, read directly as the 4 classes | 0.16 to 0.24 for 4 to 16 layers, worse than MWPM |
| Same, but 2 extra readout qubits starting in \|0⟩ (10 qubits) | 0.16 to 0.30, and 10–20 min per run |
| **⟨Z⟩ on all 8 qubits → linear layer → 4 scores (chosen)** | 0.121 (4 layers), **0.111 (6)**, 0.114 (8), in about 1 min per run |

**Why the first readout fails (worth knowing):** a circuit U maps the 256 orthonormal input states |s⟩ to 256 orthonormal states. The probability of reading outcome c on 2 of the 8 qubits, summed over all 256 inputs, is

```
sum_s q(c | s) = Tr( U† (|c><c| ⊗ I_64) U ) = Tr( |c><c| ⊗ I_64 ) = 64
```

(I_64 = identity on the other 6 qubits). So every class receives exactly a quarter of the total probability, whatever the angles. The best decoder answers I on 161 of the 256 syndromes, X on 43, Z on 43 and Y on 9: far from a quarter each. The circuit cannot express that preference. Extra readout qubits remove this constraint in principle, but in practice those circuits trained badly (training losses 0.48 to 0.97 against the MLP's 0.26). A classical linear layer on the ⟨Z⟩ values removes it directly.

**Why 6 layers:** chosen by **training loss** (0.33, 0.29 and 0.31 for 4, 6 and 8 layers), not by the exact p_L, which uses the true noise distribution and would amount to tuning on the test set. It is also the lowest p_L, but that was not the criterion.

**Starting angles:** uniform in [0, 2π]. Starting near zero (within ±0.1 or ±0.5) was tried with the 10-qubit readout and was worse (p_L 0.21 to 0.26).

**Parameters:** 144 angles + 36 in the linear layer = 180. The small MLP has 160.

**Training:** Adam, learning rate 0.05, 300 full-batch steps, exact gradients by backpropagation through the simulator (`default.qubit`, `diff_method="backprop"`), no shot noise and no hardware noise.

**What this does not test:** hardware noise on the VQC itself, shot noise in estimating ⟨Z⟩, or training with the parameter-shift rule. Each would make the VQC worse, not better.

## 10. Sweep: p from 0.01 to 0.3

**Why:** below 0.01 almost every syndrome is trivial and the decoders differ by tiny absolute amounts; around 0.1 the best decoder's logical error rate crosses the bare-qubit line p (the break-even point for distance 3 under depolarizing noise); 0.3 shows the regime where the code no longer helps. Seven points keep the VQC training to a few minutes per seed. Nothing depends on the exact grid.
