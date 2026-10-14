# Learning to decode the distance-3 surface code: VQC vs MLP vs MWPM vs the best possible

**Question:** can a small variational quantum circuit (VQC) learn to decode the distance-3 surface code, and how does it compare with a classical neural network, the standard decoder (minimum-weight perfect matching, MWPM) and the best possible decoder as noise grows?

![logical error rate vs physical error rate](figures/logical_error_vs_p.png)

## Setup

- **Code:** distance-3 rotated surface code, 9 data qubits, 8 stabilizers, so an 8-bit syndrome (256 possible syndromes).
- **Noise:** depolarizing on the data qubits (X, Y or Z, each with probability p/3), perfect syndrome measurement (code-capacity noise).
- **Decoding as classification:** following [Varsamopoulos et al. 2017](https://arxiv.org/abs/1705.00857), a fixed lookup correction C(s) removes the syndrome, and the decoder only has to say which logical operator is left: I, X, Z or Y. Syndrome (8 bits) → class (4).
- **Decoders:**

| Decoder | What it is | Parameters |
|---|---|---|
| best possible | maximum likelihood: for each syndrome, the class with the highest exact probability, from all 4^9 errors | — |
| MWPM | [`pymatching`](https://github.com/oscarhiggott/PyMatching), X and Z parts matched separately | — |
| MLP | PyTorch, 8 → 32 → 32 → 4 | 1,476 |
| MLP-small | 8 → 12 → 4, about the VQC's size | 160 |
| Linear | logistic regression on the 8 bits (the VQC with every gate removed) | 36 |
| VQC | PennyLane, 8 qubits, basis encoding, 6 layers of Rot + CNOT ring, ⟨Z⟩ on every qubit → linear layer | 180 |

The learned decoders are trained, at each p, on 10,000 sampled (syndrome, class) pairs, with 5 seeds.

**Every logical error rate is exact.** A decoder is a table over 256 syndromes, and the exact probability of each (syndrome, class) pair comes from enumerating all 4^9 = 262,144 errors. So the best and MWPM curves have no error bars at all; the learned curves' error bars (± 1 s.d. over 5 seeds) come only from training randomness.

## Results

| p | best | MWPM | MLP | MLP-small | Linear | VQC |
|---|---|---|---|---|---|---|
| 0.01 | 0.0013 | 0.0015 | 0.0018 ± 0.0001 | 0.0030 ± 0.0017 | 0.0020 ± 0.0000 | 0.0035 ± 0.0017 |
| 0.03 | 0.0111 | 0.0130 | 0.0124 ± 0.0004 | 0.0133 ± 0.0003 | 0.0166 ± 0.0001 | 0.0169 ± 0.0013 |
| 0.06 | 0.0410 | 0.0470 | 0.0430 ± 0.0006 | 0.0456 ± 0.0009 | 0.0590 ± 0.0002 | 0.0525 ± 0.0021 |
| 0.10 | 0.1019 | 0.1138 | 0.1046 ± 0.0006 | 0.1097 ± 0.0019 | 0.1371 ± 0.0011 | 0.1146 ± 0.0018 |
| 0.15 | 0.1980 | 0.2154 | 0.2012 ± 0.0008 | 0.2068 ± 0.0039 | 0.2533 ± 0.0110 | 0.2106 ± 0.0045 |
| 0.20 | 0.3020 | 0.3215 | 0.3066 ± 0.0015 | 0.3088 ± 0.0019 | 0.3848 ± 0.0061 | 0.3190 ± 0.0044 |
| 0.30 | 0.4934 | 0.5089 | 0.4995 ± 0.0010 | 0.5006 ± 0.0013 | 0.5737 ± 0.0020 | 0.5132 ± 0.0041 |

**What it shows**
- **MWPM is not optimal under depolarizing noise.** Its logical error rate is 3–19% above the best decoder's, more at low p (they disagree on 116 of the 256 syndromes). MWPM decodes the X and Z parts separately, so a Y error counts as two errors instead of one.
- **The MLP learns most of that correlation.** From p = 0.06 up it closes 60–80% of the gap between MWPM and the best decoder, from 10,000 samples.
- **The VQC is roughly level with MWPM** for p ≥ 0.1 (within about one standard deviation of it from 0.1 to 0.3, sometimes above, sometimes below), and worse below that. It clearly beats the Linear control at every p ≥ 0.06, so the circuit adds something a linear model on the bits cannot do. But it trails the small MLP, which has a similar number of parameters, at every p: on this task, the circuit is a weaker learner than a classical network of the same size.
- **At low p, 10,000 samples are not enough for any learned decoder.** At p = 0.01 all of them are worse than MWPM. Failures there come from rare syndromes that appear only a few dozen times in training. Two of five seeds for the VQC and MLP-small mislabel the syndrome of one single-qubit error (the extra failure rate, about 0.003, is about p/3, the probability of that one error), which is where the large error bars come from.
- **The best decoder breaks even with a bare qubit near p = 0.1:** below that, encoding helps; above it, the distance-3 code is worse than one unprotected qubit.

**A VQC design point that mattered** (details in [DECISIONS.md](DECISIONS.md), section 9): the first version read the 4 classes directly as the outcome probabilities of 2 of the 8 qubits. That can never work well here: for any circuit, those probabilities summed over all 256 basis inputs give exactly 64 to each class, but the best decoder answers I on 161 syndromes. Reading ⟨Z⟩ on every qubit into a small linear layer removed the constraint and took the VQC from worse than MWPM (0.16–0.24 at p = 0.1) to level with it.

## Checks (`run.py`)

1. The 8 stabilizers commute and are independent; logical X and Z commute with all of them and anticommute with each other. The lookup correction C(s) has syndrome s.
2. The exact logical error rates agree with plain Monte Carlo (10^6 sampled errors at p = 0.05 and 0.15, for the best decoder and MWPM): all within 1.2 standard errors.
3. No decoder beats the best possible one at any p, for any seed (exact numbers, so the only slack is floating-point rounding).

## Run

```
uv run python run.py             # checks + figure, loading the trained decoders from results/learned.json
uv run python run.py --retrain   # train all 140 learned decoders again (about 10 min on 11 CPU cores)
```

## Files

| File | What it holds |
|---|---|
| `surface_code.py` | the code (stabilizers, logicals), syndromes, logical classes, the lookup correction C(s), depolarizing sampling, exact P(syndrome, class) |
| `decoders.py` | best possible, MWPM, MLP, VQC; the shared training loop |
| `run.py` | the three checks, training sweep, results table, figure |
| `results/learned.json` | every trained decoder's table and exact error rate (so the figure can be redrawn without training) |
| `DECISIONS.md` | prior work, and every design choice with its alternatives |

## Where this comes from

The previous project, [`qec-repetition-code`](../qec-repetition-code), showed that the 3-qubit repetition code catches bit flips but cannot see phase flips: under phase-flip noise it is about three times worse than a bare qubit. The surface code checks both kinds of parity (X-type and Z-type stabilizers), so it corrects any single-qubit error, including Y, which is where the decoding question here gets interesting.

## Next steps

- Measurement noise and repeated rounds (phenomenological, then circuit-level noise with `stim`), where the decoder input becomes a block of syndromes over time.
- Distance 5 and up, where exact enumeration is impossible (24 syndrome bits) and evaluation needs Monte Carlo.
- A stronger classical baseline: pymatching's correlated matching, which reweights for Y errors.
- The VQC under its own hardware noise and shot noise, trained with the parameter-shift rule instead of backpropagation.
- More training samples or a mix of noise levels, to see whether the learned decoders reach the optimum at low p.
