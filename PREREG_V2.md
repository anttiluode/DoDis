# DoDis v2 — visible-local Sihti memory, masked-mean control, 100-bird evaluation

**Protocol locked October 23, 2026, before running DoDis v2 on the CIFAR data.** The historical [DoDis v1 protocol](PREREG.md) and the published v1 receipt remain unchanged. The motivating observations were the v1 mean MSEs: masked baseline unreported; whole-image memory 0.01494, added learned binding 0.01417, shuffled binding 0.01438 (12 birds). These are **not** v2 measurements.

## Source, core, missing pieces

- Reuse the already trained 77.46%-validation CIFAR TinyCNN from SihtiGraphModel without retraining. All 32×32 RGB images are encoded with the same exact 64-piece Sihti additive sieve (four Laplacian-like octaves, four-by-four 8×8 tiles at every octave, fixed CIFAR RGB mean base). The complete sum reconstructs the input to <2e−6 error, *before* clipping.
- Evaluate the first **100** CIFAR-10 *test* birds from index 100 onward correctly classified by the intact frozen CNN (fixed image selection, independent of mask/repair). Use RNG SeedSequence([41, image index, 774]) to mask every one of the 64 pieces with independent probability .5. **Observed pieces remain bitwise fixed** and the full unobserved target is used only when calculating post-solve MSE.
- Do not pass the unknown full core to the retrieval system. Only observed coarse bands and observed local neighbors may enter a donor address. The "core" is global in *extent*, but partial in *information*.

## Data roles and five training-independent decisions

All candidate and graph data originate in CIFAR-10 training images, never in test images. Split the train corpus exactly like DoDis v1 (NumPy RNG 41 shuffled separate bird and nonbird train indices):
- 256 birds solely for donor memory, all 16 source tile positions indexed separately in each of 4 bands, 4 donor slots selected per unknown target piece;
- 320 **different** birds to fit Graph A and its normalization, exactly as v1, with 80 learned bindings;
- 380 different birds + 220 different nonbirds to fit Graph B, with 70 bird + 70 nonbird validation images, four p levels (0.5,0.75,0.9,1) and a frozen TinyCNN decision-margin teacher. Graph B's held-out validation R² must be reported. This is not the per-image v3 SihtiGraphModel switch polynomial.

### New *local* memory retrieval, no hidden leakage

For each *missing* target tile j at band b and row/column (r,c):
1. Build an address from only observed eight-connected same-band neighbors, observed other-band tiles at the same position, and observed 4-connected neighboring tiles of the next coarser band when available. Record how many observed context pieces were used for each query.
2. Normalize the six per-piece features (three mean RGB values, x/y brightness gradient, RMS) per band using **memory-bank statistics** calculated across all training images and source tile positions; no target image pixels enter the learned normalizer.
3. For each training image and **each of its 16 potential source tile positions** in the same band, compare the contextual neighbors at relative offsets. A source image may supply a patch at a different absolute tile than the test target. Missing counterparts at donor boundaries receive a fixed penalty. Rank by weighted feature mismatch plus a small fixed patch-energy prior. If no visible context exists, label that address as zero evidence and fall back to the fixed energy prior, never to hidden truth.
4. Choose the four best candidates from **distinct training images** per missing tile, each candidate preserving the actual 3×8×8 residue values of the donor. Include the zero-residue/mean-color alternative. Different target pieces may use different source images and source positions. Donor identities and context counts must be saved in the per-image receipt.

This is a local content-addressing **experiment** and does not automatically transfer the P3 learned regression/codebook from SihtiImageModel v1.

## Six controlled conditions, same masks and solver

1. **masked_mean**: No fill. Missing residues remain zero; report its image pixel MSE and actual CNN margin before trying any candidate.
2. **global_memory**: DoDis v1's top-four *whole-image* donors based only on visible pieces; memory rank plus boundary smoothness; identical v1 weights.
3. **local_memory**: Top-four *per-piece local* donors; same smoothness and memory-rank weights, no consistency or CNN purpose.
4. **local_binding**: Same local candidates plus Graph A learned from other images.
5. **local_shuffled**: Same local candidates plus Graph A with its weight matrices permuted across edges (precommitted deterministic shuffle seed 91).
6. **local_purpose**: Same local candidates plus Graph A and a separately fitted shared CNN-purpose readout.

All dynamic modes start with donor slot 0 for missing pieces, use three asynchronous candidate-enumeration sweeps at fixed seed 41+test image index, and the same memory weight .05, seam weight .75, binding .4, purpose .12 (zero in the other arms). The only difference between arms is the declared information source. Every accepted move must satisfy ΔE≤1e−8. A decrease in the explicit energy is *not* a decrease in ground-truth error.

## Measurements and precommitted pass/fail gates

Every test bird is saved independently after completion for restart and audit. Full **100-bird** JSON must include each bird's six reconstruction MSEs, hidden-residue MSEs, original/corrupted/completed real CNN margins, accepted moves, energy traces, exact visible mask, training-donor provenance and local context counts. Separately save summary statistics, paired learned-minus-control MSE differences, a two-sided paired sign-flip test (5,000 flips) and a 95% paired bootstrap interval (5,000 paired resamples), seeded independently of solver.

**G0 — source safety:** source roles disjoint, training donors only; hidden target residues cannot affect selected candidates or solver; exact all-piece reconstruction; original birds classified correctly; all observed pieces retained.

**G1 — solver:** all accepted updates non-increasing under their own explicit energy in all five dynamic conditions.

**G2 — content:** new local retrieval beats **masked_mean** on mean pixel MSE across 100 test birds. Also compare against whole-image donor retrieval. Report confidence intervals and both outcomes even when they fail.

**G3 — structural consistency:** learned Graph A has smaller mean reconstruction MSE than **both** local-memory-only and *matched shuffled Graph A*; report both subcriteria. Stronger evidence requires the **upper endpoint of the paired 95% bootstrap CI for learned minus shuffled to be below zero**. No changing binding coefficients or scoring conditions after test results.

**G4 — purpose contrast:** adding Graph B raises average **actual teacher** margin compared with local binding alone; report the simultaneous change in actual pixel MSE. Higher margin is not a repair success if MSE increases. Graph B validation R² must be reported regardless of sign.

**G5 — repeatability:** exactly the first 100 originally recognized birds, one fixed seed, checkpoint-bound run manifest, one receipt per bird, no test scores used for choosing donors, coefficients or model settings. This does not prove robustness to future image distributions; further fresh-seed replication is required.

## Limits

The masks delete random *spectral patches*, not a conventional solid image hole. A local coarse/fine context query can still be weak or ambiguous if most related residues are hidden. Raw donor patches may introduce seams or incorrect texture. Local memory remains finite, and neither Graph A nor B is a calibrated image likelihood. These tests are narrow and must not be marketed as medical-reconstruction verification, industrial anomaly detection, a CNN replacement, or strong generative modeling.

## Run

~~~powershell
git pull
python3.13 -m unittest discover -s tests -v
python3.13 run_v2.py --sihti-root ..\SihtiGraphModel --device cuda
~~~

Default output: ignored local runs/v2_local_100_birds/results.json, plus one saved bird_*.json per example, manifest.json and first_visual.json. The HTML viewer can read the optional docs/receipt_v2.json generated after completion. To do a fast smoke test, use a different --out path and --eval-birds 12; this **does not replace** the locked 100-bird result. A rerun with unchanged arguments loads all saved per-bird receipts; altered checkpoint or parameters require a new output directory.
