# DoDis v1 — content / consistency / purpose (2026-10-23)

**Precommitted before the real CIFAR run.** This is a separate experiment, not a rewrite of SihtiGraphModel's v1–v3 results. It tests a memory-based reconstruction task; it must not be judged on CNN margin alone.

## What is fixed

- SihtiGraphModel's frozen 8-epoch CIFAR-10 TinyCNN checkpoint (best epoch 7) is reused, with no retraining or modification. The original 4 Laplacian-like octave residues subdivided into 4×4 8-pixel tiles form exactly **64 additive components**. Full reconstruction has max absolute error ≤2e−6 before clamping. Base is the RGB CIFAR mean.
- Work with **held-out CIFAR-10 test birds correctly classified by the CNN**, selecting the first 12 from test index ≥100. Do not choose on reconstruction performance.
- Make each 64-piece **mask independently from RNG(seed=41, test index, 774), p=.5**; observed pieces are fixed. This is a multiscale-residue mask, not a contiguous black-pixel hole.
- Candidate content is exact donor-residue material from **256 training-set birds**, with 4 nearest donors selected by comparing only observed pieces. One extra candidate is blank/mean-fill (not the target). Hidden target pieces are inaccessible to retrieval, optimization, and graphs; they are used only for final scoring.
- Graph A (**consistency**) learns 80 pairwise 6×6 feature co-occurrence matrices from a *disjoint* collection of **320 training-set birds**. Candidate edges include spatial 4-neighbors within an octave, adjacent-octave same-tile connections, and B0↔B3 same-tile connections. Rank by same-image covariance contrasted against shuffled-image covariance; normalize and optimize a bounded pair score. No CNN predictions/labels enter its training.
- Graph B (**purpose**) distills the CNN's moving bird-vs-best-other logit margin from training images distinct from memory and Graph A. Training set: 380 bird + 220 nonbird images, with four independent mask conditions p=.5/.75/.9/1; validation set: 70 bird + 70 nonbird images. Graph B is a shared ridge regression on per-tile 6-feature unary contributions and pairwise RMS interactions at Graph A's 80 learned edges, validated on disjoint training images. **Not a transplant of v3's graph coefficients, which were specific to bird #65/#70.** Store its validation R² even when negative.
- Four energies, identical candidate pools, same initialization, same visible constraints: (1) retrieved-memory rank penalty + seam smoothness, (2) plus learned Graph A, (3) plus a permuted-weight Graph A control, (4) plus Graph A and −λ*(Graph B margin / Graph B training margin SD). Fixed weights: β=0.05, seam=0.75 times 20× seam MSE, γ=0.40, λ=0.12. One-piece exhaustive coordinate descent, at most 3 asynchronous sweeps, same seeded visiting order. No CNN calls in the inner settling loop.

## Measurements and locked verdicts

For every evaluated test bird and every condition:
1. Overall **pixel MSE** between reconstructed full RGB image and withheld target; also hidden-residue tensor MSE separately. The original image is never used to select candidates, estimate energy, or tune weights. Display both, do not call whole-image MSE “contiguous-hole pixel error.”
2. Frozen-CNN bird margin on *completed* image, compared with margin on corrupted input and on undamaged truth; report whether a higher margin accompanies **worse** reconstruction.
3. Exact number of accepted piece updates and a complete energy trajectory. Every accepted move must satisfy ΔE≤1e−8. Observed residues remain bitwise unchanged.
4. Binding-vs-memory-only **mean reconstruction MSE**, paired per bird; shuffled-binding control; and CNN-graph-vs-binding-only mean CNN margin and mean reconstruction MSE. Do not choose the best settings using held-out outcomes.
5. Graph B validation R² on held-out training images and donor image provenance. Report any bad teacher surrogate separately rather than claiming reliable classifier control.

**G0 source:** all 12 test birds correctly classified originally; reconstructed all-on sieve matches truth; disjoint training roles, no hidden target used by candidate retrieval.

**G1 mechanical:** all accepted updates non-increasing in total energy, all visible pieces preserved.

**G2 content/consistency:** mean test-image pixel MSE with Graph A strictly below memory+smoothness **and** Graph A beats its shuffled-weight control in mean MSE. Report each criterion individually; either may fail. A negative result is still a valid result.

**G3 purpose disagreement:** Graph B increases mean *actual teacher* margin versus Graph A and may increase or decrease pixel MSE. Report both numbers and per-image conflicts. This is not a requirement for better reconstruction and is not a “successful repair” if pixels worsen.

**G4 generalization:** no held-out CIFAR test image, true hidden residue, or original-model margin is used for training the memory, binding, or purpose readout.

## Interpretation

The local memory contains real training-image content, but a donor fragment is not evidence of original target content. Observed image is an unusually artificial 50% spectral mask, unlike a camera hole. Graph A is a statistical covariance model, not biological synaptic learning. Graph B uses CNN teacher labels/scores and is an *optional* discriminative objective, not a generative likelihood. Coordinate descent monotonicity is about the **explicit energy**; it does not imply lower pixel error or recovery of missing details. There is no basis to claim state-of-the-art image inpainting, novel explainability, real-time webcam, or general scenes without further experiments.

## Command

Place DoDis next to SihtiGraphModel so the latter's frozen CNN code, trained checkpoint, and CIFAR-10 data remain unchanged:

~~~powershell
git clone https://github.com/anttiluode/DoDis.git
cd DoDis
python3.13 -m pip install numpy torch torchvision Pillow
python3.13 -m unittest discover -s tests -v
python3.13 run.py --sihti-root ../SihtiGraphModel --device cuda
~~~

Receipt: runs/v1/results.json, and first bird image/trace for the Pages viewer: docs/receipt.json. **The result will not appear on GitHub Pages until docs/receipt.json is committed**; the page does not fabricate a result in the meantime.

The page can also load an uncommitted local receipt via its file picker. Enable GitHub Pages source as branch main / root, then browse https://anttiluode.github.io/DoDis/.
