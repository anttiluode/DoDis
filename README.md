# DoDis — What was there / What fits / What the model wants

**[Open GitHub Pages](https://anttiluode.github.io/DoDis/)** · [Locked experimental protocol](PREREG.md)

DoDis connects three roles. **Content** is exact Sihti residues donated by other training images. **Consistency** is a graph learned from a separate set of bird images, with no CNN data. **Purpose** is an optional shared sparse graph distilled from the frozen CNN on yet another disjoint training split. This is an experimental image-repair system, not a claim of successful image generation.

The missing target pieces never enter retrieval, graph learning or settling. The solver exhaustively checks candidate memory fragments for each hidden piece and only accepts non-increasing-energy steps. The reconstructed image is scored against the withheld truth **after** settling, and the actual CNN margin is a separate secondary outcome.

## Windows RTX 3060 run

Keep the previous SihtiGraphModel repository (with its saved trained checkpoint and downloaded CIFAR-10 dataset) alongside DoDis, in sibling folders.

~~~powershell
cd E:\DocsHouse\1025
git clone https://github.com/anttiluode/DoDis.git
cd DoDis
python3.13 -m pip install -r requirements.txt
python3.13 -m unittest discover -s tests -v
python3.13 run.py --sihti-root ..\SihtiGraphModel --device cuda
~~~

DoDis does not train or replace the CNN, and does not download CIFAR again. The default experiment stores other-image candidate content from 256 training birds, a binding graph from 320 disjoint training birds, a purpose graph from 600 training images (with 140 different validation images), and scores 12 unseen correctly recognized test birds after exactly half of their 64 Sihti residues are sampled missing (p=.5 independently per piece).

## Results and GitHub Pages

The runner writes:

- runs/v1/results.json — every held-out bird, the four controlled conditions, reconstruction MSE, true classifier margins, graph quality, and exact energy traces.
- docs/receipt.json — the first actual bird's images/trace and measured aggregate numbers, readable by the browser viewer.

Open index.html and load the local receipt with the file picker. To publish measured results on the GitHub Pages site, commit and push that generated receipt:

~~~powershell
git add docs/receipt.json
git commit -m "Publish real DoDis v1 experimental receipt"
git push
~~~

In repository Settings → Pages, select Deploy from a branch, main, /(root). The root index.html is self-contained and requires no build tooling. It displays NO invented CIFAR results when no receipt is present.

## Controlled experiment

| Condition | Memory | Boundary smoothness | Binding from other images | CNN graph |
|---|---|---|---|---|
| Memory + smoothness | Yes | Yes | No | No |
| + learned binding | Yes | Yes | Yes | No |
| + shuffled binding control | Yes | Yes | Shuffled weights | No |
| + learned binding + purpose | Yes | Yes | Yes | Yes |

The same candidate content, observed target, and visit order are used for every condition. The optional teacher surrogate is trained across images, not copied from SihtiGraphModel's v3 per-photo switch polynomial. If Graph B raises the CNN bird margin while increasing reconstruction error, this is an exposed failure of classifier-purpose as an image prior, not a successful repair.

**Primary evidence** is paired held-out reconstruction error against the memory-only and shuffled-binding controls. The user screenshot of Living Paper motivates the separation between content and checking, but this is a batch CIFAR/multiscale-residue test, not yet a recurrent webcam memory system or conventional rectangular-hole inpainting.

See [PREREG.md](PREREG.md) for fixed parameters, roles, leakage restrictions and the predeclared failure conditions.
