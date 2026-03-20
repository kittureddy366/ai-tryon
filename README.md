# AI Try-On (Local)

This project supports:

- **Realtime webcam try-on** (desktop window): `main.py`
- **Offline/headless try-on** (image/video -> output file, no GUI): `scripts/headless_tryon.py`

## Install (Python)

Create/activate a venv, then install the basics:

- `pip install -U mediapipe opencv-python numpy`

Optional (only if you want the OpenGL renderer path):

- `pip install -U PyOpenGL glfw`

## Run (realtime webcam)

- `python main.py`

Quit with `Q` or `ESC`.

Options:

- `python main.py --triposr auto` (default; start fast, swap in OBJ when ready; fallback if TripoSR fails)
- `python main.py --triposr require` (wait for OBJ before rendering; stop try-on if TripoSR fails)
- `python main.py --triposr off` (skip TripoSR)
- `python main.py --no-refine` (disable refinement)
- `python main.py --esrgan` (slow; higher quality refine when available)

In the preview window:

- `r`: toggle refinement
- `space`: pause/resume (useful for analysis)
- `c`: capture a screenshot to `generated/captures/`

## Run (offline image/video)

Image:

- `python scripts/headless_tryon.py --input TripoSR/examples/police_woman.png --output generated/out.png --preset tshirt_blue`

Video:

- `python scripts/headless_tryon.py --input input.mp4 --output generated/out.mp4 --preset tshirt_blue --max-frames 300`

## TripoSR garment mesh (2D -> 3D OBJ)

The headless runner can generate a garment OBJ from a 2D garment image using TripoSR, then simulate/render that mesh:

- `python scripts/headless_tryon.py --input TripoSR/examples/police_woman.png --output generated/out_triposr.png --triposr require --triposr-image png-clipart-t-shirt-polo-shirt-clothing-sleeve-black-t-shirt-black-crew-neck-t-shirt-tshirt-fashion-thumbnail.png`

If TripoSR is not available on your machine, use `--triposr off` to force the fallback cloth grid.

### TripoSR weights

TripoSR needs pretrained weights (`model.ckpt`, ~1.6GB). They are not committed to git.

Download them into the expected local cache folder:

- `python scripts/download_triposr_weights.py`
