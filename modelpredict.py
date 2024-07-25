import argparse
import os
import numpy as np
from PIL import Image
from glob import glob
import astra
from tensorflow.keras.models import load_model

# Function Definitions
def create_projector(geom, numbin, angles, dso, dod, fan_angle):
    """Create projector based on geometry."""
    if geom == 'parallel':
        proj_geom = astra.create_proj_geom(geom, 1.0, numbin, angles)
    elif geom == 'fanflat':
        dso *= 10; dod *= 10
        ft = np.tan(np.deg2rad(fan_angle / 2))
        det_width = 2 * (dso + dod) * ft / numbin
        proj_geom = astra.create_proj_geom(geom, det_width, numbin, angles, dso, dod)
    else:
        raise ValueError(f"Unsupported geometry type: {geom}")
    return astra.create_projector('cuda', proj_geom, vol_geom)

def generateParsedArgs():
    """Generate and parse command-line arguments."""
    parser = argparse.ArgumentParser(description='SART-DnCNN Reconstruction')
    parser.add_argument('--sino', type=str, required=True, help='Path to sinogram directory')
    parser.add_argument('--out', type=str, required=True, help='Path to output directory')
    parser.add_argument('--psize', type=str, required=True, help='Pixel size or file containing pixel size')
    parser.add_argument('--numits', type=int, required=True, help='Number of iterations')
    parser.add_argument('--β', type=float, required=True, help='Regularization parameter')
    parser.add_argument('--epsilon_target', type=str, required=True, help='Target epsilon value or file containing epsilon values')
    parser.add_argument('--x0', type=str, required=True, help='Path to initial image directory')
    parser.add_argument('--xtrue', type=str, required=True, help='Path to true image directory')
    parser.add_argument('--sup_params', type=float, nargs=4, required=True, help='Superparameters (e.g., 10 1 0.5 25)')
    parser.add_argument('--make_png', type=str, choices=['True', 'False'], required=True, help='Save output as PNG')
    parser.add_argument('--overwrite', type=str, choices=['True', 'False'], required=True, help='Overwrite existing files')
    parser.add_argument('--make_intermediate', type=str, choices=['True', 'False'], required=True, help='Save intermediate files')
    parser.add_argument('--model_weights', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--geom', type=str, required=True, help='Geometry type (parallel or fanflat)')
    parser.add_argument('--dso', type=float, required=True, help='Source to object distance')
    parser.add_argument('--dod', type=float, required=True, help='Source to detector distance')
    parser.add_argument('--fan_angle', type=float, required=True, help='Fan angle in degrees')
    parser.add_argument('--numpix', type=int, required=True, help='Number of pixels in each dimension')
    parser.add_argument('--numbin', type=int, required=True, help='Number of detector bins')
    parser.add_argument('--numtheta', type=int, required=True, help='Number of projection angles')
    parser.add_argument('--theta_range', type=float, nargs=2, required=True, help='Range of theta angles in degrees')
    parser.add_argument('--ns', type=int, required=True, help='Number of subsets')
    parser.add_argument('--kmin', type=int, default=1, help='Minimum iteration for applying DnCNN')
    parser.add_argument('--kstep', type=int, default=1, help='Step size for applying DnCNN')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma parameter for DnCNN')

    return parser.parse_args()

def makePNG(f, outname):
    """Save numpy array as PNG image."""
    img = np.maximum(f, np.finfo(float).eps)
    img = (img.T / np.amax(f)) * 255
    img = np.round(img).astype(np.uint8)
    Image.fromarray(img).save(outname + '.png', 'PNG')

def makeFLT(f, outname):
    """Save numpy array as FLT file."""
    img = np.float32(f)
    img = np.maximum(img, np.finfo(np.float32).eps)
    img.tofile(outname + '.flt')

# Main script execution
args = generateParsedArgs()

# Paths and parameters
infile = args.sino
outfolder = args.out
x0file = args.x0
psize = args.psize
numpix = args.numpix
numbin = args.numbin
numtheta = args.numtheta
ns = args.ns
numits = args.numits
beta = args.β
epsilon_target = args.epsilon_target
theta_range = args.theta_range
geom = args.geom
dso = args.dso
dod = args.dod
fan_angle = args.fan_angle
make_png = args.make_png == 'True'
overwrite = args.overwrite == 'True'
make_intermediate = args.make_intermediate == 'True'
model_weights = args.model_weights
kmin = args.kmin
kstep = args.kstep
gamma = args.gamma

# Load DnCNN model
model = load_model(model_weights)

eps = np.finfo(float).eps
fnames = sorted(glob(infile + '/*.flt')) if os.path.isdir(infile) else [infile]
psizes = float(psize) if not psize.endswith('.txt') else np.loadtxt(psize, dtype='f')
epsilon_target = float(epsilon_target) if not epsilon_target.endswith('.txt') else np.loadtxt(epsilon_target, dtype='f')

vol_geom = astra.create_vol_geom(numpix, numpix)
theta_range = np.deg2rad(theta_range)
angles = theta_range[0] + np.linspace(0, numtheta - 1, numtheta, False) * (theta_range[1] - theta_range[0]) / numtheta

P, Dinv, D_id, Minv, M_id = [None] * ns, [None] * ns, [None] * ns, [None] * ns, [None] * ns
for j in range(ns):
    ind1 = range(j, numtheta, ns)
    p = create_projector(geom, numbin, angles[ind1], dso, dod, fan_angle)
    D_id[j], Dinv[j] = astra.create_backprojection(np.ones((numtheta // ns, numbin)), p)
    M_id[j], Minv[j] = astra.create_sino(np.ones((numpix, numpix)), p)
    Dinv[j] = np.maximum(Dinv[j], eps)
    Minv[j] = np.maximum(Minv[j], eps)
    P[j] = p

res_file = open(outfolder + "/residuals.txt", "w+")
for n in range(len(fnames)):
    name = fnames[n]
    head, tail = os.path.split(name)
    head, tail = tail.split("_", 1)
    outname = outfolder + "/" + head + "_recon_"
    print(f"\nReconstructing {head}:")

    sino = np.fromfile(name, dtype='f').reshape(numtheta, numbin)
    f = np.zeros((numpix, numpix))

    dx = psizes[n] if isinstance(psizes, np.ndarray) else psizes
    etarget = epsilon_target[n] if isinstance(epsilon_target, np.ndarray) else epsilon_target

    for k in range(1, numits + 1):
        if (not overwrite) and os.path.exists(outname + str(k) + '_SART.flt'):
            break
        
        if k >= kmin and (k - kmin) % kstep == 0:
            print("Applying DnCNN before the next SART iteration...")
            f_out = model.predict(np.expand_dims(f, axis=(0, -1)))[0, :, :, 0]
            p = f_out - f
            pnorm = np.linalg.norm(p, 'fro') + eps
            print(f"pnorm: {pnorm}")
            if k == kmin:
                alpha = pnorm
            else:
                alpha *= gamma
            print(f"alpha: {alpha}\n")
            if pnorm > alpha:
                p = alpha * p / (np.linalg.norm(p, 'fro') + eps)
                f = f + p
            else:
                f = f_out
            if make_intermediate:
                makeFLT(f, outname + str(k) + '_dncnn_sup')
                if make_png:
                    makePNG(f, outname + str(k) + '_dncnn_sup')

        for j in range(ns):
            ind1 = range(j, numtheta, ns)
            p = P[j]
            fp_id, fp = astra.create_sino(f, p)
            diffs = (sino[ind1, :] - fp * dx) / Minv[j] / dx
            bp_id, bp = astra.create_backprojection(diffs, p)
            ind2 = np.abs(bp) > 1e3
            bp[ind2] = 0
        f = f + beta * bp / Dinv[j]
        astra.data2d.delete(fp_id)
        astra.data2d.delete(bp_id)

    if make_intermediate:
        makeFLT(f, outname + str(k) + '_SART')
        if make_png:
            makePNG(f, outname + str(k) + '_SART')

    fp = np.zeros((numtheta, numbin))
    for j in range(ns):
        ind = range(j, numtheta, ns)
        p = P[j]
        fp_tempid, fp_temp = astra.create_sino(f, p)
        fp[ind, :] = fp_temp * dx
        astra.data2d.delete(fp_tempid)
    res = np.linalg.norm(fp - sino, 'fro')
    print(f'Iteration #{k:d}: Residual = {res:1.4f}\n')
    if res < etarget:
        print(f"Target residual of {etarget} reached!")
        break

res_file.write(f"{res:.6f}\n")
if make_intermediate:
    makeFLT(f, outname + str(k) + '_DnCNNsup' if k >= kmin and (k - kmin) % kstep == 0 else outname + str(k) + '_SART')
    if make_png:
        makePNG(f, outname + str(k) + '_DnCNNsup' if k >= kmin and (k - kmin) % kstep == 0 else outname + str(k) + '_SART')
print(”\n\nExiting…”)
for j in range(ns):
    astra.data2d.delete(D_id[j])
    astra.data2d.delete(M_id[j])
    astra.projector.delete(P[j])
res_file.close()
