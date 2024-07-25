import argparse
from glob import glob
import os
from PIL import Image
import numpy as np
import astra
from tensorflow.keras.models import load_model

# Function Definitions
def create_projector(geom, numbin, angles, dso, dod, fan_angle):
    if geom == 'parallel':
        proj_geom = astra.create_proj_geom(geom, 1.0, numbin, angles)
    elif geom == 'fanflat':
        dso *= 10; dod *= 10;
        ft = np.tan(np.deg2rad(fan_angle / 2))
        det_width = 2 * (dso + dod) * ft / numbin
        proj_geom = astra.create_proj_geom(geom, det_width, numbin, angles, dso, dod)
    p = astra.create_projector('cuda', proj_geom, vol_geom)
    return p

def generateParsedArgs():
    parser = argparse.ArgumentParser(description='')
    # Add arguments (same as before)
    # ...
    return parser.parse_args()

def makePNG(f, outname):
    img = np.maximum(f, np.finfo(float).eps)
    img = (img.T / np.amax(f)) * 255
    img = np.round(img)
    img = Image.fromarray(img.astype('uint8')).convert('L')
    img.save(outname + '.png', 'png')
    return

def makeFLT(f, outname):
    img = np.float32(f)
    img = np.maximum(img, np.finfo(np.float32).eps)
    img.tofile(outname + '.flt')
    return

# Main
args = generateParsedArgs()
infile = args.infile
outfolder = args.outfolder
x0file = args.x0_file
psize = args.psize
numpix = args.numpix
numbin = args.numbin
numtheta = args.numtheta
ns = args.ns
numits = args.num_its
beta = args.beta
epsilon_target = args.epsilon_target
theta_range = args.theta_range
geom = args.geom
dso = args.dso
dod = args.dod
fan_angle = args.fan_angle
make_png = bool(args.make_png)
overwrite = bool(args.overwrite)
make_intermediate = bool(args.make_intermediate)

# Load DnCNN model
model = load_model('dncnn_model.h5')

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
    print("\nReconstructing " + head + ":")

    sino = np.fromfile(name, dtype='f').reshape(numtheta, numbin)
    f = np.zeros((numpix, numpix))

    dx = psizes[n] if isinstance(psizes, np.ndarray) else psizes
    etarget = epsilon_target[n] if isinstance(epsilon_target, np.ndarray) else epsilon_target

    for k in range(1, numits + 1):
        if (not overwrite) and os.path.exists(outname + str(k) + '_SART.flt'):
            break
        
        if (use_sup) and (k >= kmin) and ((k - kmin) % kstep == 0):
            print("Applying DnCNN before the next SART iteration...")
            f_out = model.predict(np.expand_dims(f, axis=(0, -1)))[0, :, :, 0]
            p = f_out - f
            pnorm = np.linalg.norm(p, 'fro') + eps
            print("pnorm: " + str(pnorm))
            if k == kmin:
                alpha = pnorm
            else:
                alpha *= gamma
            print("alpha: " + str(alpha) + '\n')
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
        print('Iteration #{0:d}: Residual = {1:1.4f}\n'.format(k, res))
        if res < etarget:
            print("Target residual for " + head + " of ", end='')
            print(str(etarget) + " reached!")
            break

    res_file.write("%f\n" % res)
    if use_sup:
        makeFLT(f, outname + str(k) + '_DnCNNsup')
        if make_png:
            makePNG(f, outname + str(k) + '_DnCNNsup')
    else:
        makeFLT(f, outname + str(k) + '_SART')
        if make_png:
            makePNG(f, outname + str(k) + '_SART')

print("\n\nExiting...")
for j in range(ns):
    astra.data2d.delete(D_id[j])
    astra.data2d.delete(M_id[j])
    astra.projector.delete(P[j])
res_file.close()
