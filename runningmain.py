import argparse
import pdb
from glob import glob
import os
from PIL import Image
import numpy as np
import astra
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation

# Function to define the DnCNN model architecture
def create_dncnn_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(None, None, 1)))
    model.add(Activation('relu'))
    for _ in range(15):
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    model.add(Conv2D(filters=1, kernel_size=(3, 3), padding='same'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

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
    parser.add_argument('--sino',dest='infile', required=False, default='/mmfs1/gscratch/uwb/CT_images/sinos2024/60views', help='Path to sinogram files')
    #parser.add_argument('--infile', type=str, required=False, default='/mmfs1/gscratch/uwb/CT_images/RECONS2024/900views', help='Path to the input file or directory')
    parser.add_argument('--out', required=False, default='/mmfs1/gscratch/uwb/vdhaya/SARTDnCNN/output', help='Output directory')
    parser.add_argument('--psize', required=False, default=0.0568, help='Pixel size or path to pixel size file')
    parser.add_argument('--numits', required=False, type=int, default=1000, help='Number of iterations')
    parser.add_argument('--beta', required=False, type=float, default=1.9, help='Regularization parameter')
    parser.add_argument('--epsilon_target', required=False, default='/mmfs1/gscratch/uwb/CT_images/recons2024/60views/residuals.txt', help='Target residual or path to target residual file')
    parser.add_argument('--x0', required=False, default=None, help='Path to initial images')
    parser.add_argument('--xtrue', required=False, default='/mmfs1/gscratch/uwb/CT_images/recons2024/900views/', help='Path to true images')
    parser.add_argument('--sup_params', required=False, nargs=3, type=float, default=[10, 4, 0.95], help='Superparameters for DnCNN')
    parser.add_argument('--make_png', required=False, choices=['True', 'False'], default='False', help='Whether to save as PNG')
    parser.add_argument('--overwrite', required=False, choices=['True', 'False'], default='False', help='Whether to overwrite existing files')
    parser.add_argument('--make_intermediate', required=False, choices=['True', 'False'], default='False', help='Whether to save intermediate files')
    parser.add_argument('--model_weights', required=False, default='/mmfs1/gscratch/uwb/vdhaya/dncnn_model.weights.h5', help='Path to the trained model weights file')
    parser.add_argument('--geom', required=False, default='fanflat', help='Geometry type (e.g., fanflat, parallel)')
    parser.add_argument('--dso', required=False, type=float, default=100, help='Distance source to object')
    parser.add_argument('--dod', required=False, type=float, default=100, help='Distance object to detector')
    parser.add_argument('--fan_angle', required=False, type=float, default=35.0, help='Fan angle')
    parser.add_argument('--numpix', required=False, type=int, default=512, help='Number of pixels')
    parser.add_argument('--numbin', required=False, type=int, default=729, help='Number of bins')
    parser.add_argument('--numtheta', required=False, type=int, default=60, help='Number of angles')
    parser.add_argument('--theta_range', required=False, nargs=2, type=float, default=[0, 360], help='Range of angles')
    parser.add_argument('--ns', required=False, type=int, default=12, help='Number of subsets')
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
outfolder = args.out
x0file = args.x0
psize = args.psize
numpix = args.numpix
numbin = args.numbin
numtheta = args.numtheta
ns = args.ns
numits = args.numits
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
use_sup = False
kmin = 0    #Iteration at which superiorization begins
kstep = 0   #Interval of SARTS between each superiorization step
gamma = 0   #Geometric attenuation factor for superiorization
alpha = 1   #Computed attenuation factor for superiorization, not an arg
if not (args.sup_params is None):
    use_sup = True
    kmin = int(args.sup_params[0])
    kstep = int(args.sup_params[1])
    gamma = args.sup_params[2]
# Load DnCNN model
#model = create_dncnn_model()
#model.load_weights('/mmfs1/gscratch/uwb/vdhaya/dncnn_model.weights.h5')
model = load_model('../dncnn_model.h5')
eps = np.finfo(float).eps
#fnames = sorted(glob(x0file + '/*.flt')) if os.path.isdir(x0file) else [x0file]
fnames = []
if os.path.isdir(infile):
    fnames = sorted(glob(infile + '/*.flt'))
#Otherwise, a single filename was provided
else:
    fnames.append(infile)
#psizes = float(psize) if not psize.endswith('.txt') else np.loadtxt(psize, dtype='f')
# Check if psize is a path or a number
if isinstance(psize, str):
    if psize.endswith('.txt'):
        psizes = np.loadtxt(psize, dtype='f')
    else:
	psizes = float(psize)
else:
    psizes = float(psize)
#epsilon_target = float(epsilon_target) if not epsilon_target.endswith('.txt') else np.loadtxt(epsilon_target, dtype='f')
# Check if epsilon_target is a path or a number
if isinstance(epsilon_target, str):
    if epsilon_target.endswith('.txt'):
        epsilon_target = np.loadtxt(epsilon_target, dtype='f')
    else:
	epsilon_target = float(epsilon_target)
else:
    epsilon_target = float(epsilon_target)
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
print(f"Attempting to open file: {outfolder + '/residuals.txt'}")
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
            #pdb.set_trace()
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
            make_intermediate = False
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
        f=np.maximum(f,eps)
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
