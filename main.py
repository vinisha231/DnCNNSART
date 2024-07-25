# Import required libraries and modules
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Subtract
from glob import glob
import astra
import argparse
from bm3d import bm3d

# DnCNN model definition
def DnCNN(depth=17, filters=64, image_channels=1, use_bn=True):
    input_layer = Input(shape=(None, None, image_channels), name='input')
    x = layers.Conv2D(filters, kernel_size=3, padding='same', activation='relu')(input_layer)
    for _ in range(depth-2):
        x = layers.Conv2D(filters=filters, kernel_size=3, padding='same')(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    output_layer = layers.Conv2D(filters=image_channels, kernel_size=3, padding='same')(x)
    output_layer = Subtract()([input_layer, output_layer])
    model = models.Model(inputs=input_layer, outputs=output_layer, name='DnCNN')
    return model

# Function to read .flt file
def read_flt_file(file_path, shape):
    with open(file_path, 'rb') as f:
        data = f.read()
        return np.array(struct.unpack('f' * (len(data) // 4), data)).reshape(shape)

# Function to save .flt file
def save_flt_file(file_path, data):
    with open(file_path, 'wb') as f:
        f.write(struct.pack('f' * data.size, *data.flatten()))

# Function to display images
def display_images(original, noisy, denoised):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title('Noisy Image')
    plt.imshow(noisy, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title('Denoised Image')
    plt.imshow(denoised, cmap='gray')
    plt.axis('off')
    plt.show()

# Function to create projectors
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

# Argument parser function
def generateParsedArgs():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--sino', dest='infile', default='.', help='input sinogram -- directory or single file')
    parser.add_argument('--out', dest='outfolder', default='.', help='output directory')
    parser.add_argument('--numpix', dest='numpix', type=int, default=512, help='size of volume (n x n)')
    parser.add_argument('--psize', dest='psize', default='', help='pixel size (float) OR file containing pixel sizes (string)')
    parser.add_argument('--numbin', dest='numbin', type=int, default=729, help='number of detector pixels')
    parser.add_argument('--ntheta', dest='numtheta', type=int, default=900, help='number of angles')
    parser.add_argument('--nsubs', dest='ns', type=int, default=1, help='number of subsets. must divide evenly into number of angles')
    parser.add_argument('--range', dest='theta_range', type=float, nargs=2, default=[0, 360], help='starting and ending angles (deg)')
    parser.add_argument('--geom', dest='geom', default='fanflat', help='geometry (parallel or fanflat)')
    parser.add_argument('--dso', dest='dso', type=float, default=100, help='source-object distance (cm) (fanbeam only)')
    parser.add_argument('--dod', dest='dod', type=float, default=100, help='detector-object distance (cm) (fanbeam only)')
    parser.add_argument('--fan_angle', dest='fan_angle', default=35, type=float, help='fan angle (deg) (fanbeam only)')
    parser.add_argument('--numits', dest='num_its', default=32, type=int, help='maximum number of iterations')
    parser.add_argument('--beta', dest='beta', default=1., type=float, help='relaxation parameter beta')
    parser.add_argument('--x0', dest='x0_file', default='', help='initial image (default: zeros)')
    parser.add_argument('--xtrue', dest='xtrue_file', default='', help='true image (if available)')
    parser.add_argument('--sup_params', dest='sup_params', type=float, nargs=4, help='superiorization parameters: k_min, k_step, gamma, bm3d_sigma')
    parser.add_argument('--epsilon_target', dest='epsilon_target', default=0., help='target residual value (float, or file with residual values)')
    parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./checkpoint', help='directory containing checkpoint for DnCnn')
    parser.add_argument('--make_png', dest='make_png', type=bool, default=False, help='whether or not you would like to generate .png files')
    parser.add_argument('--make_intermediate', dest='make_intermediate', type=bool, default=False, help='whether or not you would like to generate output files each iter')
    parser.add_argument('--overwrite', dest='overwrite', type=bool, default=True, help='whether you would like to reprocess preexisting files on export')
    return parser.parse_args()

# Main function
if __name__ == "__main__":
    # Parse Arguments & Initialize
    args = generateParsedArgs()

    infile = args.infile
    outfolder = args.outfolder
    x0file = args.x0_file
    xtruefile = args.xtrue_file
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

    # Superiorization parameters
    use_sup = False
    kmin, kstep, gamma, sigma, alpha = 10, 1, 0, 0, 1  # Set kmin to 10 and kstep to 1
    if args.sup_params:
        use_sup = True
        kmin, kstep, gamma, sigma = map(float, args.sup_params)

    eps = np.finfo(float).eps

    fnames = sorted(glob(infile + '/*.flt')) if os.path.isdir(infile) else [infile]

    psizes = float(psize) if not psize.endswith('.txt') else np.loadtxt(psize, dtype='f')

    epsilon_target = float(epsilon_target) if not epsilon_target.endswith('.txt') else np.loadtxt(epsilon_target, dtype='f')

    vol_geom = astra.create_vol_geom(numpix, numpix)
    theta_range = np.deg2rad(theta_range)
    angles = theta_range[0] + np.linspace(0, numtheta-1, numtheta, False) * (theta_range[1]-theta_range[0]) / numtheta

    calc_error = False

    P, Dinv, D_id, Minv, M_id = [None]*ns, [None]*ns, [None]*ns, [None]*ns, [None]*ns
    for j in range(ns):
        ind1 = range(j, numtheta, ns)
        p = create_projector(geom, numbin, angles[ind1], dso, dod, fan_angle)
        D_id[j], Dinv[j] = astra.create_backprojection(np.ones((numtheta//ns, numbin)), p)
        M_id[j], Minv[j] = astra.create_sino(np.ones((numpix, numpix)), p)
        Dinv[j] = np.maximum(Dinv[j], eps)
        Minv[j] = np.maximum(Minv[j], eps)
        P[j] = p

    for i in range(len(fnames)):
        sino = read_flt_file(fnames[i], (numtheta, numbin))

        x0 = read_flt_file(x0file, (numpix, numpix)) if x0file else np.zeros((numpix,numpix), dtype=‘f’)
        xtrue = read_flt_file(xtruefile, (numpix, numpix)) if xtruefile else np.zeros((numpix, numpix), dtype=‘f’)
        x = x0

    for k in range(numits):
        for j in range(ns):
            ind1 = range(j, numtheta, ns)
            sino_j = sino[ind1, :]
            sino_j += np.random.normal(scale=psizes, size=sino_j.shape)
            proj_id, projdata = astra.create_sino(x, P[j])
            astra.data2d.store(proj_id, sino_j)
            bp_id, bpdata = astra.create_backprojection(projdata, P[j])
            astra.data2d.store(bp_id, bpdata)

            epsilon = np.zeros(numits)
            for _ in range(numits):
                _, res = astra.create_sino(x, P[j])
                epsilon[_] = np.linalg.norm(res)
            x = x + beta * np.mean(epsilon)

    # Load pre-trained DnCNN model for denoising
    dncnn_model = DnCNN()
    dncnn_model.load_weights(args.ckpt_dir)

    # Use DnCNN for denoising
    noisy_img = read_flt_file(fnames[i], (numpix, numpix))
    noisy_img_reshaped = np.reshape(noisy_img, (1, numpix, numpix, 1))
    denoised_img = dncnn_model.predict(noisy_img_reshaped)
    denoised_img = np.reshape(denoised_img, (numpix, numpix))

    denoised_img = denoised_img.astype(np.float32)
    save_flt_file(os.path.join(outfolder, os.path.basename(fnames[i])), denoised_img)

    # Display the images
    display_images(xtrue, noisy_img, denoised_img)

    print(f'Successfully processed: {fnames[i]}')
