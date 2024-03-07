import os
import cfl
import argparse
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from joblib import Parallel, delayed


def train_files(file_):
    lr_file, hr_file = file_
    lr_filename = os.path.splitext(lr_file)[0]
    hr_filename = os.path.splitext(hr_file)[0]
    filename = os.path.split(lr_filename)[-1]
    
    if args.dataset_tag == '0001':
        lr_img = np.squeeze(cfl.read_cfl(lr_filename))#[15:-15, 15:-15, 15:-15] # dataset01, (180, 240, 240)
        hr_img = np.squeeze(cfl.read_cfl(hr_filename))#[15:-15, 15:-15, 15:-15] # dataset01, (180, 240, 240)
    elif args.dataset_tag == '2001':
        lr_img = np.squeeze(cfl.read_cfl(lr_filename)).transpose(2, 0, 1)[30:210, :, :] # dataset02, (180, 240, 240)
        hr_img = np.squeeze(cfl.read_cfl(hr_filename)).transpose(2, 0, 1)[30:210, :, :] # dataset02, (180, 240, 240)
    else:
        lr_img = np.squeeze(cfl.read_cfl(lr_filename))[20:-30, :, :] # stanford knee, (180, 240, 240)->(130, 240, 240)
        hr_img = np.squeeze(cfl.read_cfl(hr_filename))[20:-30, :, :] # stanford knee, (180, 240, 240)->(130, 240, 240)
    
    lr_img_mag = np.abs(lr_img)
    lr_img_ph = np.angle(lr_img)
    lr_img_mag_max = lr_img_mag.max()
    lr_img_mag_min = lr_img_mag.min()
    lr_img_mag = (lr_img_mag - lr_img_mag_min)/(lr_img_mag_max - lr_img_mag_min)
    lr_img = lr_img_mag * np.exp(1j*lr_img_ph)
    
    hr_img_mag = np.abs(hr_img)
    hr_img_ph = np.angle(hr_img)
    hr_img_mag = (hr_img_mag - lr_img_mag_min)/(lr_img_mag_max - lr_img_mag_min)
    hr_img = hr_img_mag * np.exp(1j*hr_img_ph)
    
    num_patch = 0
    z, w, h = lr_img.shape
    for k in range(z):
        if w > p_max and h > p_max:
            w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=int))
            h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=int))
            w1.append(w-patch_size)
            h1.append(h-patch_size)
            for i in w1:
                for j in h1:
                    num_patch += 1
                    
                    lr_patch = lr_img[k, i:i+patch_size, j:j+patch_size]
                    hr_patch = hr_img[k, i:i+patch_size, j:j+patch_size]
                    
                    lr_savename = os.path.join(lr_tar, filename + '_' + str(num_patch).zfill(4) + '.npy')
                    hr_savename = os.path.join(hr_tar, filename + '_' + str(num_patch).zfill(4) + '.npy')
                    
                    np.save(lr_savename, lr_patch)
                    np.save(hr_savename, hr_patch)

        else:
            lr_savename = os.path.join(lr_tar, filename + '_' + str(k).zfill(4) + '.npy')
            hr_savename = os.path.join(hr_tar, filename + '_' + str(k).zfill(4) + '.npy')

            lr_patch = lr_img[k, ...]#[k, i:i+val_patch_size, j:j+val_patch_size]
            hr_patch = hr_img[k, ...]#[k, i:i+val_patch_size, j:j+val_patch_size]
            
            np.save(lr_savename, lr_patch)
            np.save(hr_savename, hr_patch)


def val_files(file_):
    lr_file, hr_file = file_
    lr_filename = os.path.splitext(lr_file)[0]
    hr_filename = os.path.splitext(hr_file)[0]
    filename = os.path.split(lr_filename)[-1]
    lr_img = np.squeeze(cfl.read_cfl(lr_filename))
    hr_img = np.squeeze(cfl.read_cfl(hr_filename))
    
    lr_img_mag = np.abs(lr_img)
    lr_img_ph = np.angle(lr_img)
    lr_img_mag_max = lr_img_mag.max()
    lr_img_mag_min = lr_img_mag.min()
    lr_img_mag = (lr_img_mag - lr_img_mag_min)/(lr_img_mag_max - lr_img_mag_min)
    lr_img = lr_img_mag * np.exp(1j*lr_img_ph)
    
    hr_img_mag = np.abs(hr_img)
    hr_img_ph = np.angle(hr_img)
    hr_img_mag = (hr_img_mag - lr_img_mag_min)/(lr_img_mag_max - lr_img_mag_min)
    hr_img = hr_img_mag * np.exp(1j*hr_img_ph)

    num_slice = 0
    z, w, h = lr_img.shape
    # i = (w-val_patch_size)//2
    # j = (h-val_patch_size)//2
    for k in range(z):
        num_slice += 1
        lr_savename = os.path.join(lr_tar, filename + '_' + str(num_slice).zfill(4) + '.npy')
        hr_savename = os.path.join(hr_tar, filename + '_' + str(num_slice).zfill(4) + '.npy')

        lr_patch = lr_img[k, ...]#[k, i:i+val_patch_size, j:j+val_patch_size]
        hr_patch = hr_img[k, ...]#[k, i:i+val_patch_size, j:j+val_patch_size]

        np.save(lr_savename, lr_patch)
        np.save(hr_savename, hr_patch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2D Patch Cropping to make training/test set.')
    parser.add_argument('--src_dir', type=str, default='/raid/MRI/Cones_Phantoms_v2/augmented', help='read from source directory')
    parser.add_argument('--tar_dir', type=str, default='/raid/MRI/Cones_Phantoms_v2/patch3d', help='write to target directory')
    parser.add_argument('--val_dir', type=str, default='/raid/MRI/offres-dataset-01/20Apr2022_Ex5735_Ser10/cones.tmp-recon.0010/gridding.no-gating.0010', help='validation source directory')
    parser.add_argument('--dataset_tag', type=str, default='0001', help='dataset tag (0001, 2001, s001)')
    parser.add_argument('--target_tag', type=str, default='0001', help='target (regrid) tag (0001, 0002, ..., 0010)')
    parser.add_argument('--train', action='store_true', help='Make training set')
    parser.add_argument('--val_real', action='store_true', help='Make real validation set')
    parser.add_argument('--patch_size', type=int, default=200, help='patch size')
    parser.add_argument('--overlap', type=int, default=190, help='overlap size of neighbored patches')
    parser.add_argument('--num_cores', type=int, default=20, help='# of CPU cores for parallel mapping')
    parser.add_argument('--tmp', action='store_true', help='tmp')

    args, unknown = parser.parse_known_args()
    
    ############ Prepare Training data ####################
    if args.train:
        num_cores = args.num_cores
        patch_size = args.patch_size
        overlap = args.overlap
        p_max = 300

        src = args.src_dir
        tar = os.path.join(args.tar_dir, 'train')

        lr_tar = os.path.join(tar, 'input_crops')
        hr_tar = os.path.join(tar, 'target_crops')

        os.makedirs(lr_tar, exist_ok=True)
        os.makedirs(hr_tar, exist_ok=True)

        if 's' in args.dataset_tag:
            lr_files = natsorted(glob(os.path.join(src, '*' + args.dataset_tag + '.' + args.target_tag + '*.cfl')))
            hr_files = natsorted(glob(os.path.join(src, '*' + args.dataset_tag + '.' + args.target_tag + '*_cc_p000.cfl')))
            for i in range(len(hr_files)):
                lr_files.remove(hr_files[i])
        else:
            lr_files = natsorted(glob(os.path.join(src, '*' + args.dataset_tag + '*.cfl')))
            hr_files = natsorted(glob(os.path.join(src, '*' + args.dataset_tag + '*_cc_p000.cfl')))
            lr_files.remove(hr_files[0])
        hr_files = hr_files * len(lr_files)

        files = [(i, j) for i, j in zip(lr_files, hr_files)]

        Parallel(n_jobs=num_cores)(delayed(train_files)(file_) for file_ in tqdm(files))


    ############ Prepare validation data ####################
    elif args.val_real:
        num_cores = args.num_cores
        
        src = args.src_dir
        tar = os.path.join(args.tar_dir, 'val')

        lr_tar = os.path.join(tar, 'input_crops')
        hr_tar = os.path.join(tar, 'target_crops')

        os.makedirs(lr_tar, exist_ok=True)
        os.makedirs(hr_tar, exist_ok=True)

        lr_files = natsorted(glob(os.path.join(src, '*cc.cfl')))
        hr_files = natsorted(glob(os.path.join(src, '*001_cc.cfl')))
        for hr_file in hr_files:
            lr_files.remove(hr_file)
        print(lr_files)
        print(hr_files)
        
        for hr_file in hr_files:
            hr_file = [hr_file] * len(lr_files)
            print(hr_file)
            files = [(i, j) for i, j in zip(lr_files, hr_file)]
            print(files)

            Parallel(n_jobs=num_cores)(delayed(val_files)(file_) for file_ in tqdm(files))
    else:
        num_cores = args.num_cores
        
        src = args.src_dir
        tar = os.path.join(args.tar_dir, 'val')

        lr_tar = os.path.join(tar, 'input_crops')
        hr_tar = os.path.join(tar, 'target_crops')

        os.makedirs(lr_tar, exist_ok=True)
        os.makedirs(hr_tar, exist_ok=True)

        if 's' in args.dataset_tag:
            lr_files = natsorted(glob(os.path.join(src, '*' + args.dataset_tag + '.' + args.target_tag + '*.cfl')))
            hr_files = natsorted(glob(os.path.join(src, '*' + args.dataset_tag + '.' + args.target_tag + '*_cc_p000.cfl')))
            for i in range(len(hr_files)):
                lr_files.remove(hr_files[i])
        elif args.tmp:
            lr_files = natsorted(['/raid/MRI/offres-motion-01/02Nov2022_Ex7521_Ser4/cones.tmp-recon.0025/gridding.no-gating.0025/imout.0025_cc.cfl'])
            hr_files = natsorted(['/raid/MRI/offres-motion-01/02Nov2022_Ex7521_Ser3/cones.tmp-recon.0001/gridding.no-gating.0001/imout.0001_cc.cfl'])
        else:
            lr_files = natsorted(glob(os.path.join(args.val_dir, '*cc.cfl')))
            hr_files = natsorted(glob(os.path.join(src, '*' + args.dataset_tag + '_cc_p000.cfl')))
        hr_files = hr_files * len(lr_files)

        files = [(i, j) for i, j in zip(lr_files, hr_files)]
        # print(files)

        Parallel(n_jobs=num_cores)(delayed(val_files)(file_) for file_ in tqdm(files))
