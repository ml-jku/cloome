import io
import os
import re
import glob
from tqdm import tqdm
import zipfile
import itertools
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from training.datasets import CellPainting


def img_to_numpy(file):
    img = Image.open(file)
    arr = np.array(img)
    return arr


def numpy_to_img(arr, outfile, outdir="."):
    img = Image.fromarray(arr)
    img.save(outfile)
    return


def illumination_threshold(arr, perc=0.0028):
    """ Return threshold value to not display a percentage of highest pixels"""

    perc = perc/100

    h = arr.shape[0]
    w = arr.shape[1]

    # find n pixels to delete
    total_pixels = h * w
    n_pixels = total_pixels * perc
    n_pixels = int(np.around(n_pixels))

    # find indexes of highest pixels
    flat_inds = np.argpartition(arr, -n_pixels, axis=None)[-n_pixels:]
    inds = np.array(np.unravel_index(flat_inds, arr.shape)).T

    max_values = [arr[i, j] for i, j in inds]

    threshold = min(max_values)

    return threshold


def sixteen_to_eight_bit(arr, display_max, display_min=0):
    threshold_image = ((arr.astype(float) - display_min) * (arr > display_min))

    scaled_image = (threshold_image * (256. / (display_max - display_min)))
    scaled_image[scaled_image > 255] = 255

    scaled_image = scaled_image.astype(np.uint8)

    return scaled_image


def process_image(arr):
    threshold = illumination_threshold(arr)
    scaled_img = sixteen_to_eight_bit(arr, threshold)
    return scaled_img


def group_samples(indir):
    dirlist = glob.glob(os.path.join(indir, "*"))

    basenames = [os.path.basename(d) for d in dirlist]

    plate_groups = [list(g) for _, g in itertools.groupby(sorted(basenames), lambda x: x[0:5])]

    fullpath_groups = []
    basenames_groups = []

    order = [1,2,4,0,3]

    for g in plate_groups:
        fullpath_group = []
        basenames_group = []
        for f in g:
            fullpath_group.append(os.path.join(indir, f))
            basenames_group.append(f)
        fullpath_groups.append(fullpath_group)
        basenames_groups.append(basenames_group)

    sample_list = []

    for i, plate in enumerate(fullpath_groups):
        plate_id = basenames_groups[i][0][0:5]

        plate_files = []
        for channel in plate:
            z = zipfile.ZipFile(channel)

            for f in z.namelist():
                if f.endswith(".tif"):
                    plate_files.append(f)

        #plate_files = [os.path.join(dirname, f) for f in plate_files]
        sample_groups = [list(g) for _, g in itertools.groupby(sorted(plate_files, key=lambda x: x[-49:-43]), lambda x: x[-49:-43])]

        for g in sample_groups:
            ordered_group = [x for _, x in sorted(zip(order, g))]
            sample_list.append(ordered_group)

    return sample_list


def process_sample(imglst, indir, outdir="."):

    sample = np.zeros((520, 696, 5), dtype=np.uint8)

    refimg = imglst[0]
    pattern = re.compile(".*(?P<plate>\d{5})\-(?P<channel>\w*).*\/.*\_(?P<well>\w\d{2})\_\w(?P<sample>\d).*")
    ref_matches = pattern.match(refimg)
    plate, well, sampleid = ref_matches["plate"], ref_matches["well"], ref_matches["sample"]
    well = well.upper()

    filenames, channels = {}, {}

    for i, imgfile in enumerate(imglst):
        dirname = os.path.dirname(imgfile)
        basename = os.path.basename(imgfile)
        base, ext = os.path.splitext(basename)

        zipname = os.path.join(indir, dirname+".zip")

        z = zipfile.ZipFile(zipname)
        data = z.read(imgfile)
        dataenc = io.BytesIO(data)

        arr = img_to_numpy(dataenc)
        scaled_arr = process_image(arr)

        sample[:,:,i] = scaled_arr

        matches = pattern.match(imgfile)
        channel = matches["channel"]

        channels[i] = channel
        filenames[channel] = base

    outfile = str(plate)+"-"+str(well)+"-"+str(sampleid)
    outpath = os.path.join(outdir, outfile)
    np.savez(outpath, sample=sample, channels=channels, filenames=filenames)

    return


def save_arr(filename, outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)


def get_mean_std(loader, outfile):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for batch in tqdm(loader):
        images = batch
        images = images["input"]
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    with open(outfile, "w") as f:
        f.write(f"Mean:{mean}\n")
        f.write(f"Std:{std}")

    return mean, std


def get_dataloader(index_file, input_filename_imgs, batch_size):
    assert input_filename_imgs

    dataset = CellPainting(
        index_file,
        input_filename_imgs,
        transforms = ToTensor(),
        )
    num_samples = len(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return dataloader


def get_data(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data_imgs:
        data["train"] = get_dataloader(args, is_train=True)
    if args.val_data_imgs:
        data["val"] = get_dataloader(args, is_train=False)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")
    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data


if __name__ == '__main__':
    indir = "/<path-to-your-folder>/cellpainting/tiffs"
    outdir = "/<path-to-your-folder>/cellpainting_full/npzs/"
    n_cpus = 60

    index_file = "/<path-to-your-folder>/cellpainting-index.csv"
    input_imgs = "/publicdata/cellpainting/npzs/chembl24"
    input_mols = "/<path-to-your-folder>/morgan_fps_1024.hdf5"
    batchsize = 32


    sample_groups = group_samples(indir)
    result = parallelize(process_sample, sample_groups, n_cpus, indir=indir, outdir=outdir)
    
#     dataloader = get_dataloader(index_file, input_imgs, batchsize)
#     mean, std = get_mean_std(dataloader, stats_file)

