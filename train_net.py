import os
import time
from datetime import datetime
import argparse
from tqdm import tqdm
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
import random
import numpy as np
from osgeo import gdal
import torch
import torch.nn.functional as F
from model import DeepNet
from segmentation import SaveLabelArrayInCompressMode

# functions and variables defined
MAX_PROCESS_COUNT = (multiprocessing.cpu_count()//2) or 1

def generate_seg_map_buffer(patch_info):
    y_offset, x_offset, y_buffersize, x_buffersize, seg_path = patch_info
    seg_map = gdal.Open(seg_path)    
    seg_map_buffer = seg_map.ReadAsArray(x_offset, y_offset, x_buffersize, y_buffersize).flatten()
    return [np.where(seg_map_buffer == u_label)[0] for u_label in np.unique(seg_map_buffer)], y_offset, x_offset
    
def train(args):
    input_image_path = "data/{}/image.tif".format(args.input)
    seg_path = "data/{}/seg.tif".format(args.input)
    result_id = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    x_buffersize, y_buffersize = args.buffersize, args.buffersize
    x_stride, y_stride = args.stride, args.stride

    '''load image'''
    src_ds = gdal.Open(input_image_path)
    channels, height, width = src_ds.RasterCount, src_ds.RasterYSize, src_ds.RasterXSize

    '''load seg_lab patches'''
    patches_info = []
    y_offset = 0
    while y_offset < height:
        is_y_last = False
        if y_offset + y_buffersize >= height:
            y_offset = height - y_buffersize
            is_y_last = True

        x_offset = 0
        while x_offset < width:
            is_x_last = False
            if x_offset + x_buffersize >= width:
                x_offset = width - x_buffersize                
                is_x_last = True            

            patches_info.append([
                y_offset, x_offset, y_buffersize, x_buffersize, seg_path
            ])

            if is_x_last:
                break
            x_offset += x_stride

        if is_y_last:
            break
        y_offset += y_stride                 


    '''Init segmap'''
    start_time1 = time.time()

    print(len(patches_info))

    dict_seg_lab = {}    
    with Pool(processes=MAX_PROCESS_COUNT) as pool:
        for seg_lab, y_offset, x_offset in list(tqdm(pool.imap_unordered(generate_seg_map_buffer, patches_info), total=len(patches_info))):
            dict_seg_lab[(y_offset, x_offset)] = seg_lab

    '''model init'''
    torch.cuda.manual_seed_all(2021)
    np.random.seed(2021)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # choose GPU:0
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(device)

    classes = args.classes
    model = DeepNet(inp_dim=channels, classes=classes).to(device)
    criterion_ce = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    '''train loop'''
    start_time2 = time.time()

    for batch_idx in range(args.epochs):
        '''forward'''
        model.train()

        shuffled_patches_info = random.sample(patches_info, len(patches_info))
        for y_offset, x_offset, y_buffersize, x_buffersize, _ in tqdm(shuffled_patches_info):            
            buffer = src_ds.ReadAsArray(x_offset, y_offset, x_buffersize, y_buffersize).astype(np.float32)
            tensor = torch.unsqueeze(torch.from_numpy(buffer).to(device)/ 255.0, 0) # np.newaxis

            optimizer.zero_grad()
            output = model(tensor)[0]
            output = output.permute(1, 2, 0).view(-1, args.classes)
            target = torch.argmax(output, 1)
            im_target = target.data.cpu().numpy()

            '''refine'''
            seg_lab = dict_seg_lab[(y_offset, x_offset)]
            for inds in seg_lab:
                u_labels, hist = np.unique(im_target[inds], return_counts=True)
                im_target[inds] = u_labels[np.argmax(hist)]

            '''backward'''
            target = torch.from_numpy(im_target).to(device)
            loss = criterion_ce(output, target)
            loss.backward()
            optimizer.step()                

        print('Loss:', batch_idx, loss.item())

        ''' Validation '''
        im_target = np.zeros((classes, height, width), dtype="uint8")
        model.eval()
        for y_offset, x_offset, y_buffersize, x_buffersize, _ in tqdm(patches_info):
            buffer = src_ds.ReadAsArray(x_offset, y_offset, x_buffersize, y_buffersize).astype(np.float32)
            tensor = torch.unsqueeze(torch.from_numpy(buffer).to(device)/ 255.0, 0) # np.newaxis
            output = F.softmax(model(tensor)[0], dim=0) * 63.
            im_target[:, y_offset:(y_offset+y_buffersize), x_offset:(x_offset+x_buffersize)] += output.data.cpu().numpy().astype("uint8")
        im_target = np.argmax(im_target, 0).astype("uint8")
        output_path = "results/{}/{}/epoch_{}.tif".format(args.input, result_id, batch_idx)
        Path(output_path).parent.mkdir(parents = True, exist_ok=True)
        SaveLabelArrayInCompressMode(im_target, output_path, seive_small_area=True)

        # if len(un_label) < args.min_label_num:
        #     break

    '''save'''
    print('SegInit: %.2f\nTrain: %.2f' % (start_time2 - start_time1, time.time() - start_time2))

def run():
    parser = argparse.ArgumentParser(
        description='Unsupervised learning on a large scene RS image.',
        epilog='Developed by CVEO Team.',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-i', '--input',
        help='name of the input image',
        metavar='image_name',
        required=True)

    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=4,
        metavar='num',
        help='training epochs (default: 4)')

    parser.add_argument(
        '-c', '--classes',
        help='number of categories',
        type=int,
        default=32)

    parser.add_argument(
        '-b', '--buffersize',
        help='buffer size',
        type=int,
        default=512)

    parser.add_argument(
        '-s', '--stride',
        help='buffer size',
        type=int,
        default=448)

    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    run()
