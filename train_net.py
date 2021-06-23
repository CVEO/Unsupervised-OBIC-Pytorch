import os
import time
from datetime import datetime
import argparse
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
from osgeo import gdal_array
import torch
import torch.nn.functional as F
# from losses.lovasz_losses import lovasz_softmax_flat
# from losses.focal_loss import focal_loss
from model import DeepNet
from segmentation import SaveLabelArrayInCompressMode

def train(args):
    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # choose GPU:0

    start_time0 = time.time()

    input_image_path = "data/{}/image.tif".format(args.input)
    seg_path = "data/{}/seg.tif".format(args.input)
    result_id = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    '''load image'''
    image = gdal_array.LoadFile(input_image_path)

    '''load segmentation result'''
    seg_map = gdal_array.LoadFile(seg_path)
    channels, height, width = image.shape

    '''train init'''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(device)

    classes = args.classes
    model = DeepNet(inp_dim=channels, classes=classes).to(device)
    criterion_ce = torch.nn.CrossEntropyLoss()
    # criterion_lovasz = lovasz_softmax_flat
    # criterion_focal = focal_loss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x_buffersize, y_buffersize = args.buffersize, args.buffersize
    x_stride, y_stride = args.stride, args.stride


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
                y_offset, x_offset, y_buffersize, x_buffersize
            ])


            if is_x_last:
                break

            x_offset += x_stride

        if is_y_last:
            break
        y_offset += y_stride                 


    '''Init segmap'''
    start_time1 = time.time()

    dict_seg_lab = {}    
    for y_offset, x_offset, y_buffersize, x_buffersize in tqdm(patches_info):
        seg_map_buffer = seg_map[y_offset:(y_offset+y_buffersize), x_offset:(x_offset+x_buffersize)].flatten()
        dict_seg_lab[(y_offset, x_offset)] = [np.where(seg_map_buffer == u_label)[0] for u_label in np.unique(seg_map_buffer)]



    '''train loop'''
    start_time2 = time.time()

    for batch_idx in range(args.epochs):
        '''forward'''
        model.train()

        shuffled_patches_info = random.sample(patches_info, len(patches_info))
        for y_offset, x_offset, y_buffersize, x_buffersize in tqdm(shuffled_patches_info):

            buffer = image[:, y_offset:(y_offset+y_buffersize), x_offset:(x_offset+x_buffersize)].astype(np.float32)
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
            # loss = criterion_ce(output, target) + criterion_lovasz(output, target)
            loss = criterion_ce(output, target)
            # loss = criterion_focal(output, target) + criterion_lovasz(output, target)

            loss.backward()
            optimizer.step()                

        print('Loss:', batch_idx, loss.item())

        ''' Validation '''
        im_target = np.zeros((classes, height, width), dtype="uint8")
        model.eval()
        for y_offset, x_offset, y_buffersize, x_buffersize in tqdm(patches_info):
            buffer = image[:, y_offset:(y_offset+y_buffersize), x_offset:(x_offset+x_buffersize)].astype(np.float32)
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
    print('PyTorchInit: %.2f\nSegInit: %.2f\nTimeUsed: %.2f' % (start_time1 - start_time0, start_time2 - start_time1, time.time() - start_time2))
    # cv2.imwrite("seg_%s_%ds.png" % (args.input_image_path[6:-4], time1), show)

    # torch.save(model.state_dict(),  args.model_path)

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
