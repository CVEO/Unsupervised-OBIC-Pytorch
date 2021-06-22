import os
import time
from tqdm import tqdm
import random
import numpy as np
from osgeo import gdal, gdal_array
import torch
import torch.nn.functional as F
from lovasz_losses import lovasz_softmax_flat
from focal_loss import focal_loss
from model import DeepNet
from segmentation import SaveLabelArrayInCompressMode

class Args(object):
    input_image_path = 'data/austin1.tif'
    output_seg_path = "data/austin1_label.tif"
    model_path = "model_as1.pt"

    # input_image_path = 'data/austin26.tif'
    # output_seg_path = "data/austin26_label.tif"
    # model_path = "model_as.pt"

    # input_image_path = 'data/G50G017079.tif'  
    # output_seg_path = "data/G50G017079_mask.tif"
    # model_path = "model_fj.pt"

    # input_image_path = 'data/G50G009077.tif'  
    # output_seg_path = "data/G50G009077_mask.tif"
    # model_path = "model_fj2.pt"

    # input_image_path = 'data/ah_image.tif'  
    # output_seg_path = "data/ah_mask.tif"
    # model_path = "model_ah.pt"

    # input_image_path = 'data/airport.tif'  
    # output_seg_path = "data/airport_mask.tif"
    # model_path = "model_ap.pt"

    # input_image_path = 'data/city.bmp'  
    # output_seg_path = "data/city_mask.tif"
    # model_path = "model_ct.pt"
    train_epoch = 4
    mod_dim1 = 64  #
    mod_dim2 = 32
    gpu_id = 0

    min_label_num = 10  # if the label number small than it, break loop
    max_label_num = 256  # if the label number small than it, start to show result image.

def run():
    start_time0 = time.time()

    args = Args()
    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # choose GPU:0
    image = gdal_array.LoadFile(args.input_image_path)
    channels, height, width = image.shape

    '''load segmentation result'''
    seg_map = gdal_array.LoadFile(args.output_seg_path)


    '''train init'''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(device)

    classes = args.mod_dim2
    model = DeepNet(inp_dim=channels, mod_dim1=args.mod_dim1, mod_dim2=classes).to(device)
    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_lovasz = lovasz_softmax_flat
    criterion_focal = focal_loss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x_buffersize, y_buffersize = 512, 512
    x_stride, y_stride = 448, 448


    '''load seg_lab patches'''
    patches_info = []
    dict_seg_lab = {}    
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

            seg_map_buffer = seg_map[y_offset:(y_offset+y_buffersize), x_offset:(x_offset+x_buffersize)].flatten()
            seg_lab = [np.where(seg_map_buffer == u_label)[0] for u_label in np.unique(seg_map_buffer)]
            dict_seg_lab[(y_offset, x_offset)] = seg_lab

            patches_info.append([
                y_offset, x_offset, y_buffersize, x_buffersize
            ])


            if is_x_last:
                break

            x_offset += x_stride

        if is_y_last:
            break
        y_offset += y_stride                 


    '''train loop'''
    start_time1 = time.time()

    for batch_idx in range(args.train_epoch):
        '''forward'''
        model.train()

        shuffled_patches_info = random.sample(patches_info, len(patches_info))
        for y_offset, x_offset, y_buffersize, x_buffersize in tqdm(shuffled_patches_info):

            buffer = image[:, y_offset:(y_offset+y_buffersize), x_offset:(x_offset+x_buffersize)].astype(np.float32) / 255.0
            buffer = buffer[np.newaxis, :, :, :]
            tensor = torch.from_numpy(buffer).to(device)

            optimizer.zero_grad()
            output = model(tensor)[0]
            output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
            target = torch.argmax(output, 1)
            im_target = target.data.cpu().numpy()

            '''refine'''
            seg_lab = dict_seg_lab[(y_offset, x_offset)]
            for inds in seg_lab:
                u_labels, hist = np.unique(im_target[inds], return_counts=True)
                im_target[inds] = u_labels[np.argmax(hist)]

            '''backward'''
            target = torch.from_numpy(im_target)
            target = target.to(device)
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
            buffer = image[:, y_offset:(y_offset+y_buffersize), x_offset:(x_offset+x_buffersize)].astype(np.float32) / 255.0
            buffer = buffer[np.newaxis, :, :, :]
            tensor = torch.from_numpy(buffer).to(device)
            output = F.softmax(model(tensor)[0], dim=0).data.cpu().numpy() * 63.
            im_target[:, y_offset:(y_offset+y_buffersize), x_offset:(x_offset+x_buffersize)] += output.astype("uint8")
        im_target = np.argmax(im_target, 0).astype("uint8")
        SaveLabelArrayInCompressMode(im_target, "test_pixel_epoch_{}.tif".format(batch_idx), seive_small_area=True)

        # if len(un_label) < args.min_label_num:
        #     break

    '''save'''
    time0 = time.time() - start_time0
    time1 = time.time() - start_time1
    print('PyTorchInit: %.2f\nTimeUsed: %.2f' % (time0, time1))
    # cv2.imwrite("seg_%s_%ds.png" % (args.input_image_path[6:-4], time1), show)

    torch.save(model.state_dict(),  args.model_path)


if __name__ == '__main__':
    run()
