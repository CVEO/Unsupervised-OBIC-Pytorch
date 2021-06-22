from tqdm import tqdm
import numpy as np
from osgeo import gdal, gdal_array
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


from model import DeepNet
from segmentation import SaveLabelArrayInCompressMode

IMAGE_PATH = "data/austin1.tif"
SEG_PATH = "data/austin1_label.tif"
MODEL_PATH = "model_as.pt"

# IMAGE_PATH = "data/austin26.tif"
# SEG_PATH = "data/austin26_label.tif"
# MODEL_PATH = "model_as.pt"

# IMAGE_PATH = "data/G50G017079.tif"
# SEG_PATH = "data/G50G017079_mask.tif"
# MODEL_PATH = "model_fj.pt"

# IMAGE_PATH = "data/G50G009077.tif"
# SEG_PATH = "data/G50G009077_mask.tif"
# MODEL_PATH = "model_fj2.pt"

# IMAGE_PATH = "data/ah_image.tif"
# SEG_PATH = "data/ah_mask.tif"
# MODEL_PATH = "model_ah.pt"

# IMAGE_PATH = "data/airport.tif"
# SEG_PATH = "data/airport_mask.tif"
# MODEL_PATH = "model_ap.pt"

# IMAGE_PATH = "data/city.bmp"
# SEG_PATH = "data/city_mask.tif"
# MODEL_PATH = "model_ct.pt"
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(device)

    image = gdal_array.LoadFile(IMAGE_PATH)
    channels, height, width = image.shape
    seg_map = gdal_array.LoadFile(SEG_PATH).flatten()


    model = DeepNet(inp_dim=channels, mod_dim1=64, mod_dim2=32).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    x_buffersize, y_buffersize = 512, 512
    x_stride, y_stride = 448, 448
    # im_target = np.zeros((32, height, width), dtype="float16")
    im_target = np.zeros((32, height, width), dtype="uint8")

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

            buffer = image[:, y_offset:(y_offset+y_buffersize), x_offset:(x_offset+x_buffersize)].astype(np.float32) / 255.0
            buffer = buffer[np.newaxis, :, :, :]
            tensor = torch.from_numpy(buffer).to(device)
            # output = F.softmax(model(tensor)[0], dim=0).data.cpu().numpy()
            # im_target[:, y_offset:(y_offset+y_buffersize), x_offset:(x_offset+x_buffersize)] += output.astype("float16")
            output = F.softmax(model(tensor)[0], dim=0).data.cpu().numpy() * 63.
            im_target[:, y_offset:(y_offset+y_buffersize), x_offset:(x_offset+x_buffersize)] += output.astype("uint8")


            if is_x_last:
                break

            x_offset += x_stride

        if is_y_last:
            break
        y_offset += y_stride                 

    
    print(im_target.shape)
    im_target = np.argmax(im_target, 0).astype("uint8")
    print(im_target.shape)
    SaveLabelArrayInCompressMode(im_target, "test_pixel_result.tif")

    im_target = im_target.flatten()

    # dict_objid_rect = {}
    # for pixel_pos in tqdm(range(height * width)):
    #     objid = seg_map[pixel_pos]
    #     if objid in dict_objid_rect:
    #         dict_objid_rect[objid].append(pixel_pos)
    #     else:
    #         dict_objid_rect[objid] = [pixel_pos]

    # for u_label in tqdm(np.unique(seg_map)):
    #     inds = dict_objid_rect[u_label]
    #     # inds = np.where(seg_map == u_label)[0]
    #     u_labels, hist = np.unique(im_target[inds], return_counts=True)
    #     im_target[inds] = u_labels[np.argmax(hist)]

    # SaveLabelArrayInCompressMode(im_target.reshape((height, width)), "test_refine_result.tif")

    image = image.reshape(-1, height*width).transpose()
    image_out = image.copy()
    for class_label in tqdm(np.unique(im_target)):
        inds = np.where(im_target == class_label)[0]
        ave_color = image[inds].mean(axis=0).astype("uint8")
        image_out[inds] = ave_color

        
    gdal_array.SaveArray(image_out.transpose().reshape((channels, height, width)), "test_refine_image_result.tif")

if __name__ == '__main__':
    run()