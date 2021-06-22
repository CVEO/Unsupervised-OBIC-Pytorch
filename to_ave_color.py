from tqdm import tqdm
import numpy as np
from osgeo import gdal, gdal_array
from segmentation import SaveLabelArrayInCompressMode

IMAGE_PATH = "data/austin1.tif"
SEG_PATH = "data/austin1_label.tif"
RESULT_PATH = "test_pixel_epoch_3.tif"


def run():
    # channels, height, width = image.shape
    seg_map = gdal_array.LoadFile(SEG_PATH).flatten()
    im_target = gdal_array.LoadFile(RESULT_PATH)
    height, width = im_target.shape
    image = gdal_array.LoadFile(IMAGE_PATH).reshape(-1, height*width).transpose()
    image_out = image.copy()
    im_target = im_target.flatten()

    dict_objid_rect = {}
    for pixel_pos in tqdm(range(height * width)):
        objid = seg_map[pixel_pos]
        if objid in dict_objid_rect:
            dict_objid_rect[objid].append(pixel_pos)
        else:
            dict_objid_rect[objid] = [pixel_pos]

    
    for u_label in tqdm(np.unique(seg_map)):
        inds = dict_objid_rect[u_label]
        # inds = np.where(seg_map == u_label)[0]
        u_labels, hist = np.unique(im_target[inds], return_counts=True)
        im_target[inds] = u_labels[np.argmax(hist)]
        
    # SaveLabelArrayInCompressMode(im_target.reshape((height, width)), "test2.tif")

    for class_label in tqdm(np.unique(im_target)):
        inds = np.where(im_target == class_label)[0]
        ave_color = image[inds].mean(axis=0).astype("uint8")
        image_out[inds] = ave_color

        
    gdal_array.SaveArray(image_out.transpose().reshape((3, height, width)), "test.tif")


if __name__ == '__main__':
    run()