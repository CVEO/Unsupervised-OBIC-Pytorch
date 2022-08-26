import os
import time
from osgeo import gdal, gdal_array
from skimage import segmentation

scale=32

input_images_path = "data/GID_example/"
output_segs_path = "data/GID_example/"


def SaveLabelArrayInCompressMode(src_array, filename, file_format="GTiff", seive_small_area = False):
    driver = gdal.GetDriverByName(file_format)
    if driver is None:
        raise ValueError("Can't find driver " + file_format)

    dst = driver.CreateCopy(filename, gdal_array.OpenArray(src_array, None), options=["COMPRESS=LZW", "PREDICTOR=2"])
    if seive_small_area:
        gdal.SieveFilter(dst.GetRasterBand(1), None, dst.GetRasterBand(1), 20)
    return dst

def run():
    for image_name in os.listdir(input_images_path):
        input_image_path=input_images_path+image_name
        output_seg_path=output_segs_path+image_name.split('.')[0]+'_seg.tif'
        image = gdal_array.LoadFile(input_image_path).transpose([1, 2, 0])
        seg_map = segmentation.felzenszwalb(image, scale=scale, sigma=0.5, min_size=64).astype("uint32")

        SaveLabelArrayInCompressMode(seg_map, output_seg_path)

if __name__ == '__main__':
    run()