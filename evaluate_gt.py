import numpy as np
from osgeo import gdal_array
from sklearn import metrics

def run():
    # gt_path = "data/austin26/gt.tif"
    # pred_path = "results/{}/{}/{}".format("austin26", "512", "test_pixel_epoch_7.tif")
    # # pred_path = "results/austin26/isodata.img"

    # gt_path = "data/top_potsdam_3_10/gt.tif"
    # # pred_path = "results/{}/{}/{}".format("top_potsdam_3_10", "21-06-22_22-53-53", "test_pixel_epoch_2.tif")
    # pred_path = "results/top_potsdam_3_10/isodata.img"

    gt_path = "data/evlab44/gt.tif"
    # pred_path = "results/{}/{}/{}".format("evlab44", "21-06-22_23-09-29", "test_pixel_epoch_2.tif")
    pred_path = "results/evlab44/isodata.img"


    gt = gdal_array.LoadFile(gt_path).flatten()
    pred = gdal_array.LoadFile(pred_path).flatten()

    assert len(gt) == len(pred)


    # ARI = metrics.adjusted_rand_score(gt, pred)  
    AMI = metrics.adjusted_mutual_info_score(gt, pred)  
    FMI = metrics.fowlkes_mallows_score(gt, pred)  

    # print("ARI", ARI)
    print("AMI", AMI)
    print("FMI", FMI)

if __name__ == '__main__':
    run()
