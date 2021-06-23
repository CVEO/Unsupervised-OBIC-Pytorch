import numpy as np
from osgeo import gdal_array
from sklearn.metrics import adjusted_mutual_info_score, fowlkes_mallows_score
from skimage.metrics import adapted_rand_error
def run():
    # gt_path = "data/austin26/gt.tif"
    # pred_path = "results/{}/{}/{}".format("austin26", "512", "test_pixel_epoch_7.tif")
    # # pred_path = "results/austin26/isodata.img"

    # gt_path = "data/top_potsdam_3_10/gt.tif"
    # # pred_path = "results/{}/{}/{}".format("top_potsdam_3_10", "21-06-22_22-53-53", "test_pixel_epoch_2.tif")
    # pred_path = "results/top_potsdam_3_10/isodata.img"

    gt_path = "data/evlab44/gt.tif"
    pred_path = "results/{}/{}/{}".format("evlab44", "21-06-22_23-09-29", "test_pixel_epoch_2.tif")
    # pred_path = "results/evlab44/kmeans.img"


    gt = gdal_array.LoadFile(gt_path)
    pred = gdal_array.LoadFile(pred_path)

    # assert len(gt) == len(pred)
    assert gt.shape == pred.shape

    ARE,_,_ = adapted_rand_error(gt, pred)
    print("ARE", ARE)


    gt = gt.flatten()
    pred = pred.flatten()

    # ARI = metrics.adjusted_rand_score(gt, pred)  
    AMI = adjusted_mutual_info_score(gt, pred)  
    FMI = fowlkes_mallows_score(gt, pred)  


    # print("ARI", ARI)
    print("AMI", AMI)
    print("FMI", FMI)

if __name__ == '__main__':
    run()
