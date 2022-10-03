
### DOWNLOAD AND UNZIP DATA

from pathlib import Path
import requests
import shutil
import sys
import gzip
import os

def get_destination(i, create):
    destination = Path(__file__).parent.parent / \
                  "data" / "case_{:05d}".format(i) / "imaging.nii.gz"
    if create and not destination.parent.exists():
        destination.parent.mkdir()
    return destination


def cleanup(msg, temp_f):
    if temp_f.exists():
        temp_f.unlink()
    print(msg)
    sys.exit()


def download(cid, imaging_url, imaging_name_tmplt, temp_f):
    remote_name = imaging_name_tmplt.format(cid)
    url = imaging_url + remote_name
    try:
        with requests.get(url, stream=True) as r:
            with temp_f.open('wb') as f:
                shutil.copyfileobj(r.raw, f)
        shutil.move(str(temp_f), str(get_destination(cid, True)))
    except KeyboardInterrupt:
        cleanup("KeyboardInterrupt", temp_f)
    except Exception as e:
        cleanup(str(e), temp_f)



def download_data(imaging_url, imaging_name_tmplt, temp_f):    
    left_to_download = []
    for i in  range(0, 210):
        dst = get_destination(i, False)
        if not dst.exists():
            left_to_download = left_to_download + [i]

    print("{} cases to download...".format(len(left_to_download)))
    for i, cid in enumerate(left_to_download):
        print("{}/{}... ".format(i + 1, len(left_to_download)))
        download(cid, imaging_url, imaging_name_tmplt, temp_f)


def unzip_data():
    for _dir in os.listdir("../data"):
        case_path = os.path.join("../data", _dir)
        
        img_path = os.path.join(case_path, "imaging.nii.gz")
        seg_path = os.path.join(case_path, "segmentation.nii.gz")
        
        target_img_path = os.path.join(case_path, "imaging.nii")
        target_seg_path = os.path.join(case_path, "segmentation.nii")

        if not os.path.exists(target_img_path):
            print(f"unzipping case {_dir}/imaging.nii.gz")
            with gzip.open(img_path, 'rb') as f_in:
                with open(target_img_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print(f"already unzipped case {target_img_path}")

        if not os.path.exists(target_img_path):
            print(f"unzipping case {_dir}/segmentation.nii.gz")
            with gzip.open(seg_path, 'rb') as f_in:
                with open(target_seg_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print(f"already unzipped case {target_img_path}")
        

### LOAD DATA INTO MEMORY

from skimage.transform import resize
from _utils import load_case
import numpy as np
        

# ilgisiz etiketleri temizlemek için
def cleanKidneyMask(mask):
    mask[mask != 1] = 0
    return mask


def cleanTumourMask(mask):
    mask[mask != 2] = 0
    return mask


def cleanCystMask(mask):
    mask[mask != 3] = 0
    return mask


def load_data_to_memory(img_row_size=512, img_col_size=512, range_end=209, step=30):
    X_train_tumour = np.empty([1, img_row_size, img_col_size, 1])
    Y_train_tumour = np.empty([1, img_row_size, img_col_size, 1])
    
    print(f"loading cases from 1 to {range_end} with step {step}")
    
    for i in range(1, range_end, step): 
        string = "case_{:05d}".format(i)
        print("loading", string)
        
        volume, segmentation = load_case(string)
        imgs = np.array(volume.dataobj,np.float32)
        segs = np.array(segmentation.dataobj,np.float32)
        print(segs.shape)

        imgs_train = np.expand_dims(imgs, axis=3)
        segs_train = np.expand_dims(cleanKidneyMask(segs), axis=3)

        X_train_tumour = np.row_stack((X_train_tumour, imgs_train))
        Y_train_tumour = np.row_stack((Y_train_tumour, segs_train))
    
    return X_train_tumour, Y_train_tumour


### RESIZE AND PREPROCESS IMAGES

from skimage import exposure


def resize_and_preprocess_images(X, y, new_rows, new_cols):
    
    X_train_tumour = X
    Y_train_tumour = y
    
    imgs_in = X_train_tumour
    [n, x, y, z] = X_train_tumour.shape
    
    Xtrain_out = np.zeros((n, new_rows, new_cols, 3))
    for n, i in enumerate(imgs_in):
        img=imgs_in[n, :, :, :]
       # img[img>512]=512
       # img[img<-512]=-512
       
        img=resize(img, Xtrain_out.shape[1:])
        img_rescaled=exposure.rescale_intensity(img,out_range=(-1, 1))
        img_adapteq = exposure.equalize_adapthist(img_rescaled, clip_limit=0.03)
        dimx, dimy,dimz=img_adapteq.shape
        im=np.zeros([dimx,dimy,3])
        
        im[:,:,0]=img_adapteq[:,:,0]
        im[:,:,1]=img_adapteq[:,:,0]
        im[:,:,2]=img_adapteq[:,:,0]
        
        Xtrain_out[n, :, :, :] = im
    
    imgs_in = Y_train_tumour
    [n, x, y, z] = Y_train_tumour.shape
    
    Ytrain_out = np.zeros((n, new_rows, new_cols, z))
    for n, i in enumerate(imgs_in):
        Ytrain_out[n, :, :, :] = resize(imgs_in[n, :, :, :], Ytrain_out.shape[1:])
        
    return Xtrain_out, Ytrain_out