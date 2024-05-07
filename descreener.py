import cv2
import numpy as np


def load_image(img_path):
    '''
    loads in image
    used in descreen_image
    param img_path (str): needs directory
    '''
    img = np.float32(cv2.imread(img_path).transpose(2, 0, 1))
    return img

def normalize_power(h, w, min_power=0.01):
    '''
    thresholds power (energy ** 2) to a minimum of 0.01 (by default)
    Used in descreen_image
    '''
    x = np.arange(w)
    y = np.arange(h)
    cx = np.abs(x - w//2) ** 0.5
    cy = np.abs(y - h//2) ** 0.5
    energy = cx[None, :] + cy[:, None]
    return np.maximum(energy * energy, 0.01)


def ellipse_mask(w, h):
    '''
    makes ellipse mask for descreening
    used in descreen_image
    '''
    offset = (w + h) / (2 * w * h)
    y, x = np.ogrid[-h:h+1, -w:w+1]
    return np.uint8((x / w) ** 2 + (y / h) ** 2 - offset  <= 1)

def save_image(out_dir, img_name, out_img):
    '''
    save images to location out_dir (str) + img_name (str)
    '''
    print('Saving image to', out_dir + img_name)
    cv2.imwrite(out_dir + img_name, (out_img.transpose(1,2,0).astype(np.uint8)))

def descreen_image(img_path, out_dir=None, default_thresh=0, rad=48, mid=8):
    '''
    main function
    param default_thresh (int): in my exploration controls how much it wipes out the printing artifacts
    param rad (int): sets the radius of the filtering circles
    param mid (int): sets the size of the middle ellipse
    '''
    img = load_image(img_path)

    out_image = np.empty(img.shape)
    rows, cols = img.shape[-2:]
    coefs = normalize_power(rows,cols)
    ew, eh = cols//mid, rows//mid
    pw, ph = (cols- ew * 2) // 2, (rows - eh * 2) // 2
    middle = np.pad(ellipse_mask(ew, eh), ((ph,rows-ph-eh*2-1), (pw,cols-pw-ew*2-1)), 'constant')

    for i in range(3):
        fftimg = cv2.dft(img[i], flags = 18)
        fftimg = np.fft.fftshift(fftimg)
        spectrum = 20 * np.log(cv2.magnitude(fftimg[:,:, 0], fftimg[:,:, 1]) * coefs)

        ret, thresh = cv2.threshold(np.float32(np.maximum(0, spectrum)), default_thresh, 255, cv2.THRESH_BINARY)
        thresh *= 1 - middle
        thresh = cv2.dilate(thresh, ellipse_mask(rad,rad))
        thresh = cv2.GaussianBlur(thresh, (0,0), rad/3., 0, 0, cv2.BORDER_REPLICATE)
        thresh = 1 - thresh / 255

        fftimg = fftimg * np.repeat(thresh[..., None], 2, axis=2)
        fftimg = np.fft.ifftshift(fftimg)
        fftimg = cv2.idft(fftimg)

        out_image[i] = cv2.magnitude(fftimg[:,:,0], fftimg[:,:,1])

    img_name = img_path.split('/')[-1]
    if out_dir is not None:
        img_name, img_format = img_name.split('.')
        # img_name += 'rad'+str(rad) + 'mid' + str(mid) +'defThresh'+ str(default_thresh) + '.tif'
        img_name += '_descreened.' + img_format
        save_image(out_dir, img_name, out_image)
        
        return
    else:
        return out_image
    
