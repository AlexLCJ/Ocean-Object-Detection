import cv2
import os
import numpy as np
import time

### HE
def he(image):
    B,G,R = cv2.split(image)
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)
    result = cv2.merge((B,G,R))
    return result
### HE

### CLAHE
def clahe(image,clipLimit=2.0, tileGridSize=(8, 8)):
    B,G,R = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    clahe_B = clahe.apply(B)
    clahe_G = clahe.apply(G)
    clahe_R = clahe.apply(R)
    result = cv2.merge((clahe_B,clahe_G,clahe_R))
    return result
### CLAHE

### defog

def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    '''if r <= 0:
        return src
    h, w = src.shape[:2]
    I = src
    res = np.minimum(I  , I[[0]+range(h-1)  , :])
    res = np.minimum(res, I[range(1,h)+[h-1], :])
    I = res
    res = np.minimum(I  , I[:, [0]+range(w-1)])
    res = np.minimum(res, I[:, range(1,w)+[w-1]])
    return zmMinFilterGray(res, r-1)'''
    return cv2.erode(src, np.ones((2*r+1, 2*r+1)))                      #使用opencv的erode函数更高效
def guidedfilter(I, p, r, eps):
    '''引导滤波，直接参考网上的matlab代码'''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r,r))
    m_p = cv2.boxFilter(p, -1, (r,r))
    m_Ip = cv2.boxFilter(I*p, -1, (r,r))
    cov_Ip = m_Ip-m_I*m_p
    m_II = cv2.boxFilter(I*I, -1, (r,r))
    var_I = m_II-m_I*m_I
    a = cov_Ip/(var_I+eps)
    b = m_p-a*m_I
    m_a = cv2.boxFilter(a, -1, (r,r))
    m_b = cv2.boxFilter(b, -1, (r,r))
    return m_a*I+m_b


def getV1(m, r, eps, w, maxV1):  #输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m,2)                                         #得到暗通道图像
    V1 = guidedfilter(V1, zmMinFilterGray(V1,7), r, eps)     #使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)                              #计算大气光照A
    d = np.cumsum(ht[0])/float(V1.size)
    for lmax in range(bins-1, 0, -1):
        if d[lmax]<=0.999:
            break
    A  = np.mean(m,2)[V1>=ht[1][lmax]].max()
    V1 = np.minimum(V1*w, maxV1)                   #对值范围进行限制
    return V1,A


def defog(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    m = m/255.0
    Y = np.zeros(m.shape)
    V1,A = getV1(m, r, eps, w, maxV1)               #得到遮罩图像和大气光照
    for k in range(3):
        Y[:,:,k] = (m[:,:,k]-V1)/(1-V1/A)           #颜色校正
    Y =  np.clip(Y, 0, 1)
    if bGamma:
        Y = Y**(np.log(0.5)/np.log(Y.mean()))       #gamma校正,默认不进行该操作
    return Y*255
### defog

### msrcr
# reference https://blog.csdn.net/weixin_38285131/article/details/88097771
def singleScaleRetinex(img, sigma):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex


def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex

def colorRestoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration


def simplestColorBalance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
    return img


def MSRCR(img, sigma_list=[15, 80, 200], G=5.0, b=25.0, alpha=125.0, beta=46.0, low_clip=0.01, high_clip=0.99):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, sigma_list)
    img_color = colorRestoration(img, alpha, beta)
    img_msrcr = G * (img_retinex * img_color + b)
    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255
    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)
    return img_msrcr
### msrcr



IMAGE = r"E:\Ocean Object Detection\UPRC2018\train\image\CHN083846_0043.jpg"
RESULT = r'E:\Ocean Object Detection\OceanObjectDetection\img_enhancement\result'


if __name__ == "__main__":
    image = cv2.imread(IMAGE)
    t = time.time()
    he_image = he(image)
    print("HE done! Use {}ms".format((time.time()-t)*1000))
    t = time.time()
    clahe_image = clahe(image)
    print("CLAHE done! Use {}ms".format((time.time()-t)*1000))
    t = time.time()
    defog_image = defog(image)
    print("Defog done! Use {}ms".format((time.time()-t)*1000))
    t = time.time()
    msrcr_image = MSRCR(image)
    print("MSRCR done! Use {}ms".format((time.time()-t)*1000))
    cv2.imwrite(RESULT+"he_XXX.jpg", he_image)
    cv2.imwrite(RESULT+"clahe_XXX.jpg", clahe_image)
    cv2.imwrite(RESULT+"defog_XXX.jpg", defog_image)
    cv2.imwrite(RESULT+"msrcr_XXX.jpg", msrcr_image)

