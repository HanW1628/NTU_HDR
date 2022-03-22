import numpy as np
import cv2
import glob
import os

# threshold <= 10 has better result, 15 has bad result
def Convert_Image(src_img, threshold_i=5):
    gray_img = np.zeros((src_img.shape[0], src_img.shape[1]), dtype=np.uint8)
    mtb_img = np.zeros((src_img.shape[0], src_img.shape[1]), dtype=np.uint8)
    mask_img = np.zeros((src_img.shape[0], src_img.shape[1]), dtype=np.uint8)
    denoise_img = np.zeros((src_img.shape[0], src_img.shape[1]), dtype=np.uint8)

    # convert to grayscale
    temp_img = src_img.astype(np.uint32)    # prevent overflow
    gray_img = (temp_img[:,:,0]*19/256 + temp_img[:,:,1]*183/256 + temp_img[:,:,2]*54/256)     # BGR
    med_i = np.median(gray_img)

    # convert to MTB
    mtb_img = np.where(gray_img > med_i, 255, 0)

    # exclusion map
    upper_bound = med_i + threshold_i
    lower_bound = med_i - threshold_i
    mask_img[gray_img < lower_bound] = 255
    mask_img[gray_img > upper_bound] = 255

    # bind mask_img and mtb_img
    denoise_img = np.logical_and(mtb_img, mask_img)
    
    
    return denoise_img

# shift tar to find current best dx, dy in 9 pixels
def Shift_Image(src, tar, last_dx, last_dy):
    # convert to grayscale first (denoise_img)
    src_dimg = Convert_Image(src)
    tar_dimg = Convert_Image(tar)

    # shift last dx,dy
    h, w = src_dimg.shape[:2]
    min = h * w
    M = np.float32([[1, 0, last_dx*2], [0, 1, last_dy*2]])
    src_dimg = src_dimg.astype(np.uint8)
    tar_dimg = tar_dimg.astype(np.uint8)
    new_tar = cv2.warpAffine(tar_dimg, M, (w, h))

    # then find current best dx, dy
    for x in range(-1,2):
        for y in range(-1,2):
            M = np.float32([[1, 0, x], [0, 1, y]])        # M為平移矩陣,x為寬移動的距離,y為高
            tmp_tar = cv2.warpAffine(new_tar, M, (w, h))  # 仿射變換函式   (w, h):平移後圖像的大小
            z = np.sum(np.logical_xor(src_dimg, tmp_tar) == 1)
            if z < min:
                min = z
                dx = x
                dy = y

    # image * 2, so shift * 2
    return dx + last_dx*2, dy + last_dy*2

# input RGB image src, tar    num = scale 1/2 times
def Image_Alignment(src, tar, num):
    if num == 0:
        dx, dy = Shift_Image(src, tar, 0, 0)
    else:
        h, w = src.shape[:2]
        h_src = cv2.resize(src, (h//2,w//2))
        h_tar = cv2.resize(tar, (h//2,w//2))
        last_dx, last_dy = Image_Alignment(h_src, h_tar, num-1)
        dx, dy = Shift_Image(src, tar, last_dx, last_dy)

    return dx, dy  


# test
# a = cv2.imread('my_data/my_1.jpg')
# b = cv2.imread('my_data/my_2.jpg')

# d_x, d_y = Image_Alignment(a, b, 5)

# print(d_x,d_y)

# h, w = b.shape[:2]
# M = np.float32([[1, 0, d_x], [0, 1, d_y]])    # M為平移矩陣,x為寬移動的距離,y為高
# tmp_tar = cv2.warpAffine(b, M, (w, h))        # 仿射變換函式   (w, h):平移後圖像的大小

# cv2.imwrite(os.path.join('my_output','shift_img.jpg'), tmp_tar)


# main()

# Read file
# imgspath = glob.glob(os.path.join('memorial','*.png'))
# imgspath = glob.glob(os.path.join('hdr_pic/house','*.jpg'))
imgspath = glob.glob(os.path.join('hdr_pic/library','*.jpg'))
imgs = [cv2.imread(i) for i in imgspath]
All_Img = []

source_image = imgs[0]
for i,ele in enumerate(imgs[1:]):
    d_x, d_y = Image_Alignment(source_image, ele, 5)
    print(d_x,d_y)

    h, w = ele.shape[:2]
    M = np.float32([[1, 0, d_x], [0, 1, d_y]])
    after_shift_ele = cv2.warpAffine(ele, M, (w, h))    

    # cv2.imwrite(os.path.join('memorial_output', f'img{i}.png'), after_shift_ele)
    # cv2.imwrite(os.path.join('hdr_pic_output/house', f'img{i}.jpg'), after_shift_ele)
    cv2.imwrite(os.path.join('hdr_pic_output/library', f'img{i}.jpg'), after_shift_ele)





# import numpy as np
# import cv2
# import glob


# # convert input image to MTB image
# class MyImage:
#     def __init__(self, src_img):
#         self.src_img = src_img
#         self.gray_img = np.zeros((self.src_img.shape[0],self.src_img.shape[1]), dtype=int)
#         self.mtb_img  = np.zeros((self.src_img.shape[0],self.src_img.shape[1]), dtype=int)
#         self.mask_img = np.zeros((self.src_img.shape[0],self.src_img.shape[1]), dtype=int)
#         self.denoise_img = np.zeros((self.src_img.shape[0],self.src_img.shape[1]), dtype=int)
#         self.med_i = 0
#         self.threshold_i = 15
    
#     def grayscale(self):
#         self.src_img = self.src_img.astype(np.int)
#         self.gray_img = (self.src_img[:,:,0]*19/256 + self.src_img[:,:,1]*183/256 + self.src_img[:,:,2]*54/256)
#         self.med_i = np.median(self.gray_img)

#     def mtb(self):
#         # # convert to MTB
#         self.mtb_img = np.where(self.gray_img > self.med_i, 255, 0)
    
#     def mask(self):
#         # # exclusion map
#         upper_bound = self.med_i + self.threshold_i
#         lower_bound = self.med_i - self.threshold_i
#         self.mask_img[self.gray_img < lower_bound] = 255
#         self.mask_img[self.gray_img > upper_bound] = 255
     
#     def bind_mtb_mask(self):
#         # bind mask_img and mtb_img
#         self.denoise_img = np.bitwise_and(self.mtb_img, self.mask_img)

#     def convert_image(self):
#         self.grayscale()
#         self.mtb()
#         self.mask()
#         self.bind_mtb_mask()
    

# # class MyImage:
# #     def __init__(self, src_img):
# #         self.src_img = src_img
# #         self.gray_img = np.zeros((self.src_img.shape[0],self.src_img.shape[1]), dtype=int)
# #         self.mtb_img  = np.zeros((self.src_img.shape[0],self.src_img.shape[1]), dtype=int)
# #         self.mask_img = np.zeros((self.src_img.shape[0],self.src_img.shape[1]), dtype=int)
# #         self.denoise_img = np.zeros((self.src_img.shape[0],self.src_img.shape[1]), dtype=int)
# #         self.med_i = 0
# #         self.threshold_i = 15
    
# #     def grayscale(self):
# #         # convert RGB to Grayscale
# #         for i in range(self.src_img.shape[0]):       #height
# #             for j in range(self.src_img.shape[1]):   #width
# #                 self.gray_img[i][j] = int((self.src_img[i][j][0]*19 + self.src_img[i][j][1]*183 + self.src_img[i][j][2]*54) / 256)
# #                 self.med_i = np.median(self.gray_img)

# #     def mtb(self):
# #         # convert to MTB
# #         for i in range(self.src_img.shape[0]):       #height
# #             for j in range(self.src_img.shape[1]):   #width
# #                 self.mtb_img[i][j] = ({True:255,False:0}[self.gray_img[i][j] > self.med_i])

# #     def mask(self):
# #         # exclusion map
# #         for i in range(self.src_img.shape[0]):       #height
# #             for j in range(self.src_img.shape[1]):   #width
# #                 self.mask_img[i][j] = ({True:0,False:255}[self.gray_img[i][j] < self.med_i + self.threshold_i and
# #                  self.gray_img[i][j] > self.med_i - self.threshold_i])

# #     def bind_mtb_mask(self):
# #         # bind mask_img and mtb_img
# #         self.denoise_img = np.bitwise_and(self.mtb_img, self.mask_img)

# #     def convert_image(self):
# #         self.grayscale()
# #         self.mtb()
# #         self.mask()
# #         self.bind_mtb_mask()

# def main():
#     # Read file

    
#     cv_img = []
#     for img in glob.glob("data/*.png"):
#         n = cv2.imread(img)
#         cv_img.append(n)

#     # img = cv2.imread('data/memorial0061.png')
#     # myimage = MyImage(img)
#     # myimage.convert_image()
#     # cv2.imwrite('class_img.jpg', myimage.denoise_img)

# if __name__ == '__main__':
#     main()


# # gray_img = np.zeros((img.shape[0],img.shape[1]), dtype=int)
# # mtb_img  = np.zeros((img.shape[0],img.shape[1]), dtype=int)
# # mask_img = np.zeros((img.shape[0],img.shape[1]), dtype=int)
# # denoise_img = np.zeros((img.shape[0],img.shape[1]), dtype=int)

# # # convert RGB to Grayscale
# # for i in range(img.shape[0]):       #height
# #     for j in range(img.shape[1]):   #width
# #         gray_img[i][j] = int((img[i][j][0] * 19 + img[i][j][1] * 183 + img[i][j][2] * 54) / 256)

# # # median intensities
# # med_i = np.median(gray_img)

# # # convert to MTB
# # for i in range(img.shape[0]):       #height
# #     for j in range(img.shape[1]):   #width
# #         mtb_img[i][j] = ({True:255,False:0}[gray_img[i][j] > med_i])

# # # exclusion map
# # threshold_i = 15    # guess
# # for i in range(img.shape[0]):       #height
# #     for j in range(img.shape[1]):   #width
# #         mask_img[i][j] = ({True:0,False:255}[gray_img[i][j] < med_i + threshold_i and gray_img[i][j] > med_i - threshold_i])

# # # bind mask_img and mtb_img
# # denoise_img = np.bitwise_and(mtb_img, mask_img)

# # ========================================================================


# # color = int(img[300][300])
# # print(color)

# # print(img[0],img[1],img[2])
# # img2 = cv2.imread('data/memorial0061.png', cv2.IMREAD_GRAYSCALE)

# # cv2.imwrite('img.jpg', myimg)
# # cv2.imshow('img2', img2)

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()