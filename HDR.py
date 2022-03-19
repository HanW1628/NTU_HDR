import numpy as np
import cv2
import glob


# convert input image to MTB image
class MyImage:
    def __init__(self, src_img):
        self.src_img = src_img
        self.gray_img = np.zeros((self.src_img.shape[0],self.src_img.shape[1]), dtype=int)
        self.mtb_img  = np.zeros((self.src_img.shape[0],self.src_img.shape[1]), dtype=int)
        self.mask_img = np.zeros((self.src_img.shape[0],self.src_img.shape[1]), dtype=int)
        self.denoise_img = np.zeros((self.src_img.shape[0],self.src_img.shape[1]), dtype=int)
        self.med_i = 0
        self.threshold_i = 15
    
    def grayscale(self):
        self.src_img = self.src_img.astype(np.int)
        self.gray_img = (self.src_img[:,:,0]*19/256 + self.src_img[:,:,1]*183/256 + self.src_img[:,:,2]*54/256)
        self.med_i = np.median(self.gray_img)

    def mtb(self):
        # # convert to MTB
        self.mtb_img = np.where(self.gray_img > self.med_i, 255, 0)
    
    def mask(self):
        # # exclusion map
        upper_bound = self.med_i + self.threshold_i
        lower_bound = self.med_i - self.threshold_i
        self.mask_img[self.gray_img < lower_bound] = 255
        self.mask_img[self.gray_img > upper_bound] = 255
     
    def bind_mtb_mask(self):
        # bind mask_img and mtb_img
        self.denoise_img = np.bitwise_and(self.mtb_img, self.mask_img)

    def convert_image(self):
        self.grayscale()
        self.mtb()
        self.mask()
        self.bind_mtb_mask()
    

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
#         # convert RGB to Grayscale
#         for i in range(self.src_img.shape[0]):       #height
#             for j in range(self.src_img.shape[1]):   #width
#                 self.gray_img[i][j] = int((self.src_img[i][j][0]*19 + self.src_img[i][j][1]*183 + self.src_img[i][j][2]*54) / 256)
#                 self.med_i = np.median(self.gray_img)

#     def mtb(self):
#         # convert to MTB
#         for i in range(self.src_img.shape[0]):       #height
#             for j in range(self.src_img.shape[1]):   #width
#                 self.mtb_img[i][j] = ({True:255,False:0}[self.gray_img[i][j] > self.med_i])

#     def mask(self):
#         # exclusion map
#         for i in range(self.src_img.shape[0]):       #height
#             for j in range(self.src_img.shape[1]):   #width
#                 self.mask_img[i][j] = ({True:0,False:255}[self.gray_img[i][j] < self.med_i + self.threshold_i and
#                  self.gray_img[i][j] > self.med_i - self.threshold_i])

#     def bind_mtb_mask(self):
#         # bind mask_img and mtb_img
#         self.denoise_img = np.bitwise_and(self.mtb_img, self.mask_img)

#     def convert_image(self):
#         self.grayscale()
#         self.mtb()
#         self.mask()
#         self.bind_mtb_mask()

def main():
    # Read file

    
    cv_img = []
    for img in glob.glob("data/*.png"):
        n = cv2.imread(img)
        cv_img.append(n)

    # img = cv2.imread('data/memorial0061.png')
    # myimage = MyImage(img)
    # myimage.convert_image()
    # cv2.imwrite('class_img.jpg', myimage.denoise_img)

if __name__ == '__main__':
    main()


# gray_img = np.zeros((img.shape[0],img.shape[1]), dtype=int)
# mtb_img  = np.zeros((img.shape[0],img.shape[1]), dtype=int)
# mask_img = np.zeros((img.shape[0],img.shape[1]), dtype=int)
# denoise_img = np.zeros((img.shape[0],img.shape[1]), dtype=int)

# # convert RGB to Grayscale
# for i in range(img.shape[0]):       #height
#     for j in range(img.shape[1]):   #width
#         gray_img[i][j] = int((img[i][j][0] * 19 + img[i][j][1] * 183 + img[i][j][2] * 54) / 256)

# # median intensities
# med_i = np.median(gray_img)

# # convert to MTB
# for i in range(img.shape[0]):       #height
#     for j in range(img.shape[1]):   #width
#         mtb_img[i][j] = ({True:255,False:0}[gray_img[i][j] > med_i])

# # exclusion map
# threshold_i = 15    # guess
# for i in range(img.shape[0]):       #height
#     for j in range(img.shape[1]):   #width
#         mask_img[i][j] = ({True:0,False:255}[gray_img[i][j] < med_i + threshold_i and gray_img[i][j] > med_i - threshold_i])

# # bind mask_img and mtb_img
# denoise_img = np.bitwise_and(mtb_img, mask_img)

# ========================================================================


# color = int(img[300][300])
# print(color)

# print(img[0],img[1],img[2])
# img2 = cv2.imread('data/memorial0061.png', cv2.IMREAD_GRAYSCALE)

# cv2.imwrite('img.jpg', myimg)
# cv2.imshow('img2', img2)

# cv2.waitKey(0)
# cv2.destroyAllWindows()