import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from skimage import io 
import skimage
import os
import math 
import time 
import random 
from scipy.ndimage.filters import gaussian_filter as spicy

''' 
This program makes a set number of images with defects in them 

The number of defects loaded in can be changed, depending on   
how many defects the user has created

The epoch controls how many defects will be in each image 
    ex. of epoch = 3, there will be 3 total defects on image 

light_img simulate lighting from different directions 

The total_img is the image with noise, defects, and lighting in it 

The total_img_map is just the map of the defects, used for training 

'''


path = "C:\\Python Projects\\Defects_for_python_created_parts"

out_dir_og = "C:\\Python Projects\\Defect_images_created\\"

defect_number = 11

number_of_runs = 50

prefix_1 = input("Please insert testing prefix: ")

for num in range(number_of_runs):

    prefix = prefix_1 + str(num) # Makes new prefix for each total run of program 

    for a in range(defect_number):

        img = ("Defect" + "_" + str(a) + ".png")

        og_img = io.imread(path + os.sep + img)

        og_img = np.mean(og_img, axis = 2)
        plt.imshow(og_img)

        rot_num = random.randint(0,3) #This is used to randomly rotate the image left or right 0 = none, 1 = 90, etc.

        flip_num = random.randint(0,1) #This is used to randomly rotate the image up or down

        og_img = np.rot90(og_img, k = rot_num)

        if flip_num == 1:
            og_img = np.fliplr(og_img) #Flip left  right
        
        flip_num_2 = random.randint(0,1)

        if flip_num_2 == 1: 
            og_img = np.flipud(og_img) # Flip up down
        
        print(og_img.shape)

    #plt.show()


        noise_level = 150; #Can adujst noise level
        image_size = (250,250)
        epoch = 1  #Sets number of defects per image 

        total_img = np.random.normal(loc = 0.0, scale = noise_level, size = image_size)

        total_img_map = np.zeros(shape = image_size) 

    #Six different lighting conditions, light coming from left, right, up, and down 
        light_img_left = np.zeros(shape = image_size)
        light_img_right = np.zeros(shape = image_size)
        light_img_south = np.zeros(shape = image_size)
        light_img_north = np.zeros(shape = image_size)
        light_img_mid_n_s = np.zeros(shape = image_size)
        light_img_mid_e_w = np.zeros(shape = image_size)
        light_img_nw_se = np.zeros(shape = image_size)
        light_img_ne_sw = np.zeros(shape = image_size)
        light_img_sw_ne = np.zeros(shape = image_size)
        light_img_se_nw = np.zeros(shape = image_size)

        light_intensity = 255

        for j in range(image_size[0]):

            light_img_left[:, j] = light_intensity  - j
            light_img_right[:, image_size[1] -1 -j] = light_intensity   - j
            light_img_south[image_size[0] - 1 -j, : ] = light_intensity   - j
            light_img_north[j, :] = light_intensity  - j
        for row in range(image_size[0]): #This for nested for loop creates lighting conditions from the four corners, coming in at an angle
            for column in range(image_size[1]):
                light_img_nw_se[row, column]= light_intensity - row - column
                light_img_ne_sw[image_size[0] - 1 - row, column]= light_intensity - row - column
                light_img_sw_ne[row, image_size[0]- 1  - column]= light_intensity - row - column
                light_img_se_nw[image_size[0]- 1 - row, image_size[0] - 1 - column] = light_intensity - row - column
            
            
            
        new_range = int(image_size[0] / 2.0 ) #want to start at center of image

        for k in range(new_range): 
            light_img_mid_n_s[(new_range -1) + k, :] = light_intensity  - k  
            light_img_mid_n_s[(new_range -1) - k, :] = light_intensity  - k
            light_img_mid_e_w[:, (new_range -1) + k] = light_intensity   - k 
            light_img_mid_e_w[:, (new_range -1) - k] = light_intensity  - k 
            
        for i in range(epoch): 

            for j in range(epoch): 

    #Gives random coordinates for where the defect should be 
                x_coor = random.randint(0,image_size[0] - og_img.shape[0])
                y_coor = random.randint(0, image_size[1] - og_img.shape[1])

    #Padding for defect image to become same size as total image 
                before_x = x_coor
                before_y = y_coor
                after_x = image_size[0] - x_coor - og_img.shape[0]
                after_y= image_size[1] - y_coor - og_img.shape[1]

                new_og_img = np.pad(og_img, pad_width= ((before_x, after_x) , (before_y, after_y))  )

            
    #Random int to determine what lighting condition to use 
                light_num = random.randint(0, 10)
                if light_num == 0: 
                    light_condition = light_img_left; 
                elif light_num == 1: 
                    light_condition = light_img_right; 
                elif light_num == 2: 
                    light_condition = light_img_north;
                elif light_num == 3: 
                    light_condition = light_img_south;
                elif light_num == 4: 
                    light_condition = light_img_nw_se;
                elif light_num == 5: 
                    light_condition = light_img_ne_sw;
                elif light_num == 6: 
                    light_condition = light_img_sw_ne;
                elif light_num == 7: 
                    light_condition = light_img_se_nw;
                elif light_num == 8: 
                    light_condition = light_img_mid_n_s;
                else:
                    light_condition = light_img_mid_e_w;

                plt.imshow(light_condition, cmap = "gray")
                plt.show()
    #Adding defect and lighting condition to output image 
                total_img = total_img + new_og_img  + light_condition#Put whatever light condition you want here 
                total_img = spicy(total_img, 3) #Adds guassian filter 
    #Adding defect to output map
                total_img_map = total_img_map + new_og_img

            if a == 0 or a == 2 or a == 3 or a == 5 or a == 8:
                out_dir = out_dir_og + "\\Type_1_Defect_Pencil\\"
            elif a == 1 or a == 4 or a == 6 or a == 7 or a == 9:
                out_dir = out_dir_og + "\\Type_2_Defect_Eraser\\"
            else: 
                out_dir = out_dir_og + "\\Type_0_Defect_None\\"
    #Save output defect image
            plt.imshow(total_img, cmap = "gray")
            io.imsave(out_dir + prefix + "Image" + "_" + str(i) + str(a) + "_defects" + ".png", skimage.img_as_float(total_img))
            plt.show()

    #Save map 
            total_img_map = total_img_map == 0 
            #plt.imshow(total_img_map, cmap = "gray")
            io.imsave(out_dir + prefix + "Image" + "_" + str(i) + str(a) + ".png", skimage.img_as_float(total_img_map))

    #Reset total image and total image map 
        total_img = np.random.normal(loc = 0.0, scale = noise_level, size = image_size)
        total_img_map = np.zeros(shape = image_size) 



""" Test to see if defect image map can be created 


total_img_map = np.zeros(shape = image_size) 

plt.imshow(total_img_map)

plt.show()

x_coor = random.randint(0,image_size[0] - og_img.shape[0])
y_coor = random.randint(0, image_size[1] - og_img.shape[1])
before_x = x_coor
before_y = y_coor
after_x = image_size[0] - x_coor - og_img.shape[0]
after_y= image_size[1] - y_coor - og_img.shape[1]

new_og_img = np.pad(og_img, pad_width= ((before_x, after_x) , (before_y, after_y))  )

total_img_map = total_img_map + new_og_img

plt.imshow(total_img_map)

plt.show() """ 
