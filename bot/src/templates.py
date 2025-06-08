###
# templates.py
# the opencv processing for template matching and crops is done here
###

# external modules
import cv2 as cv
import numpy as np
import os
import pandas as pd
import pytesseract
from dotenv import load_dotenv
from MTM import matchTemplates, drawBoxesOnRGB
from PIL import Image, ImageEnhance, ImageFilter

# to enable testing functionality
load_dotenv()
TEST: bool = True if str(os.getenv("ENVIRONMENT")) == "test" else False
# the directory we will store our images
output_append = "/app/output/"

# create directories for debugging output
def create_output_folders(test_cases):
    os.makedirs("/app/output", exist_ok=True)
    for i, case in enumerate(test_cases):
        path = f"/app/output/case{i}"
        os.makedirs(path, exist_ok=True)
    #os.makedirs("/app/output/vlm", exist_ok=True)


# process what is given by attachement.read()
async def load_image_from_discord(image_data):
    try:
        array = np.asarray(bytearray(image_data), dtype=np.uint8)
        image = cv.imdecode(array, -1)
    except Exception as e:
        print(f"Error load image from disc: {e}")
    return image


# load template into memory via given filename
def load_template(template_filename):
    template = cv.imread(template_filename, cv.IMREAD_GRAYSCALE)
    assert template is not None, "File could not be read."
    return template


# create some templates
def create_templates(template):
    # where we store our templates
    list_of_templates = []           

    j = 0
    for i in range(1, 30, 3):
        template_resized = cv.resize(template, (0,0), fx=1/i, fy=1/i)
        list_of_templates.append((f"{j}", template_resized))
        j += 1
            
    return list_of_templates



# print replay code list, used in testing
def print_codes(replaycodes):
    print("Replay codes:")
    for code in replaycodes:
        print(code)


# image processing ------------------------------------------------------------

# process input image genius comment
# just turn it gray ez
def pre_process_input_image(image):
    # grayscale it
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


# how to solve the copy problem
# process each of the matched replaycode images for clearer text
def process_cropped_image(image):
    _, y = image.shape[::-1]
    # resize 2x
    image = cv.resize(image, (0,0), fx=24, fy=24)
    # invert
    image = cv.bitwise_not(image)
    # contrast / brightness control
    alpha = 6
    beta = 11
    # for low resolutions
    if y < 15:
        alpha = 7 
        beta = 20
    # call convertScaleAbs function
    image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    #thresholding 
    _,image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return image


# template match to generate crops --------------------------------------------

# match templates to input image
def template_match(img_input, templates):
    # process input image
    img_final = pre_process_input_image(img_input)
    #cv.imwrite(output_append + "input_final.png", img_final)

    # validate templates against input image
    list_of_templates = get_valid_templates(img_final, templates)

    # get list of template matches
    hits = get_hits(img_final, list_of_templates)
    draw_boxes_around_templates(img_input, hits)

    # get raw cropped codes
    raw_crops = get_raw_crops(img_final, hits)    

    # image processing on codes
    list_of_crops = process_crops(raw_crops)
    
    # get map blocks
    #map_crops = get_map_crops(img_final, hits)

    #return list_of_crops, map_crops
    return list_of_crops
    #return raw_crops


# out of the generated templates make sure you only match with the valid ones
def get_valid_templates(img_final, templates):
    # get width and height of image to check later
    w, h = img_final.shape[::-1]
    #print(f"w:{w} h:{h}")
    # check for template size larger than input image
    list_of_templates = []
    for j, template in enumerate(templates):
        wr, hr = template[1].shape[::-1]
        #print(f"wr:{wr} hr:{hr}")
        if wr <= w and hr <= h:
            list_of_templates.append(template)
            # write for log
            #output_filename = output_append + "template" + str(j) + ".png"
            #cv.imwrite(output_filename, template[1])

    return list_of_templates


# draw boxes to see the templates in the input image
def draw_boxes_around_templates(img_input, hits_sorted):
    # draw boxes around templates
    image_boxes = drawBoxesOnRGB(img_input, 
               hits_sorted, 
               boxThickness=1, 
               boxColor=(255, 255, 00), 
               showLabel=True,  
               labelColor=(255, 255, 0), 
               labelScale=0.5 )
    if TEST:
        #output_filename = output_append + "boxes.png"
        output_filename = f"{output_append}boxes.png"
        cv.imwrite(output_filename, image_boxes)


# get locations of template found in input image
#def get_hits(img_final: MatLike, list_of_templates: List[MatLike]) -> pd.DataFrame:
def get_hits(img_final, list_of_templates):
    # find matches and store locations in a pandas dataframe
    hits = matchTemplates(list_of_templates,  
               img_final,  
               method=cv.TM_CCOEFF_NORMED,  
               N_object=float("inf"),   
               score_threshold=0.79,   
               maxOverlap=0.25,   
               searchBox=None)

    # sort hits in order found in image, height, y pos, low to high
    # uses a sorting column of the extracted box
    hits_sorted = hits.sort_values(by='BBox', key=lambda x: x.str[1])
    
    return hits_sorted




# perform image processing on each cropped code to enhance readability
def process_crops(raw_crops):
    list_of_crops = []
    for i, crop in enumerate(raw_crops):
        # save crops to folder
        # before
        if TEST:
            #output_filename_before = output_append + "before" + str(i) + ".png"
            output_filename_before = f"{output_append}before{i}.png"
            cv.imwrite(output_filename_before, crop)
        # process our replay code crops
        crop_final = process_cropped_image(crop)
        list_of_crops.append(np.array(crop_final))
        # after
        if TEST:
            #output_filename_after = output_append + "after_" + str(i) + ".png"
            output_filename_after = f"{output_append}after{i}.png"
            cv.imwrite(output_filename_after, crop_final)
    
    return list_of_crops
    

# slice crops out of image
def get_raw_crops(img_final, hits):
    # get locations of matches out of the dataframe
    bboxes = hits["BBox"].tolist()

    # where we are storing all the crop data
    list_of_crops = []

    # loop through each locations
    for _, box in enumerate(bboxes):        
            #print(box)

            # positions for crop
            template_width = box[2] + int(1/box[2]) + 1
            template_height = box[3]
            start_y = box[1]
            end_y = box[1] + template_height + 1
            start_x = box[0] + template_width + 1
            #end_x = start_x + int(1/pow(template_width, 2))
            scalefactor = 4.5
            if box[3] > 28:
                scalefactor = 5.5
            elif box[3] < 13:
                scalefactor = 3.5
            end_x = start_x + template_width * scalefactor
            
            crop = img_final[start_y:int(end_y), start_x:int(end_x)]
            cv.imwrite("output/test12312.png", crop)

            list_of_crops.append(crop)

    return list_of_crops
