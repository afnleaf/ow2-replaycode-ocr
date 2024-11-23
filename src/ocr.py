import cv2 as cv
import numpy as np
import os
import pandas as pd
import pytesseract
from dotenv import load_dotenv
from MTM import matchTemplates, drawBoxesOnRGB
from PIL import Image, ImageEnhance, ImageFilter
#from cv2.typing import MatLike
#from typing import List
#from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# to enable testing functionality
load_dotenv()
TEST: bool = True if str(os.getenv("ENVIRONMENT")) == "test" else False

# should go in a config file along with other logging related stuff
# O, U, I and L excluded as OW2 replay codes don't use them due to similarity with other characters.
# run once on import
config_psm8 = "-l eng --oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHJKMNPQRSTVWXYZ0123456789"
config_psm10 = "-l eng --oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHJKMNPQRSTVWXYZ0123456789"
# the directory we will store our images
output_append = "/app/output/"

# load trocr model
#trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
#trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

'''
map_image_names = [
        "ilios.png",
        "eichenwalde.png",
        "dorado.png",
        "colosseo.png",
        "suravasa.png"
    ]
'''

# tools  ----------------------------------------------------------------------

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
        output_filename = output_append + "boxes.png"
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
            output_filename_before = output_append + "before" + str(i) + ".png"
            cv.imwrite(output_filename_before, crop)
        # process our replay code crops
        crop_final = process_cropped_image(crop)
        list_of_crops.append(np.array(crop_final))
        # after
        if TEST:
            output_filename_after = output_append + "after_" + str(i) + ".png"
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


# templates for maps
def get_map_crops(img_final, hits):
    # get locations of matches out of the dataframe
    bboxes = hits["BBox"].tolist()

    # where we are storing all the crop data
    list_of_maps = []

    # loop through each locations
    for i, box in enumerate(bboxes):        
            #print(box)
            scalefactor = 2.5
            print(box)
            if box[3] > 28:
                scalefactor = 3
            elif box[3] < 13:
                scalefactor = 2
            # positions for crop
            template_width = box[2] + (1/box[2]) + 1
            template_height = box[3]
            start_y = box[1] + template_height + (template_height/4) + 1
            end_y = start_y + template_height 
            start_x = box[0] + template_width + 1
            end_x = start_x + template_width * scalefactor
            
            crop = img_final[int(start_y):int(end_y), int(start_x):int(end_x)]

            list_of_maps.append(crop)
            if TEST:
                output_filename = output_append + "map_" + str(i) + ".png"
                cv.imwrite(output_filename, crop)

    return list_of_maps

# processed the cropped code images  ------------------------------------------

# gonna need to keep working on this to make it better
def process_codes(list_of_crops):
    # list of replay code text
    replaycodes = []

    for index, crop in enumerate(list_of_crops):
        #print(f"code {index + 1}:")
        
        # method 1, image to letters to text
        code = process_code_mode1(crop, index)

        # check for code length, fallback
        if len(code) < 6:
            # method 2, image to word to text
            code = process_code_mode2(crop, index)

        # add code if not already added
        if code not in replaycodes:
            replaycodes.append(code)
        
    return replaycodes


# find each letter using contours
def process_code_mode1(crop, index):
    # our code
    code = ""

    # back to rgb for drawing bounding box with colour
    crop_copy = cv.cvtColor(crop, cv.COLOR_BGR2RGB)

    # adaptive threshold to create binary image
    window_size = 41
    constant_value = 8
    binary_image = cv.adaptiveThreshold(crop, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV, window_size, constant_value)

    # find contours on the binary image
    contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # sort bounding boxes
    bboxes = [cv.boundingRect(c) for c in contours]
    (contours, bboxes) = zip(*sorted(zip(contours, bboxes),
        key=lambda b:b[1][0], reverse=False))

    # loop through each bounding box
    for i, bound_rect in enumerate(bboxes):
        # Draw the rectangle on the input image:
        # Get the dimensions of the bounding rect:
        rect_x = int(bound_rect[0])
        rect_y = int(bound_rect[1])
        rect_w = int(bound_rect[2])
        rect_h = int(bound_rect[3])

        letter = crop[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]
        ws = 120
        # create border around letter for better accuracy
        letter = cv.copyMakeBorder(letter,ws,ws,ws,ws,
            cv.BORDER_CONSTANT,value=[255,255,255])
        if TEST:
            cv.imwrite(f"{output_append}char_{i}.png", letter)

        # ocr tesseract 
        character = pytesseract.image_to_string(letter, config=config_psm8)
        if not character:
            character = pytesseract.image_to_string(letter, config=config_psm10)
        #print(f"str <{character.strip()}>")
                
        '''
        letter_rgb = cv.cvtColor(letter,cv.COLOR_GRAY2RGB)
        pixel_values = trocr_processor(letter_rgb, return_tensors="pt").pixel_values
        #generated_ids = trocr_model.generate(pixel_values, max_length=1)
        generated_ids = trocr_model.generate(pixel_values, max_new_tokens=1)
        generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #print(generated_text.upper())
        character = generated_text.upper()
        '''

        # remove whitespace
        character = character.strip()

        if character == "O":
            code += "0"
        
        elif len(character) > 1:
            code += character[0]
        else:
            code += character
        
        # green bounding box for testing
        color = (0, 255, 0)
        cv.rectangle(crop_copy, (int(rect_x), int(rect_y)),
                    (int(rect_x + rect_w), int(rect_y + rect_h)), color, 2)
        output_filename = f"/app/output/boxedcontours_{index}.png"
        if TEST:
            cv.imwrite(output_filename, crop_copy)
        

        # take first 6 letters
        if i == 5:
            break

    #print(code)
    return code



# process code using image to boxes method
# filter out bad boxes
def process_code_mode2(crop, index):
    print("method 1")
    # get letters
    boxes = pytesseract.image_to_boxes(crop, config=config_psm8)
    # setup
    code = ""
    h, w, = crop.shape
    # back to rgb instead of grayscale for colour
    to_box = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
    # loop through all letters found
    for b in boxes.splitlines():
        print(f"{b} -- ")
        b = b.split(' ')
        
        # Get the dimensions of the bounding rect:
        x1 = int(b[1])
        x2 = int(b[3])
        y1 = h - int(b[4])
        y2 = h - int(b[2])
        rect_width = x2 - x1
        rect_height = y2 - y1

        # Compute contour area:
        contour_area = rect_height * rect_width

        # Compute aspect ratio:
        reference_ratio = 1.0
        contour_ratio = rect_width / rect_height
        epsilon = 1.1
        ratio_diff = abs(reference_ratio - contour_ratio)
        #print((ratio_diff, contour_area))

        # add box to letter

        # if height is much larger than width, ignore
        # if width is much larger than height, ignore
        if rect_height < 1.5 * rect_width and rect_width < 1.5 * rect_height:
            pass

        to_box = cv.rectangle(to_box, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (255, 0, 0), 2)

        
        # add letter to code
        if b[0] == "O":
            code += "0"
        else:
            code += b[0]


    print(f"{code}")

    # save boxed image
    if TEST:
        output_filename = f"/app/output/boxed_{index}.png"
        cv.imwrite(output_filename, to_box)
    
    return code.strip()


# processed the cropped map images  ------------------------------------------

def process_maps(list_of_maps):
    # list of map names
    map_names = []

    for i, crop in enumerate(list_of_maps):        
        map_name = process_map_mode1(crop, i)
        map_names.append(map_name)
        
    return map_names


def process_map_mode1(crop, index):
    
    maps = load_maps()

    for i, map_template in enumerate(maps):
        result = cv.matchTemplate(crop, map_template, cv.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(result >= threshold)
        if loc[0].size > 0:
            #print(map_image_names[i])
            image_name = map_image_names[i]
            map_name = image_name[0:image_name.index(".")]
            #print(map_name)
            return map_name
    return


def load_maps():
    maps = []
    for map_image_name in map_image_names:
        input_filename = "images/map_templates/" + map_image_name
        map_image = cv.imread(input_filename, cv.IMREAD_GRAYSCALE)
        maps.append(map_image)

    return maps

# standard --------------------------------------------------------------------

# main function, small testing when bot is in prod 
def main():
    # load templates
    template_filename="/app/images/template_large.png"
    template = load_template(template_filename)
    list_of_templates = create_templates(template)
    # test an image
    #input_filename="/app/images/test_cases/image_case12.png"
    #image = cv.imread(input_filename)
    #assert image is not None, "file could not be read, check with os.path.exists()"
    #crops, map_crops = template_match(image, list_of_templates)
    replaycodes = process_codes(crops)
    print_codes(replaycodes)
    # image processing on maps
    #list_of_maps = process_maps(map_crops)
    #print(list_of_maps)

    


# Default notation
if __name__ == "__main__":
    main()
