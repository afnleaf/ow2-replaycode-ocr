import cv2 as cv
import numpy as np
import os
import pandas as pd
import pytesseract
from dotenv import load_dotenv
from MTM import matchTemplates, drawBoxesOnRGB
from PIL import Image, ImageEnhance, ImageFilter

import templates

# to enable testing functionality
load_dotenv()
TEST: bool = True if str(os.getenv("ENVIRONMENT")) == "test" else False
# the directory we will store our images
output_append = "/app/output/"

# should go in a config file along with other logging related stuff
# run once on import
config_psm8 = "-l eng --oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
config_psm10 = "-l eng --oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
VALID_CHARS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

# processed the cropped code images  ------------------------------------------

# gonna need to keep working on this to make it better
def process_codes(list_of_crops, case_index=0):
    # list of replay code text
    replaycodes = []

    for code_index, crop in enumerate(list_of_crops):
        #print(f"code {index + 1}:")
        
        # method 1, image to letters to text
        code = process_code_mode1(crop, case_index, code_index)

        # check for code length, fallback
        if len(code) < 6:
            # method 2, image to word to text
            code = process_code_mode2(crop, case_index, code_index)

        # add code if not already added
        if code not in replaycodes:
            replaycodes.append(code)
        
    return replaycodes

def classify_character(letter):
    scales = [0.5, 1.0, 1.5, 2.0, 2.5]
    results = []
    for scale in scales:
        scaled_letter = cv.resize(letter, None, fx=scale, fy=scale)
        character = pytesseract.image_to_string(scaled_letter, config=config_psm8)
        if not character:
            character = pytesseract.image_to_string(scaled_letter, config=config_psm10)
        if character:
            character = character[0].strip()
        else:
            character = "?"
        #print(f"{character}:")
        if character and character in VALID_CHARS:
            results.append(character)
    if not results:
        return None
    return max(set(results), key=results.count)


# find each letter using contours
def process_code_mode1(crop, case_index, code_index):
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
   
        # ocr tesseract 
        #character = pytesseract.image_to_string(letter, config=config_psm8)
        #if not character:
        #    character = pytesseract.image_to_string(letter, config=config_psm10)
        #print(f"str <{character.strip()}>")
        # remove whitespace
        #character = character.strip()

        character = classify_character(letter)
                
        '''
        letter_rgb = cv.cvtColor(letter,cv.COLOR_GRAY2RGB)
        pixel_values = trocr_processor(letter_rgb, return_tensors="pt").pixel_values
        #generated_ids = trocr_model.generate(pixel_values, max_length=1)
        generated_ids = trocr_model.generate(pixel_values, max_new_tokens=1)
        generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #print(generated_text.upper())
        character = generated_text.upper()
        '''
        '''
        # rapidocr test
        result, elapse = engine(letter)
        print(result)
        #print(result)
        character = None
        if result:
            character = result[0][1]
        #print(character)
        '''

        #print(character)
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
        if TEST:
            #output_filename = f"/app/output/boxedcontours_{index}.png"
            output_filename = f"/app/output/case{case_index}/boxedcontours_{code_index}.png"
            cv.imwrite(output_filename, crop_copy)
        

        # take first 6 letters
        if i == 5:
            break

    #print(code)
    return code

# process code using image to boxes method
# filter out bad boxes
def process_code_mode2(crop, case_index, code_index):
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
        #output_filename = f"/app/output/boxed_{index}.png"
        output_filename = f"/app/output/case{case_index}/boxed_{code_index}.png"
        cv.imwrite(output_filename, to_box)
    
    return code.strip()


# standard --------------------------------------------------------------------

# main function, small testing when bot is in prod 
def main():
    # load templates
    template_filename="/app/images/template_large.png"
    template = templates.load_template(template_filename)
    list_of_templates = templates.create_templates(template)
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

