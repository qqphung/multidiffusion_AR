import os
import openai
import pandas as pd
import pickle 
import csv 
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from urllib.request import urlopen

openai.api_key = ''

def generate_box(text):
    # example_prompt = "I want you to act as a programmer. I will provide the description of an image, you should output the corresponding layout of this image, the size of the objects should be as large as possible. Each object in the image is one rectangle or square box in the layout. You should return each object and the corresponding coordinate of its boxes. the size of the image is 1024 * 1024.\nthe prompt :\"four cats in the field\", \ncat: (220, 318, 380, 460)\ncat: (440, 220, 714, 460)\ncat: (858, 242, 1002, 560)\ncat: (350, 694, 606, 846)\nthe prompt: \"a cat on the right of a dog on the road\"\ncat: (482, 200, 804, 592)\ndog: (428, 634, 820, 970)\nthe prompt: \"five balls in the room\"\nball: (148, 560, 386, 824)\nball: (84, 138, 420, 404)\nball: (588, 104, 922, 368)\nball: (620, 436, 912, 672)\nball: ( 640, 750, 896, 964)\nthe prompt: \"a cat sitting on the car\"\nBecause the cat sitting on the car so the car bellow the cat and cat in the surface of car, therefore the result\ncat: (305, 384, 590, 600)\ncar: (100, 600, 928, 906)\n"
    example_prompt = "I want you to act as a programmer. I will provide the description of an image, you should output the corresponding layout of this image, spacial relationship of the objects should be followed in the description and size of the objects should be as large as possible. Each object in the image is one rectangle or square box in the layout. You should return each object and the corresponding coordinate of its boxes. the size of the image is 512 * 512.\nthe prompt :\"four cats in the field\", \ncat: (110, 159, 190, 230)\ncat: (220, 110, 357, 230)\ncat: (429, 121, 501, 280)\ncat: (175, 347, 303, 423)\nthe prompt: \"a cat on the right of a dog on the road\"\ncat: (241, 100, 402, 296)\ndog: (217, 317, 410, 485)\nthe prompt: \"five balls in the room\"\nball: (74, 280, 193, 412)\nball: (42, 69, 210, 202)\nball: (294, 52, 461, 184)\nball: (310, 218, 456, 336)\nball: ( 320, 375, 448, 482)\nthe prompt: \"a cat sitting on the car\"\nBecause the cat sitting on the car so the car bellow the cat and cat in the surface of car, therefore the result\ncat: (153, 192, 295, 300)\ncar: (50, 300, 464, 453)\n"
    prompt = example_prompt  + text
    # call api
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt= prompt,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    # complete text
    completed_text = response['choices'][0]['text']
    print('text', completed_text)
    # getting boxes and object names
    boxes = completed_text.split('\n')[1:]
    d = {}
    name_objects = []
    boxes_of_object = []
    # convert boxes from string to int
    for b in boxes:
        if b == '\n': continue
       
        b_split = b.split(":")
        name_objects.append(b_split[0])
        boxes_of_object.append(text_list(b_split[1]))


    # for o in list_object:
    #     for b in boxes:
            
    #         if o in b.lower():

    #             if not o in d.keys():
    #                 d[o] = [text_list(b.split(": ")[1])]
    #             else:      
    #                 d[o].append(text_list(b.split(": ")[1]))
    
    return name_objects, boxes_of_object

def text_list(text):
    text =  text.replace(' ','')
    digits = text[1:-1].split(',')
    # import pdb; pdb.set_trace()
    result = []
    for d in digits:
        result.append(int(d))
    # coodinate chat GPT api is opposite
    tempt = result[0]
    result[0] = result[1]
    result[1] = tempt 
    tempt = result[2]
    result[2] = result[3]
    result[3] = tempt
    return tuple(result)

def save_img(folder_name, img, prompt, iter_id, img_id):
    os.makedirs(folder_name, exist_ok=True)
    img_name = str(img_id) + '_' + str(iter_id) + '_' + prompt.replace(' ','_')+'.jpg'
    img.save(os.path.join(folder_name, img_name))
# def draw_box(text, boxes,output_folder, img_name,sample ):
def draw_box(output_folder, sample, prompt, iter_id, img_id, text, boxes):
    draw = ImageDraw.Draw(sample)
    font = ImageFont.truetype(urlopen("https://criptolibertad.s3.us-west-2.amazonaws.com/img/fonts/Roboto-LightItalic.ttf"), size=20)
    for i, box in enumerate(boxes):
        t = text[i]
        draw.rectangle([(box[1], box[0]),(box[3], box[2])], outline=128, width=2)
        draw.text((box[1]+5, box[0]+5), t, fill=200,font=font )
    img_name = str(img_id) + '_' + str(iter_id) + '_' + prompt.replace(' ','_')+'.jpg'
    sample.save(os.path.join(output_folder, img_name))


def load_gt(csv_pth):
    gt_data = pd.read_csv(csv_pth).to_dict('records')
    meta = []
    syn_prompt = []
    
    # import pdb; pdb.set_trace()
    for sample in gt_data:
        meta.append(sample['meta_prompt'])
        syn_prompt.append(sample['synthetic_prompt'])
    return meta, syn_prompt
def load_box(pickle_file):
    with open(pickle_file,'rb') as f:
        data = pickle.load(f)
    return data

# def read_csv(path_file):
#     with open(path_file,'r') as f:
#         reader = csv.reader(f)
#         for row in reader:
#             l = row.split(",")
#             if l[1] == 'Positional' or l[1] == 'Counting':


# read_csv()
# boxes = generate_box("three cats", ['cat'])
# print(boxes)
