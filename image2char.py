import argh
import cv2
from IPython import embed
import matplotlib.colors as mcolors
import numpy as np
import PIL
import pickle
import os
import string
from tqdm import tqdm

from PIL import Image, ImageFont, ImageDraw

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def may_create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def create_char_image(char, image_path):
    #print("creating {}".format(char))
    img = Image.new("L", (12,12), 255)  # grayscale, blank white image
    font = ImageFont.truetype("unifont-11.0.02.ttf",12)
    draw = ImageDraw.Draw(img)
    draw.text((0,0), char, font=font)
    img.save(image_path)

def print_lines(lines):
    for line in lines:
        print(line)

CHAR_SIZE = 14

def draw_lines(lines, colors, output_path):
    H = len(lines)
    W = len(lines[0])
    img = Image.new("RGB", (CHAR_SIZE * W,CHAR_SIZE * H), (255,255,255))  # grayscale, blank black image
    font = ImageFont.truetype("unifont-11.0.02.ttf", CHAR_SIZE)
    draw = ImageDraw.Draw(img)
    for w in range(W):
        for h in range(H):
            draw.text((w*CHAR_SIZE, h*CHAR_SIZE), lines[h][w], font=font, fill=tuple(colors[h, w]))
    img.save(output_path)

def get_gray_scale(char):
    base_dir = "./images/"
    may_create_dir("./images/")
    img_path = base_dir+ str(ord(char))+".png"
    if not os.path.exists(img_path):
        create_char_image(char, img_path)
    img = Image.open(img_path)
    array = np.asarray(img)
    return np.mean(array)

SCALE = 100

def read_charset(charset_name):
    with open("./charset/{}.txt".format(charset_name)) as f:
        return list(f.read())

def charset_from_range(start_char, end_char):
    return [chr(i) for i in range(ord(start_char), ord(end_char)+1)]

def main(inp="input.png", out="output.jpg", regen=False):
    print("preparing")
    #charset = list(string.printable[:-5])
    #charset = [chr(i) for i in range(19968, 40918)]
    if regen:
        charset = []
        # common chinese
        charset += read_charset("chinese_common")

        # fullwidth chars
        charset += charset_from_range("！", "～")
        
        # math symbols
        #charset += charset_from_range("∀", "⋿")

        # misc symbols
        #charset+= charset_from_range("☀", "⛿")

        char_grayscale = []
        for char in tqdm(charset):
            char_grayscale.append(get_gray_scale(char))
        min_gray, max_gray = np.min(char_grayscale), np.max(char_grayscale)
        # normalize to 0 to 255
        char_grayscale = (char_grayscale - min_gray) / (max_gray-min_gray) * 255 * SCALE
        lookup_table = sorted(zip(char_grayscale, charset), key = lambda x: x[0])
        lookup_table.append((512 * SCALE, None))

        idx = 0
        gray_lookup = []
        for i in range(256 * SCALE):
            while not lookup_table[idx][0] <= i < lookup_table[idx+1][0]:
                idx+=1
            gray_lookup.append(lookup_table[idx][1])
        pickle.dump(gray_lookup, open("gray_lookup.pkl","bw"))
    else:
        gray_lookup = pickle.load(open("gray_lookup.pkl","br"))
    print("resizing...")
    img = np.asarray(Image.open(inp))[:,:,:3]
    
    h, w = img.shape[0]//CHAR_SIZE, img.shape[1]//CHAR_SIZE

    small_img = cv2.resize(img, (w, h))
    gray_img = rgb2gray(small_img)

    print("generating")
    lines = ["".join([gray_lookup[int(gray_img[j,i]*SCALE)] for i in range(w)]) for j in range(h)]
    #print_lines(lines)
    print("drawing")
    draw_lines(lines, small_img, out)

if __name__=="__main__":
    #argh.dispatch_command(main)
    import glob;
    input_names = glob.glob("./inputs/*")
    for i,input_name in enumerate(input_names):
        print("==processing {}==".format(input_name))
        output_name = "./outputs/"+os.path.basename(input_name).split(".")[0] + ".png"
        main(input_name, output_name, regen=(i==0))
