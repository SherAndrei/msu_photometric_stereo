from PIL import Image
import sys
import os


def do_crop(img: Image, box: tuple, result_filename: str):
    cropped = img.crop(box)
    cropped.save(result_filename)


def main(argv):
    if (len(argv) != 2):
        print(f"Usage: {argv[0]} <Path>")
        return

    name = argv[1]
    name_wo_extention, extention = os.path.splitext(name)
    img = Image.open(name)

    do_crop(img, (250, 450, 650, 800), './camera_lid/' +
            name_wo_extention + extention)

    do_crop(img, (600, 600, 950, 950), './rock/' +
            name_wo_extention + extention)

    do_crop(img, (250, 850, 600, 1200), './sphere/' +
            name_wo_extention + extention)


if __name__ == '__main__':
    main(sys.argv)
