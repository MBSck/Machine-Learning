from PIL import Image
import os

filepath = os.path.abspath("Pictures")
reformatted_path = os.path.dirname(os.path.abspath(__file__))


def get_filename(filename, path):
    return os.path.join(path, filename)


def resize(filename, path):
    file = get_filename(filename, path)
    im = Image.open(file)
    imResize = im.resize((28,28), Image.ANTIALIAS)
    imResize.save(file , 'JPEG', quality=90)


def black_n_white(filename, path):
    file = get_filename(filename, path)
    im = Image.open(file).convert("1")
    im.save(filename[:-4] + "_BNW.png")


def setting_pixel_values(filename, path):
    file = get_filename(filename, path)
    im = Image.open(file)
    pix = im.load()
    for i in range(50):
        for j in range(50):
            print(pix[i, j])


if __name__ == "__main__":
    setting_pixel_values("hand-drawn-number-1-vector-1468591_BNW.png", reformatted_path)