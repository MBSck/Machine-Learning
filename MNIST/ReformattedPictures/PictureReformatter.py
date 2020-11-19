from PIL import Image
import os

filepath = os.path.abspath("Pictures")
reformatted_path = os.path.dirname(os.path.abspath(__file__))


def get_filename(filename, path):
    """Gets the filename of picture to be reformatted"""
    return os.path.join(path, filename)


def resize(filename, path):
    """Resizes the picture so it can be progressed by ML"""
    file = get_filename(filename, path)
    im = Image.open(file)
    imResize = im.resize((28,28), Image.ANTIALIAS)
    imResize.save(file[:-4] + "28Pixel.png", 'PNG', quality=90)


def black_n_white(filename, path):
    """Removes RGB and turns the picture BW"""
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
    resize("Nummer6Mama.png", "Pictures")
    resize("Nummer9Mama.png", "Pictures")
