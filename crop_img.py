from PIL import Image

path = '000000.png'
with open(path, 'rb') as f:
    with Image.open(f) as img:
        img.convert('RGB')
        print(img.size)
        img1 = img.crop((0,191,640,383))
        img1.save('croped.png')