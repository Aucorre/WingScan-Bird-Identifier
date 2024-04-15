from struct import unpack
import os


marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
    
    def decode(self):
        data = self.img_data
        while True:
            marker, = unpack(">H", data[0:2])
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2+lenchunk:]            
            if len(data) == 0:
                break        


bads = []

image_dir = 'C:\\Users\emman\Desktop\YNOV\B3\Projet_Data\\birdsnap\download\images'

for file, subdir, files in os.walk(image_dir):
    folder_name = os.path.basename(file)  # Obtenir le nom du dossier parent
    for img in files:
        image_path = os.path.join(file, img)
        image = JPEG(image_path) 
        try:
            image.decode()
            print(folder_name + "/" + img + " is a valid image") 
        except:
            bads.append(os.path.join(file, img))
            print(folder_name + "/" + img + " is not a valid image")

print(bads)
print(len(bads))
