from PIL import Image
import os

# Chemin vers le dossier contenant tes images
image_directory = 'C:\\Users\emman\Desktop\YNOV\B3\Projet_Data\\birdsnap\download\images'

cpt = 0
# Parcourir toutes les images dans le dossier
for subdir, dirs, files in os.walk(image_directory):
    for file in files:
        filepath = os.path.join(subdir, file)
        try:
            img = Image.open(filepath)  # Ouvrir l'image pour vérifier si elle est corrompue
            img.verify()  # Vérifier si c'est une image valide
            cpt+=1
            if(cpt % 100 == 0):
                print(f'{cpt} images vérifiées')
        except (IOError, SyntaxError) as e:
            print('Bad file:', filepath)