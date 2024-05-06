# WingScan : Projet de reconnaissance d'espèces d'oiseau

## Contexte :

---

Depuis plusieurs années les ornithologues cherchent à suivre les flux migratoires de oiseaux pour étudier leur comportement et comprendre comment nos activités influent sur leur mode de vie (déforestation/chasse/réchauffement climatique). L'idée derrière ce projet est de proposer un modèle de classification d'image suffisament fiable pour identifier une espèce parmis une liste de 500 espèces d'oiseau / proposer un top 3 si le modèle n'est pas assez sur. De cette façon n'importe quel particulier pourrait participer au travail de suivi des espèces sans nécessiter des connaisances poussées en ornithologie. Les scientifiques auraient ainsi un flux de données fiables et continu.

## Provenance des données d'entraînement :

---

Nos données proviennent d'un étude produite par Thomas Berg : [Birdsnap](https://thomasberg.org/papers/birdsnap-cvpr14.pdf). Un dataset de 500 classes a été constitué avec entre 70 et 100 images par classe, ces 500 classes correspondent aux espèces d'oiseaux les plus communes en Amérique du Nord. Toutes les images ont été classés à l'aide de Flickr en recherchant le nom de l'oiseau en question, pour les classes manquants d'images Amazon Mechanical Turk a été utilisé pour labelliser des images d'oiseau similaires. Le Dataset initial contenait 49 829 images, étant donné que ce travail date de 2015 nous avons été en mesure de récupérer uniquement ~39 600 images. La métadonnée de la majorité de ces images semble également altérée, nous n'avons donc pas pu l'utiliser comme dans le travail original.

## Modèles étudiés :

---

- MobileNetV2
- MobileNetV3
- ResNet101V2
- DenseNet169
- DenseNet201
- BaseNet
- EfficientNetB0

## Travail sur les données :

Un script python pour récupérer et trier toutes les images était fourni dans le travail initial (**GetData.py**). Nous avons commencé par revoir rapidement ce script pour inclure du multithreading afin d'accélérer la récupération des images, nous avons ensuite utilisé la librairie Pillow pour s'assurer de la conformité de chaque image enregistrée car nous avions des fichiers jpeg corrompus.

## Modèles avec 100 classes :

- MobileNetV2
- EfficientNet
- ResNet50 

## Modèles avec 126 classes :
- MobileNetV3 précision : 348/1975 (17.6%) dans le top 3 des prédictions
- MobileNetV2 avec ajout de bruit Gaussien précision : 996/1975 (50.4%) dans le top 3 prédictions, la précision de validation est de 30.3%
- MobileNetV2 avec 4 dernières couches entrainables précision : 977/1975 (49.5%) dans le top 3 prédictions, la précision de validation est de 29.67%

## Modèles avec 500 classes :

- VGG16 : 15.75% de précision de validation, 2231/7923 (28.15%) dans le top 3 des prédictions
- ResNet101V2 précision : 16.72% de précision de validation, 2316/7923 (29.2%) dans le top 3 des prédictions
- DenseNet169 précision : 26.18% de précision de validation 3355/7923 (42.3%) dans le top 3 des prédictions
- DenseNet201 précision : 28.9% de précision de validation 3636/7923 (45.9%) dans le top 3 des prédictions
- DenseNet169 avec format d'image en 224x224 précision : 35.31% de précision de validation 4439/7923 (56.0%) dans le top 3 des prédictions
- MobileNetV3 précision : 348/1975 (17.6%) dans le top 3 des prédictions
- MobileNetV2 avec ajout de bruit Gaussien précision : 996/1975 (50.4%) dans le top 3 prédictions, la précision de validation est de 30.3%
- MobileNetV2 avec 4 dernières couches entrainables précision : 977/1975 (49.5%) dans le top 3 prédictions, la précision de validation est de 29.67%

## Modèles avec 500 classes :

- VGG16 : 15.75% de précision de validation, 2231/7923 (28.15%) dans le top 3 des prédictions
- ResNet101V2 précision : 16.72% de précision de validation, 2316/7923 (29.2%) dans le top 3 des prédictions
- DenseNet169 précision : 26.18% de précision de validation 3355/7923 (42.3%) dans le top 3 des prédictions
- DenseNet201 précision : 28.9% de précision de validation 3636/7923 (45.9%) dans le top 3 des prédictions
- DenseNet169 avec format d'image en 224x224 précision : 35.31% de précision de validation 4439/7923 (56.0%) dans le top 3 des prédictions

## Modèle final :

- DenseNet169 avec format d'image en 224x224, batch normalization, dropout et data augmentation. Précision de validation : 42.63% de précision de validation 5054/7923 (63.79%) dans le top 3 des prédictions
