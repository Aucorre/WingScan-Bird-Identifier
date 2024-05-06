# WingScan : Projet de reconnaissance d'espèces d'oiseau

## Contexte :
Depuis plusieurs années les ornithologues cherchent à suivre les flux migratoires de oiseaux pour étudier leur comportement et comprendre comment nos activités influent sur leur mode de vie (déforestation/chasse/réchauffement climatique). L'idée derrière ce projet est de proposer un modèle de classification d'image suffisament fiable pour identifier une espèce parmis une liste de 500 espèces d'oiseau / proposer un top 3 si le modèle n'est pas assez sur. De cette façon n'importe quel particulier pourrait participer au travail de suivi des espèces sans nécessiter des connaisances poussées en ornithologie. Les scientifiques auraient ainsi un flux de données fiables et continu.

## Provenance des données d'entraînement :
Nos données proviennent d'un étude produite par Thomas Berg : [Birdsnap](https://thomasberg.org/papers/birdsnap-cvpr14.pdf). Un dataset de 500 classes a été constitué avec entre 70 et 100 images par classe, ces 500 classes correspondent aux espèces d'oiseaux les plus communes en Amérique du Nord. Toutes les images ont été classés à l'aide de Flickr en recherchant le nom de l'oiseau en question, pour les classes manquants d'images Amazon Mechanical Turk a été utilisé pour labelliser des images d'oiseau similaires. Le Dataset initial contenait 49 829 images, étant donné que ce travail date de 2015 nous avons été en mesure de récupérer uniquement ~39 600 images. La métadonnée de la majorité de ces images semble également altérée, nous n'avons donc pas pu l'utiliser comme dans le travail original.

## Modèles étudiés :
- MobileNetV2
- ResNet50
- DenseNet169
- BaseNet
- EfficientNetB0

## Travail sur les données :