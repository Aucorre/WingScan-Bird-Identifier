# WingScan : Projet de reconnaissance d'espèces d'oiseau

## Contexte :

---

Depuis plusieurs années les ornithologues cherchent à suivre les flux migratoires de oiseaux pour étudier leur comportement et comprendre comment nos activités influent sur leur mode de vie (déforestation/chasse/réchauffement climatique). L'idée derrière ce projet est de proposer un modèle de classification d'image suffisament fiable pour identifier une espèce parmis une liste de 500 espèces d'oiseau / proposer un top 3 si le modèle n'est pas assez sur. De cette façon n'importe quel particulier pourrait participer au travail de suivi des espèces sans nécessiter des connaisances poussées en ornithologie. Les scientifiques auraient ainsi un flux de données fiables et continu.

## Provenance des données d'entraînement :

---

Nos données proviennent d'un étude produite par Thomas Berg : [Birdsnap](https://thomasberg.org/papers/birdsnap-cvpr14.pdf). Un dataset de 500 classes a été constitué avec entre 70 et 100 images par classe, ces 500 classes correspondent aux espèces d'oiseaux les plus communes en Amérique du Nord. Toutes les images ont été classés à l'aide de Flickr en recherchant le nom de l'oiseau en question, pour les classes manquants d'images Amazon Mechanical Turk a été utilisé pour labelliser des images d'oiseau similaires. Le dataset initial contenait 49 829 images, étant donné que ce travail date de 2015 nous avons été en mesure de récupérer uniquement ~39 600 images. La métadonnée de la majorité de ces images semble également altérée, nous n'avons donc pas pu l'utiliser comme dans le travail original.

## Modèles étudiés :

---

- [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
- [MobileNetV3](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html#torchvision.models.mobilenet_v3_small)
- [ResNet101V2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet101V2)
- [DenseNet169](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet169.html)
- [DenseNet201](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet201.html)
- [BaseNet](https://pypi.org/project/basenet-api/1.2.0/)
- [EfficientNetB0](https://keras.io/api/applications/efficientnet/)

## Travail sur les données :

Un script python pour récupérer et trier toutes les images était fourni dans le travail initial (**GetData.py**). Nous avons commencé par revoir rapidement ce script pour inclure du multithreading afin d'accélérer la récupération des images, nous avons ensuite utilisé la librairie Pillow pour s'assurer de la conformité de chaque image enregistrée car nous avions des fichiers jpeg corrompus.

[Dataset disponible sur Kaggle](https://www.kaggle.com/datasets/emmanueljova/birdspecies)

## Modèles avec 100 classes :

> **Python 3.9 & Tensorflow 2.10**

- MobileNetV2 avec ajout de bruit de Gaussien (images 200x200 / 12 epoch / LR = 0,001 / Split 0.2 / Batchsize 64): précision 30.3%, précision de validation pas encore ajoutée
- EfficentNetB0 (images 224x224 / 10 epoch / LR = 0,001 / Split 0.2 / Batchsize 32) : précision 33.2%, précision de validation 15.7%

## Modèles avec 126 classes :

> **Python 3.9 & Tensorflow 2.10**

- MobileNetV3 précision (images 150x150 / 15 epoch / LR = 0,001 / Split 0.2 / Batchsize 50&75) : 348/1975 (17.6%) dans le top 3 des prédictions
- MobileNetV2 avec ajout de bruit Gaussien précision (images 150x150 / 15 epoch / LR = 0,001 / Split 0.2 / Batchsize 50&75) : 996/1975 (50.4%) dans le top 3 prédictions, la précision de validation est de 30.3%
- MobileNetV2 avec 4 dernières couches entrainables précision (images 150x150 / 15 epoch / LR = 0,001 / Split 0.2 / Batchsize 50&75) : 977/1975 (49.5%) dans le top 3 prédictions, la précision de validation est de 29.67%

## Modèles avec 500 classes (passage sur Kaggle):

> **Python 3.11 & Tensorflow 2.15**

- VGG16 : 15.75% de précision de validation, 2231/7923 (28.15%) dans le top 3 des prédictions
- ResNet101V2 précision : 16.72% de précision de validation, 2316/7923 (29.2%) dans le top 3 des prédictions
- DenseNet169 précision : 26.18% de précision de validation 3355/7923 (42.3%) dans le top 3 des prédictions

```py
base_model = DenseNet169(include_top = False)

base_model.trainable = False

model = Sequential([
    RandomFlip("horizontal", input_shape=(150, 150, 3)),
    RandomRotation(0.3),
    GaussianNoise(0.1),
    BatchNormalization(),
    base_model,
    Conv2D(32,3, activation="relu"),
    GlobalAveragePooling2D(),
    Dense(256, activation="relu")
    Dropout(0.2),
    Dense(500, activation='softmax')
])

model.summary()
```

- DenseNet201 précision : 28.9% de précision de validation 3636/7923 (45.9%) dans le top 3 des prédictions

```py
base_model = DenseNet201(include_top = False)

fine_tune_at = 700
base_model.trainable = True

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

model = Sequential([
    RandomFlip("horizontal", input_shape=(150, 150, 3)),
    RandomRotation(0.3),
    GaussianNoise(0.1),
    BatchNormalization(),
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(256, activation="relu"),
    Dense(500, activation='softmax')
])

model.summary()
```

- DenseNet169 avec format d'image en 224x224 précision : 35.31% de précision de validation 4439/7923 (56.0%) dans le top 3 des prédictions

```py

```

## Modèle final :

> **Python 3.11 & Tensorflow 2.15**

- DenseNet169 avec format d'image en 224x224, batch normalization, dropout et data augmentation. Précision de validation : 42.63% de précision de validation 5054/7923 (63.79%) dans le top 3 des prédictions

```py
base_model = DenseNet169(include_top = False)

fine_tune_at = 550
base_model.trainable = True

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

model = Sequential([
    RandomFlip("horizontal", input_shape=(224, 224, 3)),
    RandomRotation(0.3),
    GaussianNoise(0.1),
    BatchNormalization(),
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(400, activation="relu"),
    Dense(500, activation='softmax')
])

model.summary()
```
