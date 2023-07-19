# ClassifyImagesWithEmbeddings
Este es un script de Python para clasificar imágenes utilizando el modelo CLIP de OpenAI y la biblioteca Sentence-Transformers.

## Requisitos
Python 3.x
Bibliotecas de Python: os, sentence_transformers, PIL, numpy, shutil

Puede instalar las bibliotecas necesarias usando pip:

Code
```
pip install sentence_transformers Pillow numpy
```

## Uso
El script está diseñado para clasificar imágenes en un conjunto de datos 'Tornillos'. De manera predeterminada, las imágenes se clasifican utilizando la similitud de coseno máxima entre la representación de la imagen dada y las representaciones de las imágenes de clase. También se puede utilizar la similitud de coseno promedio.

El script buscará las imágenes a clasificar en la ruta ./Data/Tornillos/Images. Se espera que las imágenes de clase estén en la ruta ./Data/Tornillos/Clases, donde cada subcarpeta representa una clase y contiene imágenes de esa clase.

El script moverá las imágenes clasificadas a la ruta ./Data/Tornillos/Dataset, donde cada subcarpeta representa una clase y contiene las imágenes clasificadas en esa clase.

Para ejecutar el script, simplemente corre:
```
python ClassifyImagesWithEmbeddings.py
```

## Funcionamiento
El script funciona de la siguiente manera:

Carga el modelo CLIP usando Sentence-Transformers.
Para cada imagen en el conjunto de datos, calcula su representación utilizando el modelo CLIP.
Para cada clase, calcula la similitud de coseno entre la representación de la imagen y la representación de cada imagen en esa clase.
Dependiendo del modo seleccionado ('max' o 'avg'), se selecciona la similitud de coseno máxima o promedio entre la imagen y cada clase.
La imagen se clasifica en la clase con la similitud de coseno máxima o promedio más alta.
La imagen se mueve a la subcarpeta correspondiente a su clase en el conjunto de datos de salida.

## Notas
Es importante tener en cuenta que este script moverá las imágenes originales al conjunto de datos de salida. Si desea mantener las imágenes originales en su lugar, puede modificar el script para copiar las imágenes en lugar de moverlas.

Además, el script puede requerir una gran cantidad de memoria y poder de cómputo si se manejan conjuntos de datos grandes o imágenes de alta resolución, debido a la naturaleza intensiva en recursos del modelo CLIP.
