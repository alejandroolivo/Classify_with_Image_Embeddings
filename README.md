# Proyecto de Clasificación de Imágenes con Embeddings Mixtos
Este es un script de Python para clasificar imágenes utilizando el modelo CLIP.

## Descripción
El objetivo central de este proyecto es clasificar imágenes haciendo uso de embeddings mixtos, combinando tanto características visuales como semánticas. La idea detrás de los embeddings de imágenes, en particular, es convertir las imágenes en representaciones vectoriales de alta dimensión que encapsulen su contenido. Estas representaciones, conocidas como "embeddings", permiten que las imágenes se comparen y clasifiquen con facilidad.

![Capacitación: cálculo embeddings de imágenes patrón](./Info/Screenshot%20-%20capacitacion.PNG)

La arquitectura CLIP (Contrastive Language–Image Pre-training) se destaca en esta tarea ya que ha sido entrenada con una amplia variedad de imágenes y textos, aprendiendo a asociar imágenes y palabras en un espacio de embeddings común. Este proyecto aprovecha este aprendizaje transferible para clasificar imágenes en diferentes categorías, basándose en la similitud semántica entre las imágenes desconocidas y un conjunto de imágenes de referencia.

![Clasificación: calculo embeddings de imágenes a clasificar y nueva imagen](./Info/Screenshot%20-%20clasificacion.PNG)

El proceso de fine-tuning o afinamiento es esencial para adaptar modelos preentrenados, como el Vision Transformer (ViT), a conjuntos de datos específicos o dominios particulares. Al realizar fine-tuning, podemos beneficiarnos de la rica representación aprendida por el modelo en tareas anteriores y adaptarla para rendir al máximo en nuestro conjunto de datos objetivo.

![Fine-Tunning](./Info/Screenshot%20-%20fine-tuning.PNG)

Una variación interesante del enfoque original es comparar embeddings de imágenes con embeddings de texto. En lugar de clasificar imágenes comparándolas con imágenes de referencia, en este enfoque, las imágenes se comparan directamente con descripciones textuales de clases. Esto permite que el modelo clasifique imágenes en categorías que no tienen imágenes de referencia, siempre que se proporcione una descripción textual de la clase.

## Estructura del Proyecto

### Scripts Principales
- ModelTrainingWithCustomImages.py: Script para entrenar el modelo Vision Transformer (ViT) con un conjunto de datos específico.
- ClassifyImagesWithEmbeddings.py: (El script que proporcionaste) Utiliza el modelo SentenceTransformer con CLIP para clasificar imágenes en función de su similitud con imágenes de referencia.
- ClassifyImagesWithSentenceTransformers.py: Utiliza el modelo SentenceTransformer con CLIP para clasificar imágenes en función de su similitud con descripciones textuales de clases.

### Carpetas de Datos
- ./Data/[dataset_name]/Clases: Contiene carpetas para cada clase, cada una con imágenes de referencia para esa clase.
- ./Data/[dataset_name]/Images: Contiene imágenes que se clasificarán y se moverán a las carpetas de clases correspondientes.
- ./Data/[dataset_name]/Dataset: Lugar donde se moverán las imágenes clasificadas.

## Cómo usar
Preparativos
Asegúrate de tener todas las dependencias instaladas. Este proyecto utiliza sentence_transformers, PIL, numpy, entre otros.

Configura tus carpetas de datos siguiendo la estructura mencionada anteriormente.

## Clasificación de Imágenes

### Para clasificar imágenes usando el script ClassifyImagesWithEmbeddings.py:

Establece el nombre del dataset que deseas clasificar al principio del script.
Elige el mode de clasificación entre 'avg' y 'max'.
Ejecuta el script. Las imágenes de ./Data/[dataset_name]/Images serán clasificadas y movidas a sus respectivas carpetas en ./Data/[dataset_name]/Dataset.


### Para clasificar imágenes usando el script ClassifyImagesWithSentenceEmbeddings.py:

- Funcionamiento:
Se generan embeddings semánticos para descripciones textuales de cada clase (por ejemplo, 'vestido', 'pantalones', 'camiseta'). Estas descripciones son tomadas de los nombres de las carpetas en la carpeta ./Data/[nombre_del_dataset]/Clases.
Para cada imagen en el conjunto de datos, se genera un embedding semántico utilizando el modelo CLIP.
La similitud semántica entre el embedding de la imagen y los embeddings de texto se calcula utilizando la similitud del coseno.
La imagen se clasifica en la clase cuya descripción textual tenga la mayor similitud con la imagen.
Finalmente, las imágenes se organizan en subcarpetas dentro de ./Data/[nombre_del_dataset]/Dataset, basado en estas similitudes semánticas con las descripciones textuales.

- Ventajas:
Este enfoque permite una flexibilidad increíble en la definición y modificación de clases simplemente añadiendo o eliminando carpetas y ajustando sus nombres para reflejar diferentes descripciones de clase.
También evita la necesidad de tener imágenes de referencia para cada clase, ya que las descripciones textuales sirven como referencia semántica.
