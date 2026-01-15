# TFG_Aplicación de la IA para corregir exámenes de Dibujo Técnico.
# Aplicación de modelos de Inteligencia Artificial para la corrección automática de ejercicios por imagen.

Repositorio oficial del Trabajo de Fin de Grado (TFG) presentado por **Gabriel Gómez Silva** para el grado en **Ingeniería Mécanica** de la **Escuela de Ingenierías Industriales de la Universidad de Extremadura**.

## 📋 Descripción
Este proyecto desarrolla la utilización de la Inteligencia Artificial para poder llegar a corregir distintos ejercicios de Dibujo Técnico, empleando diversas herramientas como puede ser la  **Visión Artificial (OpenCV)** y **OCR (EasyOCR)**.

## 🚀 Tecnologías utilizadas
* **Lenguaje:** Python 3.11.9.
* **Librerías principales:** OpenCV, Pandas, EasyOCR, NumPy.

## 📂 Estructura del proyecto
## El proyecto se encuentra dividido en varias carpetas, donde cada una de ellas recoge un código desarrollado para la resolución de las distintas cuestiones que se comentan en el trabajo:
* Comparador_de_imagenes: En esta carpeta se realiza la comparación de dos imágenes iguales y dos dintintas, dando como resultado una serie de fotos señalando los errores y un número (SSIM), que resume lo parecido que son los dibujos.
* Puntuador_examenes: Aqui se realiza la puntuación tanto del examen del alumno en comparación con la corrección del profesor como la verificación de que se puntúa de forma correcta, comparando el examen del profesor con su solución.
* Localizador_Coordenadas: Este código de basa en mostrar de forma exacta la localización de las coordenadas, en donde se encuentran las notas de los exámenes.
* Alineador_y_puntuador_examenes: Se basa en el desarrollo de la alineación y limpiado de las imágenes para el desarrolo de la primera versión de la IA. Extrayendo además las notas correspondientes de cada uno de los exámenes. 

## ⚠️ Nota sobre Privacidad
Debido a la Ley de Protección de Datos, el dataset de imágenes de exámenes reales de alumnos no se incluye en este repositorio público. El código funciona con cualquier conjunto de imágenes que siga la plantilla de referencia.
