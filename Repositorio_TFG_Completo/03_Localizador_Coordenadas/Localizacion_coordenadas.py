# Localización_coordenadas.py
# Localiza las coordenadas de la sección que deseamos hallar. 
# Autor: Gabriel Gómez Silva.
# Proyecto: TFG_Aplicación de la IA para corregir exámenes de Dibujo Técnico.

import cv2

NOMBRE_IMAGEN = 'solucion_profesor.jpg' 

# Cargar imagen
img = cv2.imread(NOMBRE_IMAGEN)

if img is None:
    print(f"Error: No encuentro la imagen {NOMBRE_IMAGEN}")
else:
    print("INSTRUCCIONES:")
    print("1. Se abrirá la imagen.")
    print("2. Pincha y arrastra el ratón para rodear la nota (dibuja el recuadro).")
    print("3. Pulsa la tecla ENTER o ESPACIO para confirmar.")
    print("4. Pulsa 'c' si quieres cancelar y volver a dibujar.")
    
    # Esta función abre una ventana y te deja dibujar
    cv2.namedWindow("Selecciona la Nota", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Selecciona la Nota", img, showCrosshair=True, fromCenter=False)
    
    # Cerrar ventana
    cv2.destroyAllWindows()

    # Imprimir los valores listos para copiar
    x, y, w, h = roi
    
    print("\n" + "="*40)
    print("¡COORDENADAS OBTENIDAS!")
    print("Copia y pega esto en tu script principal:")
    print("="*40)
    print(f"ROI_X = {x}")
    print(f"ROI_Y = {y}")
    print(f"ROI_W = {w}")
    print(f"ROI_H = {h}")
    print("="*40)