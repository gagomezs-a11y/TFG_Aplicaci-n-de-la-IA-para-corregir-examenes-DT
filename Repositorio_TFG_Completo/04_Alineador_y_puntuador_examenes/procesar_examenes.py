import cv2
import numpy as np
import pandas as pd
import easyocr
import os

# --- CONFIGURACIÓN ---
CARPETA_ALUMNOS = 'examenes_alumnos/'
IMAGEN_REFERENCIA = 'solucion_profesor.jpg'
CARPETA_SALIDA = 'examenes_alineados/'
CSV_SALIDA = 'dataset_notas.csv'

# Coordenadas de la casilla de la nota 
ROI_X = 4192
ROI_Y = 5866
ROI_W = 440
ROI_H = 540

# Inicializar OCR
reader = easyocr.Reader(['es', 'en'], gpu=True) 

def leer_imagen_segura(ruta):
    """
    Lee imágenes en Windows aunque la ruta tenga tildes o caracteres especiales.
    Sustituye a cv2.imread(ruta)
    """
    try:
        # Leemos el archivo como bytes puros
        stream = open(ruta, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        # Decodificamos con OpenCV
        return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Error crítico leyendo {ruta}: {e}")
        return None

def guardar_imagen_segura(ruta, imagen):
    """
    Guarda imágenes en Windows aunque la ruta tenga tildes.
    Sustituye a cv2.imwrite(ruta, img)
    """
    try:
        exito, buffer = cv2.imencode(".jpg", imagen)
        if exito:
            with open(ruta, "wb") as f:
                f.write(buffer)
            return True
    except Exception as e:
        print(f"Error guardando {ruta}: {e}")
    return False

# ---------------------------------------------------------

def alinear_imagen(img_ref, img_target):
    """
    Alinea img_target para que coincida con img_ref usando ORB y Homografía.
    """
    # 1. Convertir a escala de grises
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR_GRAY)
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR_GRAY)

    # 2. Detectar características (ORB)
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_target, None)

    # 3. Emparejar características (Matcher)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    
    # Ordenar por calidad
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Top 20%
    num_good_matches = int(len(matches) * 0.2)
    matches = matches[:num_good_matches]

    # 4. Extraer coordenadas
    puntos_ref = np.zeros((len(matches), 2), dtype=np.float32)
    puntos_target = np.zeros((len(matches), 2), dtype=np.float32)

    for i, m in enumerate(matches):
        puntos_ref[i, :] = kp1[m.queryIdx].pt
        puntos_target[i, :] = kp2[m.trainIdx].pt

    # 5. Calcular Homografía y transformar
    h, mask = cv2.findHomography(puntos_target, puntos_ref, cv2.RANSAC, 5.0)
    
    alto, ancho = img_ref.shape[:2]
    img_alineada = cv2.warpPerspective(img_target, h, (ancho, alto))
    
    return img_alineada

def limpiar_texto_nota(texto):
    """
    Limpia el texto del OCR para dejar solo números decimales.
    """
    caracteres_validos = "0123456789,."
    texto_limpio = "".join([c for c in texto if c in caracteres_validos])
    texto_limpio = texto_limpio.replace(',', '.')
    
    try:
        valor = float(texto_limpio)
        if 0 <= valor <= 10:
            return valor
    except ValueError:
        pass
    
    return None

# --- BLOQUE PRINCIPAL ---

if not os.path.exists(CARPETA_SALIDA):
    os.makedirs(CARPETA_SALIDA)

# Cargar referencia (USANDO LA NUEVA FUNCIÓN)
ref_img = leer_imagen_segura(IMAGEN_REFERENCIA)

if ref_img is None:
    print(f"¡ERROR FATAL! No se encuentra la imagen de referencia: {IMAGEN_REFERENCIA}")
    exit()

datos = []

print(f"Comenzando procesamiento de imágenes en {CARPETA_ALUMNOS}...")

for archivo in os.listdir(CARPETA_ALUMNOS):
    if archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
        ruta_completa = os.path.join(CARPETA_ALUMNOS, archivo)
        
        # 1. CARGA SEGURA (Reemplaza cv2.imread)
        target_img = leer_imagen_segura(ruta_completa)
        
        # Verificación de seguridad por si falla la carga
        if target_img is None:
            print(f"AVISO: No se pudo leer {archivo} (posible archivo dañado), saltando...")
            continue

        try:
            # 2. ALINEACIÓN
            img_aligned = alinear_imagen(ref_img, target_img)
            
            # 3. GUARDADO SEGURO (Reemplaza cv2.imwrite)
            ruta_guardado = os.path.join(CARPETA_SALIDA, "aligned_" + archivo)
            guardar_imagen_segura(ruta_guardado, img_aligned)
            
            # 4. RECORTE DE LA NOTA (ROI)
            roi = img_aligned[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
            
            # Pre-procesamiento
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR_GRAY)
            _, roi_thresh = cv2.threshold(roi_gray, 120, 255, cv2.THRESH_BINARY_INV)
            
            # 5. LECTURA OCR
            lectura = reader.readtext(roi_thresh)
            
            nota_encontrada = "MANUAL"
            texto_crudo = ""
            
            for detection in lectura:
                texto = detection[1]
                texto_crudo += texto + " "
                val = limpiar_texto_nota(texto)
                if val is not None:
                    nota_encontrada = val
                    break 
            
            print(f"Procesado {archivo}: Nota detectada -> {nota_encontrada}")
            
            datos.append({
                'id_examen': archivo,
                'path_alineada': ruta_guardado,
                'ocr_raw': texto_crudo,
                'nota_profesor': nota_encontrada
            })
            
        except Exception as e:
            print(f"Error procesando lógica de {archivo}: {e}")

# Guardar CSV final
df = pd.DataFrame(datos)
df.to_csv(CSV_SALIDA, index=False)
print(f"¡Terminado! Revisa el archivo {CSV_SALIDA}")