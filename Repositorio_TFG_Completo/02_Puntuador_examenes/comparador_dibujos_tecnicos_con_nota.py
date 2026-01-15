# comparar_dibujos_notas_v2.py
# Comparación de dibujos técnicos mediante SSIM y asignación de nota (Con Validación)
# Autor: Gabriel Gómez Silva (Modificado)
# Proyecto: TFG

import cv2
import numpy as np

# === FUNCIONES AUXILIARES ===

def calcular_ssim(img1, img2):
    """Cálculo manual del índice SSIM entre dos imágenes en escala de grises."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Constantes de estabilidad
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Medias y desviaciones
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean(), ssim_map

def asignar_nota(ssim_score, umbral_aprobado, umbral_notable, umbral_sobresaliente):
    """
    Asigna una nota del 0 al 10 basada en el índice SSIM.
    """
    if ssim_score >= umbral_sobresaliente:
        # Interpolación lineal entre 9 y 10
        nota = 9 + (ssim_score - umbral_sobresaliente) / (1 - umbral_sobresaliente)
    elif ssim_score >= umbral_notable:
        # Interpolación lineal entre 7 y 9
        nota = 7 + (ssim_score - umbral_notable) / (umbral_sobresaliente - umbral_notable) * 2
    elif ssim_score >= umbral_aprobado:
        # Interpolación lineal entre 5 y 7
        nota = 5 + (ssim_score - umbral_aprobado) / (umbral_notable - umbral_aprobado) * 2
    else:
        # Interpolación lineal entre 0 y 5
        nota = ssim_score / umbral_aprobado * 5

    return round(max(0, min(10, nota)), 2)

def evaluar_ejercicio(titulo, img_referencia, img_evaluar, umbrales):
    """
    Procesa la comparación, calcula la nota y muestra los resultados.
    """
    u_aprobado, u_notable, u_sobresaliente = umbrales
    
    print(f"\n--- Analizando: {titulo} ---")

    # 1. Redimensionar imagen a evaluar si es necesario
    if img_evaluar.shape != img_referencia.shape:
        img_evaluar = cv2.resize(img_evaluar, (img_referencia.shape[1], img_referencia.shape[0]), interpolation=cv2.INTER_AREA)

    # 2. Convertir a escala de grises
    gray_ref = cv2.cvtColor(img_referencia, cv2.COLOR_BGR2GRAY)
    gray_eval = cv2.cvtColor(img_evaluar, cv2.COLOR_BGR2GRAY)

    # 3. Calcular SSIM
    score, diff_map = calcular_ssim(gray_ref, gray_eval)
    
    # 4. Asignar Nota
    nota = asignar_nota(score, u_aprobado, u_notable, u_sobresaliente)
    
    print(f"SSIM Score: {score:.4f}")
    print(f"NOTA FINAL: {nota}/10")

    # 5. Visualización de diferencias
    diff_norm = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    _, mask = cv2.threshold(diff_norm, 180, 255, cv2.THRESH_BINARY_INV)

    resultado_visual = img_evaluar.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        if cv2.contourArea(c) < 50:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(resultado_visual, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 6. Mostrar ventanas (con prefijo del título para no sobrescribir)
    cv2.imshow(f"[{titulo}] - Evaluacion", resultado_visual)
    # Opcional: Mostrar mapa de calor
    # cv2.imshow(f"[{titulo}] - Mapa SSIM", diff_norm)


# === CONFIGURACIÓN ===
IMAGEN_SOLUCION = "imagen_solucion.png"
IMAGEN_ALUMNO = "imagen_alumno.png"

# Tupla de umbrales: (Aprobado, Notable, Sobresaliente)
UMBRALES = (0.88, 0.95, 0.98) 

# === BLOQUE PRINCIPAL ===

# Cargar imágenes
img_solucion = cv2.imread(IMAGEN_SOLUCION)
img_alumno = cv2.imread(IMAGEN_ALUMNO)

if img_solucion is None or img_alumno is None:
    raise FileNotFoundError("Error cargando las imágenes. Verifica los nombres de archivo.")

# ---------------------------------------------------------
# CASO 1: VALIDACIÓN (Profesor vs Profesor)
# ---------------------------------------------------------
# Aquí comparamos la solución consigo misma. 
# La nota DEBE ser 10.00 y el SSIM 1.0000.
evaluar_ejercicio("Validacion (Profesor vs Profesor)", img_solucion, img_solucion, UMBRALES)

# ---------------------------------------------------------
# CASO 2: EVALUACIÓN (Profesor vs Alumno)
# ---------------------------------------------------------
evaluar_ejercicio("Ejercicio Alumno", img_solucion, img_alumno, UMBRALES)


print("\nCierra las ventanas para finalizar.")
cv2.waitKey(0)
cv2.destroyAllWindows()