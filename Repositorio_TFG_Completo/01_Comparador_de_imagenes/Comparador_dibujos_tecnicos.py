# comparar_dibujos_tecnicos_v2.py
import cv2
import numpy as np

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

def analizar_comparacion(titulo, imagen_a, imagen_b):
    """
    Función auxiliar para procesar, comparar y mostrar resultados
    entre dos imágenes dadas.
    """
    print(f"--- Procesando: {titulo} ---")
    
    # 1. Asegurar dimensiones (redimensionar B al tamaño de A)
    if imagen_a.shape != imagen_b.shape:
        imagen_b = cv2.resize(imagen_b, (imagen_a.shape[1], imagen_a.shape[0]))

    # 2. Convertir a escala de grises
    gray_a = cv2.cvtColor(imagen_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(imagen_b, cv2.COLOR_BGR2GRAY)

    # 3. Calcular SSIM
    score, diff_map = calcular_ssim(gray_a, gray_b)
    print(f"Índice SSIM ({titulo}): {score:.4f}")

    # 4. Normalizar mapa de diferencias para visualización
    diff_norm = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # 5. Generar máscara de zonas distintas
    # Umbral de 180 (ajustable): zonas oscuras en SSIM son diferencias
    _, mask = cv2.threshold(diff_norm, 180, 255, cv2.THRESH_BINARY_INV)

    # 6. Buscar contornos y dibujar en la imagen original
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    resultado_visual = imagen_a.copy()
    for c in contours:
        if cv2.contourArea(c) < 20: # Filtrar ruido pequeño
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(resultado_visual, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 7. Mostrar resultados en ventanas con nombres únicos
    cv2.imshow(f"{titulo} - Original", imagen_a)
    cv2.imshow(f"{titulo} - Comparada", imagen_b)
    cv2.imshow(f"{titulo} - Diferencias Detectadas", resultado_visual)
    
    # (Opcional) Ver el mapa de calor SSIM
    # cv2.imshow(f"{titulo} - Mapa SSIM", diff_norm) 

# === BLOQUE PRINCIPAL ===

# Cargar imágenes
# Nota: Según tu código, img1 es "imagen2.png" y img2 es "imagen3.png"
img_referencia = cv2.imread("imagen2.png") # img1 original
img_objetivo = cv2.imread("imagen3.png")   # img2 original

if img_referencia is None or img_objetivo is None:
    raise FileNotFoundError("No se han encontrado las imágenes.")

# CASO 1: Comparar Imagen 2 vs Imagen 3 (Lo que ya tenías)
analizar_comparacion("Caso 1 (Img2 vs Img3)", img_referencia, img_objetivo)

print("\n" + "="*30 + "\n")

# CASO 2: Comparar Imagen 3 con ella misma (Lo nuevo que pediste)
# Usamos img_objetivo (imagen3.png) para ambas entradas
analizar_comparacion("Caso 2 (Img3 vs Img3)", img_objetivo, img_objetivo)

print("\nCierra las ventanas para finalizar.")
cv2.waitKey(0)
cv2.destroyAllWindows()