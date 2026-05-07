import cv2
import numpy as np
import os
import config

def create_side_by_side_comparison(img1_path, img2_path, output_path):
    # 1. Citim imaginile
    img_original = cv2.imread(img1_path)
    img_processed = cv2.imread(img2_path)

    if img_original is None or img_processed is None:
        print(f"Eroare: Nu am putut găsi imaginile la căile specificate!")
        return

    # 2. Ne asigurăm că ambele imagini au exact aceeași înălțime (vital pentru hconcat)
    # Extragem dimensiunile (înălțime, lățime)
    h1, w1 = img_original.shape[:2]
    h2, w2 = img_processed.shape[:2]

    # Dacă înălțimile diferă (de ex. după preprocesare s-a schimbat rezoluția), 
    # redimensionăm imaginea procesată la înălțimea celei originale
    if h1 != h2:
        # Păstrăm proporția lățimii (aspect ratio)
        new_w2 = int(w2 * (h1 / h2))
        img_processed = cv2.resize(img_processed, (new_w2, h1))

    # 3. Ne asigurăm că au același număr de canale de culoare (ex. ambele să fie BGR, nu una grayscale)
    if len(img_original.shape) == 3 and len(img_processed.shape) == 2:
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)
    elif len(img_original.shape) == 2 and len(img_processed.shape) == 3:
        img_original = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)

    # 4. Lipim imaginile orizontal
    combined_img = cv2.hconcat([img_original, img_processed])

    # 5. Opțional: Putem desena o linie neagră fină între ele ca să se vadă clar despărțitura
    # Grosimea liniei: 2 pixeli
    h, w = combined_img.shape[:2]
    w_original = img_original.shape[1]
    cv2.line(combined_img, (w_original, 0), (w_original, h), (0, 0, 0), 2)

    # 6. Salvăm rezultatul
    cv2.imwrite(output_path, combined_img)
    print(f"Imaginea comparativă a fost salvată cu succes la: {output_path}")

# --- RULAREA SCRIPTULUI ---
# Va trebui să pui căile corecte către o poză din fiecare set (Temple, Dino, Kendama)

if __name__ == "__main__":
    # Exemplu pentru dataset-ul Kendama
    create_side_by_side_comparison(
        img1_path=os.path.join(config.BASE_DIR, "data\\dinoR0001.png"),
        img2_path=os.path.join(config.BASE_DIR, "data\\dino_processed.png"),
        output_path='dino_split.png'
    )