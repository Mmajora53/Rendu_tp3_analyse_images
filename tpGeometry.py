import cv2
import numpy as np
import math
import os


DOSSIER_SCRIPT = os.path.dirname(os.path.abspath(__file__))

#### Exercice 1 :

def transpose(image) :
    """
    On applique la transposée sur une image, donc on va échanger les lignes avec les colonnes.
    
    :param image: image d'entrée
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape

    #on inverse les dimensions
    res = np.zeros((width, height), dtype=image.dtype)

    for y in range(height) :
        for x in range(width) :
            res[y,x] = image[x,y] #on "inverse" les pixels : on inverse les lignes et les colonnes
    
    return res



#### Exercice 2 :

def interpolate_nearest(image, x,y) :
    """
    On effectue l'interpolation par plus proche voisin.
    On prend en entrée les coordonnées d'un pixel et on retourne la valeur du pixel
    le plus proche (en terme de coordonnées).
    
    On retourne la valeur du pixel. Si en dehors de l'image, on retourne 0 (noir)
    
    :param image: image d'entrée
    :param x: coordonnée x du pixel qu'on veut traiter
    :param y: coordonnée y du pixel qu'on veut traiter
    """
    height, width = image.shape[:2]

    y_arr = int(np.floor(y+0.5)) #arrondi inférieur
    x_arr = int(np.floor(x+0.5)) #arrondi inférieur

    #si l'arrondi est hors des dimensions de l'image
    if y_arr<0 or x_arr<0 or y_arr>=height or x_arr>=width :
        return 0
    
    return image[y_arr, x_arr]


def expand(image, facteur) :
    """
    Agrandit l'image par un facteur (ex: 3 = 3 fois plus large et haut)
    en utilisant l'interpolation nearest neighbor
    
    :param image: image d'entrée
    :param facteur: coeff d'agrandissement
    """
    #on prend en entrée des images de couleurs ou grises
    height, width = image.shape[:2]
    #nouvelles dimensions
    n_height = int(height*facteur) 
    n_width = int(width*facteur)

    if len(image.shape) == 2: #si une image en niveaux de gris
        res = np.zeros((n_height, n_width), dtype=image.dtype)
    else: #si une image en couleurs
        res = np.zeros((n_height, n_width, image.shape[2]), dtype=image.dtype)

    #on parcours l'image
    for y in range(n_height) :
        for x in range(n_width) :
            img_y = (y+0.5)/facteur - 0.5 #on va calculer la position correspondant dans l'image originale
            img_x = (x+0.5)/facteur - 0.5

            res[y,x] = interpolate_nearest(image, img_x, img_y) #on applique l'inetrpolation par plus proche voisin
    
    return res





#### Exercice 3 :


def interpolate_bilinear(image, x,y) :
    """
    Effectue l'interpolation bilinéaire d'un point de coordonnées (x, y).
    
    :param image: image d'entrée
    :param x: coordonnée x du pixel qu'on veut traiter
    :param y: coordonnée y du pixel qu'on veut traiter
    """
    height, width = image.shape[:2]

    #on calcule les coordonnées des 4 pixels entourant le pixel au point (x,y)
    x0 = int(np.floor(x)) #colonne gauche
    x1 = x0+1 #Colonne droite
    y0 = int(np.floor(y)) #ligne haute
    y1 = y0+1 #ligne basse

    #Si on est en dehors des limites de l'image
    if x0<0 or y0<0 or x1>=width or y1>=height :
        return 0
    
    #calcule les coeff alpha et beta
    alpha = x-x0
    beta = y-y0

    #on récupère les valeurs des pixels voisins
    fy0x0 = image[y0, x0]
    fy0x1 = image[y0, x1]
    fy1x0 = image[y1, x0]
    fy1x1 = image[y1, x1]

    #on applique la formule
    res = (1-alpha)*(1-beta)*fy0x0 + alpha*(1-beta)*fy0x1 + (1-alpha)*beta*fy1x0 + alpha*beta*fy1x1

    return res



def expand_bilinear(image, facteur) :
    """
    Agrandit l'image avec interpolate_bilinear. 
    On a repris expand et on l'a modifiée pour appliquer interpolate bilinear.
    
    :param image: image d'entrée
    :param facteur: coeff d'agrandissement
    """

    height, width = image.shape[:2]

    #nouvelles dimensions
    n_height = int(height*facteur) 
    n_width = int(width*facteur)

    if len(image.shape) == 2:
        res = np.zeros((n_height, n_width), dtype=image.dtype) #pour une image en niveau de gris
    else:
        res = np.zeros((n_height, n_width, image.shape[2]), dtype=image.dtype) #pour une image en couleurs

    for y in range(n_height) :
        for x in range(n_width) :
            img_y = (y+0.5)/facteur - 0.5 #on va calculer la position correspondant dans l'image originale
            img_x = (x+0.5)/facteur - 0.5

            res[y,x] = interpolate_bilinear(image, img_x, img_y) #on applique interpolate_bilinear
    
    return res






#### Exercice 4 :

def rotate(image, angle, interpolation) :
    """
    Effectue la rotation d'une image autour de son centre. On donne en paramètre quel type
    d'interpolation on veut utiliser.

    :param image: image d'entrée
    :param angle: angle de rotation
    :param interpolation: "nearest" ou "bilinear", permet de choisir l'interpolation à utiliser
    """
    height, width = image.shape[:2]
    mh = height/2
    mw = width/2

    #calcule les angles
    cos_angle = math.cos(math.radians(angle)) 
    sin_angle = math.sin(math.radians(angle))

    res = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            #coordonnées depuis le centre
            ny = y - mh
            nx = x - mw
            #chercher les coordonnées du pixel source
            src_x = cos_angle * nx + sin_angle * ny + mw
            src_y = -sin_angle * nx + cos_angle * ny + mh
            
            if interpolation == 'nearest':
                val = interpolate_nearest(image, src_x, src_y)
            else:
                val = interpolate_bilinear(image, src_x, src_y)
                            
            res[y, x] = val
    
    return res








############# TESTS : #############

# Test de la transposition
image1 = cv2.imread(os.path.join(DOSSIER_SCRIPT,"camera.png"))
test_transpose = transpose(image1)
cv2.imshow("Transpose :",test_transpose)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Test de l'interpolation avec nearest neighbor
image2 = cv2.imread(os.path.join(DOSSIER_SCRIPT,"cat.png"))
test_expand = expand(image2, 3)
cv2.imshow("Expand interpolate nearest :",test_expand)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Test de l'interpolation bilinear
image3 = cv2.imread(os.path.join(DOSSIER_SCRIPT,"cat.png"))
test_expand_bili = expand_bilinear(image3, 3)
cv2.imshow("Expand interpolate bilinear",test_expand_bili)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
ref = cv2.resize(image3, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
cv2.imshow("Référence OpenCV INTER_LINEAR ×3", ref)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# Test de la rotation
image4 = cv2.imread(os.path.join(DOSSIER_SCRIPT,"camera.png"))
test_rotate = rotate(image4, 45, "nearest")
cv2.imshow("Rotate nearest",test_rotate)
cv2.waitKey(0)
cv2.destroyAllWindows()



