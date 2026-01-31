import cv2
import numpy as np
import os


DOSSIER_SCRIPT = os.path.dirname(os.path.abspath(__file__))


#### Exercice 1 :

def meanFilter(img, k) :
    """
    Cette fonction applique un filtre moyenneur sur une image.
    Cela consiste à remplacer chaque pixel de l'image par la moyenne des pixels dans une fenêtre glissante
    de taille k*k et dont le centre est centré sur le pixel en cours de traitement.
    Ce filtre permet de réduire le bruit et flouter une image.

    Ici, on choisit de ne pas prendre en compte les pixels en dehors de bords de l'image dans nos calculs.

    :param image: image d'entrée (couleur BGR ou niveaux de gris)
    :param k: taille de la fenêtre (doit être un nb impair)
    """
    #on passe l'image en niveaux de gris
    image = img.copy()
    if len(image.shape) == 3 :
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = image.shape #on récupère les dimensions de l'image
    res = np.zeros_like(image) #copie "vide" de l'image originale

    k_central = k//2 #rayon du pixel central dans la fenêtre, va servir au déplacement dans la fenêtre de convolution

    #se déplace dans l'image
    for y in range(height) :
        for x in range(width) :

            sum_val = 0 #va contenir la somme des valeurs des pixels
            nb_val = 0 #va compter le nb de valeurs additionnées pour pouvoir plus tard diviser sum_val avec

            #se déplace dans la fenêtre 
            for wy in range(-k_central, k_central+1) :
                for wx in range(-k_central, k_central+1) :
                    newY = y+wy #nouvel index
                    newX = x+wx #nouvel index

                    #On ne prend pas en compte les pixels en dehors des bords
                    if 0<=newY<height and 0<=newX<width : 
                        sum_val +=image[newY, newX]
                        nb_val+=1
            res[y,x] = sum_val//nb_val
    
    return res






#### Exercice 2 :

def convolution(img, kernel, normalize=True) :
    """
    On applique une convolution sur une image en ayant choisit le noyau au préalable.
    C'est une généralisation du filtre moyenneur. On suit donc le même principe que pour meanFilter.
    
    :param img: image d'entrée
    :param kernel: matrice qui correspond au noyau de convolution
    :param normalize: permet de savoir si on normalise ou non (divise par le nb de pixels scannés)
    """
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = img.copy()
    if len(image.shape) == 3 :
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = image.shape
    k_height, k_width = kernel.shape #on récupère les dimensions du noyau
    mkh, mkw = k_height//2, k_width//2 #va servir au déplacement dans la fenêtre du noyau

    res = np.zeros_like(image) #copie "vide" de l'image originale

    #on va scanner l'image
    #on applique le même principe que pour le filtre médian
    for y in range(height) :
        for x in range(width) :
            sum_val = 0
            nb_val = 0
            #se déplace dans la fenêtre de convolution
            for wy in range(-mkh, mkh+1) :
                for wx in range(-mkw, mkw+1) :
                    newY = y+wy #nouvel index
                    newX = x+wx #nouvel index

                    #On ne prend pas en compte les pixels en dehors des bords
                    if 0<=newY<height and 0<=newX<width :
                        sum_val += image[newY, newX]*kernel[wy+mkh, wx+mkw] #applique la formule : multiplie la valeur du pixel dans l'image par celle du noyau
                        nb_val+=1
            #Pour les filtres où il faut normaliser :
            if normalize : 
                res[y,x] = sum_val/nb_val
            else : 
                res[y,x] = sum_val
    
    return res






#### Exercice 3 :

def edgeSobel(img) :
    """
    On utilise la méthode d Sobel pour calculer la norme du gradient
    et ainsi détecter les contours d'une image.
    On calcule les gradients selon x et y et on fait ensuite la somme
    de leurs valeurs absolues. 
    
    :param img: image d'entrée
    """
    
    #noyau par rapport à x (dérivée selon x)
    dx_kernel = np.array([[-1,0,1], [-2,0,2], [-1,0,1]], dtype=np.float32)
    #noyau oar rapport à y (dérivée selon y)
    dy_kernel = np.array([[1,2,1], [0,0,0], [-1,-2,-1]], dtype=np.float32)

    #calcul la première dérivée, par rapport à x
    conv1 = convolution(img, dx_kernel, normalize=False) #le kernel de Sobel n'est pas normalisé, donc ne doit pas diviser par nb_val
    #calcul la deuxième dérivée, par rapport à y
    conv2 = convolution(img, dy_kernel, normalize=False)

    #calcul la norme du gradient avec la valeur absolue
    res = np.abs(conv1) + np.abs(conv2)

    return res




#### Exercice 4 :


# Pas fait


#### Exercice 5 :

def median(img, k) :
    """
    Cette fonction applique le filtre médian sur une image.
    Cela consiste à remplacer chaque pixel par la valeur médiane des pixels dans 
    une fenêtre de taille k*k et dont le centre est centré sur le pixel traité.
    
    :param img: image d'entrée
    :param k: taille de la fenêtre
    """
    image = img.copy()
    if len(image.shape) == 3 :
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = image.shape
    res = np.zeros_like(image)

    k_centre = k//2
    wind_size = k*k
    mid_wind = wind_size//2

    #se déplace dans l'image
    for y in range(height) :
        for x in range(width) :
            values_list = [] #cette liste va contenir les valeurs des pixels scannés

            #se déplace dans la fenêtre
            for wy in range(-k_centre, k_centre+1) :
                for wx in range(-k_centre, k_centre+1) :
                    ny = y+wy #nouvel index
                    nx = x+wx #nouvel index

                    #ne prend pas en compte les pixels en dehors des bords de l'image
                    if 0<=ny<height and 0<=nx<width :
                        values_list.append(image[ny,nx])
            
            #trie la liste des valeurs
            values_list.sort()
            res[y,x] = values_list[len(values_list)//2] #prend la médiane
    
    return res





############# TESTS : #############


# Test sur le filtre moyenneur
image1 = cv2.imread(os.path.join(DOSSIER_SCRIPT,"camera_bruit_gaussien.png"))
test_meanFilter = meanFilter(image1, 3)
cv2.imshow("Mean filter",test_meanFilter)
cv2.waitKey(0)
cv2.destroyAllWindows()




# Test sur la convolution
image2 = cv2.imread(os.path.join(DOSSIER_SCRIPT,"cat.png"))
#Différents kernels :
laplacien = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]], dtype=np.float32)
gaussian_5x5_sigma1 = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
], dtype=np.float32) / 256.0
filtre_moyenneur = np.array(np.ones((3,3), dtype=np.float32))
#test avec le filtre Laplacien
test_conv = convolution(image2, laplacien)
cv2.imshow("Convolution",test_conv)
cv2.waitKey(0)
cv2.destroyAllWindows()


#img_opencv = cv2.filter2D(image2, ddepth=-1, kernel=filtre_moyenneur)
#cv2.imshow("Convolution",img_opencv)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



# Test sur Sobel
image3 = cv2.imread(os.path.join(DOSSIER_SCRIPT,"camera.png"))
test_Sobel = edgeSobel(image3)
cv2.imshow("Sobel",test_Sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Test sur le filtre médian
image5 = cv2.imread(os.path.join(DOSSIER_SCRIPT,"camera_bruit_poivre_et_sel.png"))
test_median = median(image5, 3)
cv2.imshow("Median",test_median)
cv2.waitKey(0)
cv2.destroyAllWindows()



