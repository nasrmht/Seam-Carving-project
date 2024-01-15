import numpy as np
from scipy.ndimage.filters import convolve

def energy_func(image):
    #Dérivé par rapport à x
    dx_filter = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    #conversition pour qu'il soit un filtre 3D
    dx_filter = np.stack([dx_filter]*3,axis=2)

    
    #Dérivée par rapport à y
    dy_filter = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    #conversition pour qu'il soit un filtre 3D
    dy_filter = np.stack([dy_filter]*3, axis=2)

    
    #Convolution de l'image avec nos deux noyaux pour la dérivation
    image = image.astype('float32')
    convolved = np.absolute(convolve(image, dx_filter)) + np.absolute(convolve(image, dy_filter))

    # On somme pour définir la fonction d'énergie
    energy_map = convolved.sum(axis=2)

    return energy_map


#Finding the seam with least energy
def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = energy_func(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=int)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index -1
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack


# Recherche du seam optimal avec contrainte
def find_seam_with_constraint(energy_matrix, constraint_matrix):
    # Taille de l'image
    rows, cols = energy_matrix.shape

    # Initialisation de la matrice des coûts cumulatifs
    cumulative_cost_matrix = np.zeros_like(energy_matrix, dtype=float)

    # Initialisation de la première ligne de la matrice des coûts cumulatifs
    cumulative_cost_matrix[0, :] = energy_matrix[0, :]

    # Mise à jour de la matrice des coûts cumulatifs
    for i in range(1, rows):
        for j in range(cols):
            # Calcul du coût cumulatif en ajoutant le coût minimum voisin
            min_neighbor = min(cumulative_cost_matrix[i-1, max(j-1, 0):min(j+2, cols)])
            cumulative_cost_matrix[i, j] = energy_matrix[i, j] + min_neighbor

            # Ajout du coût supplémentaire pour les pixels à l'intérieur de la contrainte
            if constraint_matrix[i, j]:
                cumulative_cost_matrix[i, j] += float('inf')

    # Recherche du pixel avec le coût cumulatif minimal dans la dernière ligne
    min_cost_index = np.argmin(cumulative_cost_matrix[-1, :])

    # Recherche du seam optimal en remontant depuis le pixel avec le coût minimal
    seam_row = [rows - 1]
    seam_col = [min_cost_index]
    seam = [seam_row, seam_col]
    for i in range(rows - 2, -1, -1):
        j = seam[1][-1]
        min_neighbor = min(cumulative_cost_matrix[i, max(j-1, 0):min(j+2, cols)])
        min_cost_index = np.argmin(cumulative_cost_matrix[i, max(j-1, 0):min(j+2, cols)]) + max(j-1, 0)
        seam_row.append(i)
        seam_col.append(min_cost_index)

    return seam

#Suppression des pixels de la 
def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)

    # Create a (r, c) matrix filled with the value True
    # We'll be removing all pixels from the image which
    # have False later
    mask = np.ones((r, c), dtype=bool)

    # Find the position of the smallest element in the
    # last row of M
    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        # Mark the pixels for deletion
        mask[i, j] = False
        j = backtrack[i, j]

    # Since the image has 3 channels, we convert our
    # mask to 3D
    mask = np.stack([mask] * 3, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = img[mask].reshape((r, c - 1, 3))

    return img


#Suppression des pixels du seam de l'image
def carve_column_cons(img, KK):
    r, c, _ = img.shape

    img_ener = energy_func(img)
    min_seam = find_seam_with_constraint(img_ener,KK)

    # Create a (r, c) matrix filled with the value True
    # We'll be removing all pixels from the image which
    # have False later
    mask = np.ones((r, c), dtype=bool)

    # Find the position of the smallest element in the
    # last row of M
    j = min_seam[1][0]

    for i in range(r):
        # Mark the pixels for deletion
        mask[i, j] = False
        j = min_seam[1][i]

    # Since the image has 3 channels, we convert our
    # mask to 3D
    mask = np.stack([mask] * 3, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = img[mask].reshape((r, c - 1, 3))

    return img


## une fonction qui réunit toutes les étapes pour le seam carving simple
def crop_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in range(c - new_c): # use range if you don't want to use tqdm
        img = carve_column(img)

    return img


## une fonction qui réunit toutes les étapes pour le seam carving complexe
def crop_cc(img,KK, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in range(c - new_c): # use range if you don't want to use tqdm
        img = carve_column_cons(img,KK)
    
    return img