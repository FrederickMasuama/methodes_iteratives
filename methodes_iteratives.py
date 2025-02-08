import numpy as np
import matplotlib.pyplot as plt

#Matrices
A = np.array([[10, 2, 1], #Matrice symétrique à diagonale fortement dominant
     [2, 10, 3],
     [1, 3, 10]])

b_1 = np.array([[15, 29, 27]])#sans oublier les doubles crochets
b = b_1.T #transposée du vecteur

"""Random matrices: Matrices à diagonales fortement dominant"""
#A = np.random.rand(5, 5) + np.diag(4*np.ones(5))
#b = np.random.rand(5)

# Données
max_iter = 1000
tol = 1e-6
x = np.zeros_like(b , dtype = float)

# Méthode de Jacobi
def jacobi(A, b, x, max_iter, tol):
    M = np.diag(np.diag(A))
    N = - ( np.tril(A, k=-1) + np.triu(A, k=1) )
    M_inv = np.linalg.inv(M)
    num_iter = 0
    
    for i in range(max_iter):
        x_new = ( M_inv @ N ) @ x + M_inv @ b #itérations
        num_iter += 1
        
        """conditions d'arrêt"""
        if np.linalg.norm(x_new - x, ord = 2) < tol:
            break
        if num_iter == max_iter :
            print("Nous avons atteint le nombre max d'itérations sans convergence")

        x = x_new
        
    return [x, num_iter]


#Méthode de Gauss-Seidel
def gauss_seidel(A, b, x, max_iter, tol):
    M = np.diag(np.diag(A)) + np.tril(A, k=-1)
    N = - np.triu(A, k=1)
    M_inv = np.linalg.inv(M)
    num_iter = 0

    for i in range(max_iter):
        x_new = ( M_inv @ N ) @ x + M_inv @ b #itérations
        num_iter += 1

        """conditions d'arrêt"""
        if np.linalg.norm(x_new - x, ord = 2) < tol:
            break
        if num_iter == max_iter :
            print("Nous avons atteint le nombre max d'itérations sans convergence")

        x = x_new
        
    return [x, num_iter]


#Méthode de relaxation
def relax(A, b, x, max_iter, tol, w):
    num_iter = 0
    M = (1/w)*np.diag(np.diag(A)) + np.tril(A, k=-1)
    Nn = ((1-w)/w)*np.diag(np.diag(A)) - np.triu(A, k=1)
    M_inv = np.linalg.inv(M)
    Lw = M_inv @ Nn
   

    for i in range(max_iter) :
        x_new = Lw @ x + M_inv @ b  #itérations
        num_iter += 1
    
        """conditions d'arrêt"""
        if np.linalg.norm(x_new - x, ord = 2) < tol:
            break
        if num_iter == max_iter :
            print("Nous avons atteint le nombre max d'itérations sans convergence")

        x = x_new
        
    return [x, num_iter]



#Solutions
"""Jacobi"""
jac = jacobi(A, b, x, max_iter, tol)
res = jac[0]
it = jac[1]

"""Gauss-Seidel"""
G_S = gauss_seidel(A, b, x, max_iter, tol)
resultat = G_S[0]
nb_iter = G_S[1]

"""Relaxation"""
rel = relax(A, b, x, max_iter, tol, 1.065)
result = rel[0]
nb_it = rel[1]


#affichages
print ("La matrice A :\n", A)
print ("La matrice b :\n", b)
print()
print("le vecteur solution par Jacobi est : \n", res, "et est obtenu après ", it, " itérations")
print()
print("le vecteur solution par Gauss-Seidel est : \n", resultat, "et est obtenu après ", nb_iter, " itérations")
print()
print("le vecteur solution par Relaxation est : \n", result, "et est obtenu après ", nb_it, " itérations")
