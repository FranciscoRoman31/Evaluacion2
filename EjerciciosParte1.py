import numpy as np
import matplotlib.pyplot as plt

def generarMatriz(a,b):
    m = np.random.randint(1,100,(a,b))
    return m

def generaNumCond(a):
    cond = np.linalg.cond(a,1)
    return cond

def potenciaMatriz(a,b):
    matrizPow = np.linalg.matrix_power(a,b)
    return matrizPow

def potenciaMatrizAutoV(a,b, max_iter):
    autovalores = []
    autovectores = []
    matrizPow = np.linalg.matrix_power(a,b)
    for i in range(max_iter):
        matrizPow = np.linalg.matrix_power(a, b)
        autovalor, autovector = np.linalg.eig(matrizPow)
        
        autovalores.append(autovalor[0])
        autovectores.append(autovector[:,0])
        
        a = matrizPow
    return autovalores, autovectores

def metodo_potencia_simetrico(a, max_iter=100, tol=1e-10):
    n = a.shape[0]
    b_k = np.random.rand(n)
    b_k = b_k / np.linalg.norm(b_k)  # Normalizar el vector inicial

    autovalores = []
    for _ in range(max_iter):
        b_k1 = np.dot(a, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm

        autovalores.append(b_k1_norm)

        if len(autovalores) > 1 and abs(autovalores[-1] - autovalores[-2]) < tol:
            break
            
    autovalor_dominante = autovalores[-1]
    autovector_dominante = b_k
    return autovalor_dominante, autovector_dominante, autovalores

def metodo_potencia_inverso(a, max_iter=100, tol=1e-10):
    n = a.shape[0]
    b_k = np.random.rand(n)
    b_k = b_k / np.linalg.norm(b_k) 

    autovalores_inversos = []
    for _ in range(max_iter):
        try:
            b_k1 = np.linalg.solve(a, b_k)
        except np.linalg.LinAlgError:
            print("La matriz es singular o casi singular.")
            return None, None, autovalores_inversos
        
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm

        autovalor_inverso = 1 / b_k1_norm
        autovalores_inversos.append(autovalor_inverso)

        if len(autovalores_inversos) > 1 and abs(autovalores_inversos[-1] - autovalores_inversos[-2]) < tol:
            break
    
    autovalor_mas_pequeno = autovalores_inversos[-1]
    autovector_asociado = b_k
    return autovalor_mas_pequeno, autovector_asociado, autovalores_inversos


matriz = generarMatriz(3,3)
condicion = generaNumCond(matriz)
matrizPot = potenciaMatriz(matriz,2)
print(matriz)
print(condicion)
print(matrizPot)

plt.scatter(condicion, 0)
plt.show()

autovalores, autovectores = potenciaMatrizAutoV(matriz,2,6)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(autovalores)
plt.title('Convergencia del autovalor dominante')
plt.xlabel('Iteraciones')
plt.ylabel('Autovalor')

plt.subplot(2, 1, 2)
autovectores = np.array(autovectores)
for i in range(autovectores.shape[1]):
    plt.plot(autovectores[:,i], label=f'Componente {i+1}')
plt.title('Convergencia del autovector dominante')
plt.xlabel('Iteraciones')
plt.ylabel('Autovector')
plt.legend()

plt.tight_layout()
plt.show()

autovalor_dominante, autovector_dominante, autovalores = metodo_potencia_simetrico(matriz)

print("Autovalor Dominante:", autovalor_dominante)
print("Autovector Dominante:", autovector_dominante)

plt.plot(autovalores, marker='o', color='b')
plt.show()

autovalor_mas_pequeno, autovector_asociado, autovalores_inversos = metodo_potencia_inverso(matriz)

print("Autovalor Más Pequeño:", autovalor_mas_pequeno)
print("Autovector Asociado:", autovector_asociado)

plt.figure(figsize=(8, 4))
plt.plot(autovalores_inversos, marker='.', linestyle='-', color='g', markersize=5)
plt.grid(visible=False)
plt.tight_layout()
plt.show()