import numpy as np
import matplotlib.pyplot as plt

# Generación de una matriz aleatoria de tamaño `a` x `b`
def generarMatriz(a, b):
    m = np.random.randint(1, 100, (a, b)) 
    return m

# Cálculo del número de condición de una matriz `a`
def generaNumCond(a):
    cond = np.linalg.cond(a, 1)  # Calcula el número de condición utilizando la norma 1
    return cond

# Gráfica del número de condición en función del tamaño de la matriz
def grafica_numero_condicion():
    tamanos = range(2, 11)  # Tamaños de 2x2 a 10x10
    num_condiciones = []

    # Generación y cálculo del número de condición para cada tamaño de matriz
    for n in tamanos:
        matriz = generarMatriz(n, n)  # Genera una matriz de tamaño n x n
        num_cond = generaNumCond(matriz)
        num_condiciones.append(num_cond)

    # Gráfico
    plt.figure()
    plt.plot(tamanos, num_condiciones, marker='o')
    plt.title("Número de Condición en Función del Tamaño de la Matriz")
    plt.xlabel("Tamaño de la Matriz (nxn)")
    plt.ylabel("Número de Condición")
    plt.grid()
    plt.show()

# Método de Potencia para calcular el autovalor dominante y autovector asociado de `a`
def metodo_potencia(a, max_iter=100, tol=1e-10):
    n = a.shape[0]
    b_k = np.random.rand(n)
    b_k = b_k / np.linalg.norm(b_k)
    
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

# Método de Potencia Inverso para obtener el autovalor más pequeño de `a`
def metodo_potencia_inverso(a, max_iter=100, tol=1e-10):
    n = a.shape[0]
    b_k = np.random.rand(n)
    b_k = b_k / np.linalg.norm(b_k)
    
    autovalores_inverso = []
    for _ in range(max_iter):
        b_k1 = np.linalg.solve(a, b_k)  # Resolver sistema lineal para el producto inverso
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm

        autovalores_inverso.append(b_k1_norm)
        if len(autovalores_inverso) > 1 and abs(autovalores_inverso[-1] - autovalores_inverso[-2]) < tol:
            break

    autovalor_minimo = autovalores_inverso[-1]
    autovector_minimo = b_k
    return autovalor_minimo, autovector_minimo, autovalores_inverso

# Gráfico de convergencia de autovalores en Método de Potencia
def graficar_convergencia_potencia(autovalores, titulo):
    plt.figure()
    plt.plot(autovalores, marker='o')
    plt.title(f'Convergencia de Autovalores - {titulo}')
    plt.xlabel('Iteraciones')
    plt.ylabel('Valor de Autovalor')
    plt.grid()
    plt.show()

# Simulación y análisis de una matriz de covarianza para un portafolio de inversiones
def analisis_portafolio():
    # Generar una matriz de covarianza simétrica positiva definida de ejemplo
    np.random.seed(0)
    cov_matrix = np.random.rand(5, 5)
    cov_matrix = np.dot(cov_matrix, cov_matrix.T)  # Aseguramos que sea simétrica positiva definida
    print(f'Matriz: ', cov_matrix)

    # Calcular el número de condición del portafolio
    condicion = generaNumCond(cov_matrix)
    print(f'Número de condición del portafolio: {condicion}')

    # Método de Potencia para autovalor dominante
    autovalor_dominante, autovector_dominante, autovalores = metodo_potencia(cov_matrix)
    print(f'Autovalor dominante (riesgo principal): {autovalor_dominante}')
    
    # Método de Potencia Inverso para autovalor mínimo
    autovalor_minimo, autovector_minimo, autovalores_inverso = metodo_potencia_inverso(cov_matrix)
    print(f'Autovalor mínimo (diversificación): {autovalor_minimo}')

    # Gráficas de convergencia
    graficar_convergencia_potencia(autovalores, "Método de Potencia - Portafolio")
    graficar_convergencia_potencia(autovalores_inverso, "Método de Potencia Inverso - Portafolio")

# Simulación y análisis de una matriz de rigidez para el diseño de un puente
def analisis_estructura_puente():
    # Generar una matriz de rigidez simétrica positiva definida
    np.random.seed(1)
    rigidity_matrix = np.random.rand(5, 5)
    rigidity_matrix = np.dot(rigidity_matrix, rigidity_matrix.T)
    print(f'Matriz: ', rigidity_matrix)

    # Calcular el número de condición del diseño de la estructura
    condicion = generaNumCond(rigidity_matrix)
    print(f'Número de condición del diseño del puente: {condicion}')

    # Método de Potencia para autovalor dominante
    autovalor_dominante, autovector_dominante, autovalores = metodo_potencia(rigidity_matrix)
    print(f'Autovalor dominante (rigidez máxima): {autovalor_dominante}')
    
    # Método de Potencia Inverso para autovalor mínimo
    autovalor_minimo, autovector_minimo, autovalores_inverso = metodo_potencia_inverso(rigidity_matrix)
    print(f'Autovalor mínimo (vulnerabilidad): {autovalor_minimo}')

    # Gráficas de convergencia
    graficar_convergencia_potencia(autovalores, "Método de Potencia - Estructura del Puente")
    graficar_convergencia_potencia(autovalores_inverso, "Método de Potencia Inverso - Estructura del Puente")

# Llamadas a los análisis de simulación
grafica_numero_condicion()  # Gráfico de número de condición en función del tamaño de la matriz
analisis_portafolio()        # Análisis de portafolio de inversiones
analisis_estructura_puente() # Análisis estructural del puente
