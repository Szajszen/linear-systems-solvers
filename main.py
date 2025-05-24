import time
import math
import matplotlib.pyplot as plt

def create_system(N, a1, a2, a3, f):
    # Tworzenie macierzy A
    A = [[0]*N for _ in range(N)]
    for i in range(N):
        A[i][i] = a1
        if i < N-1:
            A[i][i+1] = a2
            A[i+1][i] = a2
        if i < N-2:
            A[i][i+2] = a3
            A[i+2][i] = a3

    # Tworzenie wektora b
    b = [math.sin(n * (f + 1)) for n in range(1, N+1)]

    return A, b



def Jacobi(A,b,N):
    # Inicjalizacja wektora x
    x = [0]*N
    max_res = 0

    # Algorytm Jacobiego
    start_time = time.time()
    res_norms_jacobi = []
    for iteration in range(100):  # maksymalna liczba iteracji
        x_new = x.copy()
        for i in range(N):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, N))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        # Obliczanie residuum
        res = [sum(A[i][j] * x_new[j] for j in range(N)) - b[i] for i in range(N)]
        norm_res = math.sqrt(sum(i**2 for i in res))
        max_res = max(max_res, norm_res)
        res_norms_jacobi.append(norm_res)

        x = x_new
        # Sprawdzenie warunku zatrzymania
        if norm_res < 10**-9:
            break

    jacobi_time = time.time() - start_time
    print("Metoda Jacobiego:")
    print("Czas trwania: ", jacobi_time)
    print("Liczba iteracji: ", iteration+1)
    if iteration == 99:  # jeśli algorytm nie zbiegł
        print("Maksymalne residuum: ", max_res)
    else:
        print("Norma residuum: ", norm_res)

    return res_norms_jacobi

def Gauss_seidla(A,b,N):
    # Algorytm Gaussa-Seidla

    x = [0]*N  # reset wektora x
    max_res = 0

    start_time = time.time()
    res_norms_gauss = []
    for iteration in range(100):  # maksymalna liczba iteracji
        x_new = x.copy()
        for i in range(N):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, N))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        # Obliczanie residuum
        res = [sum(A[i][j] * x_new[j] for j in range(N)) - b[i] for i in range(N)]
        norm_res = math.sqrt(sum(i**2 for i in res))
        res_norms_gauss.append(norm_res)
        max_res = max(max_res, norm_res)

        x = x_new
        # Sprawdzenie warunku zatrzymania
        if norm_res < 10**-9:
            break

    gauss_seidel_time = time.time() - start_time
    print("\nMetoda Gaussa-Seidla:")
    print("Czas trwania: ", gauss_seidel_time)
    print("Liczba iteracji: ", iteration+1)
    if iteration == 99:  # jeśli algorytm nie zbiegł
        print("Maksymalne residuum: ", max_res)
    else:
        print("Norma residuum: ", norm_res)

    return res_norms_gauss


N = 919
f = 1

# Dla a1 = 5 + 7, a2 = a3 = -1
print("\nDla a1 = 5 + 7, a2 = a3 = -1:")
A, b = create_system(N, 5 + 7, -1, -1, f)
res_norms_jacobi = Jacobi(A,b,N)
res_norms_gauss = Gauss_seidla(A, b, N)
# Rysowanie wykresów
plt.figure()
plt.plot(res_norms_jacobi, label='Metoda Jacobiego')
plt.plot(res_norms_gauss, label='Metoda Gaussa-Seidla')
plt.xlabel('Iteracja')
plt.ylabel('Norma residuum')
plt.legend()
plt.savefig("Wykres_B.png")
plt.show()


#ZADANIE C
# Dla a1 = 3, a2 = a3 = -1
print("\nDla a1 = 3, a2 = a3 = -1:")
A, b = create_system(N, 3, -1, -1, f)
res_norms_jacobi = Jacobi(A,b,N)
res_norms_gauss = Gauss_seidla(A, b, N)

# Rysowanie wykresów
plt.figure()
plt.plot(res_norms_jacobi, label='Metoda Jacobiego')
plt.plot(res_norms_gauss, label='Metoda Gaussa-Seidla')
plt.xlabel('Iteracja')
plt.ylabel('Norma residuum')
plt.legend()
plt.savefig("Wykres_C.png")
plt.show()

#ZADANIE D
def lu_factorization(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0

    start_time = time.time()

    for j in range(n):
        for i in range(j+1):
            s1 = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - s1

        for i in range(j+1, n):
            s2 = sum(L[i][k] * U[k][j] for k in range(j))
            L[i][j] = (A[i][j] - s2) / U[j][j]

    factorization_time = time.time() - start_time

    print("\nMetoda faktoryzacji LU:")
    print("Czas faktoryzacji LU: ", factorization_time, "\n")

    return L, U

def solve_lu_factorization(A, b):
    L, U = lu_factorization(A)
    n = len(A)
    y = [0.0] * n
    x = [0.0] * n

    # Rozwiązanie Ly = b
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

    # Rozwiązanie Ux = y
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]

    return x

def calculate_residual_norm(A, b, x):
    n = len(A)
    res = [0.0] * n

    for i in range(n):
        res[i] = b[i] - sum(A[i][j] * x[j] for j in range(n))

    norm_res = math.sqrt(sum(r**2 for r in res))

    return norm_res


# Rozwiązanie układu równań metodą faktoryzacji LU
x = solve_lu_factorization(A, b)

# Obliczenie normy residuum
norm_res = calculate_residual_norm(A, b, x)

print("Norma residuum: ", norm_res, "\n")

#ZADANIE E
def measure_time(N_values):
    jacobi_times = []
    gauss_seidel_times = []
    lu_factorization_times = []

    for N in N_values:
        # Dla a1 = 5 + 7, a2 = a3 = -1
        A, b = create_system(N, 5 + 7, -1, -1, f)

        # Pomiar czasu dla algorytmu Jacobiego
        start_time = time.time()
        Jacobi(A,b,N)
        jacobi_times.append(time.time() - start_time)

        # Pomiar czasu dla algorytmu Gaussa-Seidla
        start_time = time.time()
        Gauss_seidla(A,b,N)
        gauss_seidel_times.append(time.time() - start_time)

        # Pomiar czasu dla faktoryzacji LU
        start_time = time.time()
        solve_lu_factorization(A, b)
        lu_factorization_times.append(time.time() - start_time)

    return jacobi_times, gauss_seidel_times, lu_factorization_times

def plot_times(N_values, jacobi_times, gauss_seidel_times, lu_factorization_times):
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, jacobi_times, label='Metoda Jacobiego')
    plt.plot(N_values, gauss_seidel_times, label='Metoda Gaussa-Seidla')
    plt.plot(N_values, lu_factorization_times, label='Faktoryzacja LU')
    plt.xlabel('Liczba niewiadomych N')
    plt.ylabel('Czas wykonania (s)')
    plt.legend()
    plt.grid(True)
    plt.savefig("Wykres_E.png")
    plt.show()

# Wywołanie funkcji measure_time i plot_times

A, b = create_system(N, 5 + 7, -1, -1, f)
N_values = [100, 500, 1000, 2000, 3000]  # dodaj więcej wartości N, jeśli potrzebujesz
jacobi_times, gauss_seidel_times, lu_factorization_times = measure_time(N_values)
plot_times(N_values, jacobi_times, gauss_seidel_times, lu_factorization_times)