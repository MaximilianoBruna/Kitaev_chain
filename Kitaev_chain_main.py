import numpy as np
import matplotlib.pyplot as plt

# --- Construir el Hamiltoniano ---

def buildAndSolve_BdG_hamiltonian(N,mu,t,delta):
    ''' Construye y Diagonaliza la hamiltoniano BdG 2n x 2n'''
    
    H_0 = np.zeros((N,N))
    Delta_M = np.zeros((N,N))
    for i in range(N):
        H_0[i,i] = mu
        if i < N-1: 
            H_0[i, i+1] = -t
            H_0[i+1, i] = -t
            Delta_M[i, i+1] = delta
            Delta_M[i+1, i] = -delta
    
    # H_BdG = [H_0 , Delta_M]
    #             [-Delta_M* , -H_0*]
    top = np.hstack((H_0,Delta_M))
    bottom = np.hstack((-np.conjugate(Delta_M), -np.conjugate(H_0)))
    H_BdG = np.vstack((top,bottom))

    #Diagonalizamos la matriz
    eigenval , eigenfunc = np.linalg.eigh(H_BdG)

    return eigenval, eigenfunc


N = 3
mu = 0.5
t = - 1
delta = 1
eigenval, _ = buildAndSolve_BdG_hamiltonian(N,mu,t,delta)
print(eigenval[0])


''' Ahora podemos recorrer para distintos mu's y obtener el espectro de las energías
    para valores constantes de la ccantidad de sitios N, la amplitud de tunelamiento t y el gap de energía delta.
    Para esto hay que hacer un for y recorrer los valores de mu.'''

N = 3
t = -1
delta = 1
mu_values = np.linspace(0,3,100)
Eigenvalues = []
for mu in mu_values:
    BdG, _ = buildAndSolve_BdG_hamiltonian(N,mu,t,delta)
    Eigenvalues.append(BdG)

Eigenvalues= np.array(Eigenvalues)
for i in range(2*N):
    plt.plot(mu_values , Eigenvalues[:,i], color= 'green')
plt.axvline(x=2.0, color='red', linestyle='--', label=r'Punto Crítico $|\mu| = 2t$')
#plt.title('Espectro de Energía (Matriz BdG)', fontsize=14)
plt.xlabel('Potencial Químico $\mu$', fontsize=12)
plt.ylabel('Energía $E$', fontsize=12)
plt.legend()
plt.savefig('Espectro de Energía (Matriz BdG).pdf', format='pdf')
plt.show()


''' Con este hamiltoniano BdG hemos las excitacciones individuales por particula (checa esto, esta explicado cómo las weas). 
    Ahora obtendremos el espectro para todos los posibles estados simultaneamente.
    Para esto pasamos del formalismo de la matriz BdG de dimensiones 2n x 2n a trabajar en el espacio de Hilbert
    en 2^n x 2^n'''

''' Para conseguir esto necesitamos utilizar los operadores de subida y bajada fermiónicos, 
    cómo mi computador no tiene idea de que es eso habrá que definir primero los operadores de subida
    y bajada con matrices de pauli y posteriormente utilizar una transformación de Jordan-Wigner para traducir los operadores
    fermiónicos a matrices que podamos multiplicar con numpy'''

Id = np.eye(2,dtype= 'float') 
sx = np.array([[0,1],[1,0]],dtype=complex)
sy = np.array([[0,-1j],[1j,0]],dtype=complex)
sz = np.array([[1,0],[0,-1]],dtype=complex)
s_minus = 0.5*(sx - 1j*sy)
s_plus = 0.5*(sx + 1j*sy)
#Whit this done, we move to the Jordan-Wigner transform

def sum_z(N):
    if N == 0: 
        return 1
    sz_1 =sz
    sz_2 = sz   
    for k in range(N-1):
        sz_1 = np.kron(sz_1,sz_2)
    return sz_1

def sum_id(N): 
    if N == 0: 
        return 1
    prod = Id
    for k in range(N-1):
        prod = np.kron(prod,Id)
    return prod

def c_plus(N,J):
    if J> N or J<0: 
        return 'Invalid site'
    nz = sum_z(J-1)
    nI = sum_id(N-J)
    # Jordan-Wigner: (sigma_z string) x (sigma_+) x (Identity string)
    c_dagger = np.kron(nz, np.kron(s_plus,nI))
    
    return c_dagger

def c_minus(N, J):
    if J> N or J<0: 
        return 'Invalid site'
    nz = sum_z(J-1)
    nI = sum_id(N-J)
    
    # Jordan-Wigner: (sigma_z string) x (sigma_-) x (Identity string)
    c = np.kron(nz, np.kron(s_minus, nI))
    return c

'''Test for the c operators
print(sum_z(1))
print(c_plus(3,2))
print(c_minus(3,1))
'''

def Hamiltonian(N,mu,t,delta,corr_op=None):
    dim = 2**N
    H = np.zeros((dim,dim),dtype=complex)
    I_dim = np.eye(dim) 

    #Término del potencial químico: sumatoria( mu * c_j^dagger * c_j)
    for j in range(1,N+1):
        c_dagger= c_plus(N,j)
        c = c_minus(N,j)
        operator = c_dagger@c
        H += mu * (operator - 0.5 * I_dim) #Añadir un electrón aumenta la energía en +mu, para mantener el espectro centrado
                                           #Hacemos la resta al operador número. 

    #Termino de la amplitud t y el gap delta:
    for j in range(1,N):
        c_dag_j = c_plus(N, j)
        c_j = c_minus(N, j) #we don't need this one lol.
        
        c_dag_next = c_plus(N, j + 1)
        c_next = c_minus(N, j + 1)
        
        # Hopping: -t * (c_j^dagger * c_{j+1} + h.c.)
        hop = -t * (c_dag_j @ c_next)
        hop_hc = np.conjugate(hop.T) # Hermitian conjugate
        
        # Pairing: -delta * (c_dag_j * c_dag_{j+1} + h.c.)
        pair = -delta * (c_dag_j @ c_dag_next)
        pair_hc = np.conjugate(pair.T)
        
        H += hop + hop_hc + pair + pair_hc 
    if corr_op is not None:
        H += 1e-8 * corr_op #una perturbación pequeña, es únicamente un arreglo para poder plotear correctamente la correlación de los 
                            #Majorana.

    Eigenval, Eigenfunctions = np.linalg.eigh(H)
    return Eigenval, Eigenfunctions

''' Ahora graficamos de la misma forma para el caso de la matriz BdG'''

N = 3
mu_values = np.linspace(0,3,200)
t = -1
delta = 1
Eigenvalues = []

for mu in mu_values:
    eigenval, _ = Hamiltonian(N,mu,t,delta)
    Eigenvalues.append(eigenval)

Eigenvalues= np.array(Eigenvalues)

for i in range(2**N):
    plt.plot(mu_values , Eigenvalues[:,i], color= 'black')

plt.axvline(x=2.0, color='red', linestyle='--', label=r'Punto Crítico $|\mu| = 2t$')
#plt.title('Espectro de Energía de muchos cuerpos', fontsize=14)
plt.xlabel('Potencial Químico $\mu$', fontsize=12)
plt.ylabel('Energía $E$', fontsize=12)
plt.legend()
plt.savefig('Espectro de Energía muchos cuerpos(cadena completa).pdf', format='pdf')
plt.show()

'''Esta parte ya muestra correctamente el mismo tipo de comportamiento que el observado en la matriz BdG,
   La gracia ahora recae en que los niveles de energía están pareados gracias a la paridad de Fermiones. 
   Para observar esto en el gráfico definimos la función Paridad: '''
 
def build_parity_operator(N):
    ''' El operador paridad está definido cómo la 
        suma de los productos de 1-2*(c_dag_j*c_j)'''
    dim = 2**N
    P = np.eye(dim, dtype=complex)
    I_dim = np.eye(dim, dtype=complex)

    for j in range(1, N+1):
        n_j = c_plus(N, j) @ c_minus(N, j)
        P = P @ (I_dim - 2 * n_j)
        
    return P

N = 3
mu_values = np.linspace(0, 3, 200) 
t = -1
delta = 1
#Generamos el operador paridad una sola vez
P_operator = build_parity_operator(N)
Eigenvalues = []
Parities = []
#Recorremos para los valores de mu.
for mu in mu_values:
    Eigenval, Eigenfunctions = Hamiltonian(N, mu, t, delta) #Para la paridad se vuelve necesario guardar el autovector 
    Eigenvalues.append(Eigenval)                            #pues su valor de expectación nos indicará la paridad del estado.
    
    # Calculamos la paridad de los estados.
    current_parities = []
    for i in range(2**N):
        psi = Eigenfunctions[:, i]
        # Expectation value: <psi | P | psi>
        # Usamos np.real porque matemáticamente el resultado debe ser real.
        parity_val = np.real(np.conjugate(psi).T @ P_operator @ psi)
        current_parities.append(parity_val)
        
    Parities.append(current_parities)

Eigenvalues = np.array(Eigenvalues)
Parities = np.array(Parities)

#   --- Graficamos ---
plt.figure(figsize=(8, 6))

# plt.scatter en vez de plt.plot, al cruzarse los niveles de energía
# np.linalg.eigh los ordena por tamaño, lo que causa que lineas de paridades distintas
# "intercambien" posiciones.
for i in range(2**N):
    # Máscaras para separar las paridades positivas de las negativas
    mask_even = Parities[:, i] > 0
    mask_odd = Parities[:, i] < 0
    
   
    plt.scatter(mu_values[mask_even], Eigenvalues[mask_even, i], color='red', s=4) #par es rojo
    plt.scatter(mu_values[mask_odd], Eigenvalues[mask_odd, i], color='blue', s=4)  #impar es azul

# Líneas falsas para que la leyenda se vea bien.
plt.plot([], [], color='red', label='Paridad par (+1)')
plt.plot([], [], color='blue', label='Paridad impar (-1)')

plt.axvline(x=2.0, color='black', linestyle='--', label=r'Punto crítico $|\mu| = 2|t|$')
#plt.title('Espectro de energía muchos cuerpos (Paridad)', fontsize=14)
plt.xlabel('Potencial químico $\mu$', fontsize=12)
plt.ylabel('Energía $E$', fontsize=12)
plt.legend()
plt.savefig('Espectro de energía muchos cuerpos (Paridad).pdf', format='pdf')
plt.show()

''' Ahora nos enfocamos en el valor de expectación del número de particulas N,
    puesto que el Hamiltoniano no conserva el número de particulas 
    obtenemos resultados extraños, en un aislante normal el estado base tiene un número entero de particulas fijo. Pero en 
    nuestra fase topologica superconductora este no es el caso debido a que nuestro estado base es una superposición de 
    estados vacíos y pares de Cooper.'''

def Number_op(N):
    dim = 2**N
    N_op = np.zeros((dim,dim),dtype=complex)
    # El operador tiene la forma de la sumatoria de c_dag_j * c_j
    for j in range(1,N+1):
        N_op += c_plus(N,j)@c_minus(N,j)
    return N_op

''' Querremos buscar también el valor de expectación del número de correlación de Majorana, pues este
    nos asegurara que el estado cero de energía se encuentra deslocalizado en los dos extremos de la cadena'''


def build_correlation_operator(N):
    #Majorana izq
    gamma_left = -1j*(c_minus(N,1) - c_plus(N,1)) #notar que usamos 1 puesto que es el primer sitio de la cadena

    #Majorana der
    gamma_right = c_minus(N,N) + c_plus(N,N) # N pues estamos en el último sitio de la cadena
    
    correlation = -1j*(gamma_left @ gamma_right)

    return correlation

'''Ahora corremos otro ciclo, puesto que ya calculamos los autovalores, autofunciones y las paridades solo nos
   enfocamos en estos operadores'''

N = 3
mu_values = np.linspace(0, 3, 200) 
t = -1
delta = 1
N_operador = Number_op(N)
corr_operador = build_correlation_operator(N)
'''Creamos diccionarios para guardar los distintos valores de expectación siguiendo el formato 
   {indice_autofunción: [valor_1,valor_2,...,valor_2**N]}'''
P_numbers = {}
edge_corrs = {}
for i in range(2**N):
    P_numbers[i] = []
    edge_corrs[i] = []

for mu in mu_values:
    Eigenval, Eigenfunctions = Hamiltonian(N, mu, t, delta, corr_operador)
    
    for i in range(2**N):
        state = Eigenfunctions[:,i] #corta una columna de autofunciones
        particle_number = np.real(np.conjugate(state).T @ N_operador @ state)
        edge_corr = np.real((np.conjugate(state).T @ corr_operador @ state))     
        P_numbers[i].append(particle_number)
        edge_corrs[i].append(edge_corr)

#Ploteamos para el número de partículas

for i in range(2**N):
    plt.scatter(mu_values,P_numbers[i],s=4)

plt.axvline(x=2.0, color='black', linestyle='--', label=r'Critical Point $|\mu| = 2|t|$')
#plt.title('Particle number')
plt.xlabel('Potencial químico  $\mu$', fontsize=12)
plt.ylabel('Número de particulas $\langle \hat{N} \u27E9$ ', fontsize=12)
plt.savefig('Número de particulas.pdf', format='pdf')
plt.show()

#Ahora para el majorana

for i in range(2**N):
    plt.scatter(mu_values , edge_corrs[i] , s=4)

plt.axvline(x=2.0, color='black', linestyle='--', label=r'Critical Point $|\mu| = 2|t|$')
#plt.title('Majorana correlation')
plt.xlabel('Potencial químico $\mu$', fontsize=12)
plt.ylabel(r'$\langle -i \gamma_1 \gamma_{2N} \rangle$', fontsize=12) #ARREGLA EL GAMMA_2N!!! TE DEBERÍA TIRAR LA MULTIPLICACIÓN
plt.savefig('Correlación de los majorana.pdf', format='pdf')
plt.show()


# ===== Paridad de los majorana ====

N_operador = Number_op(N)
corr_operador = build_correlation_operator(N)
P_op = build_parity_operator(N) 

# Mejor que lo que hice arriba con los diccionarios xd
mu_even, P_num_even, edge_corr_even = [], [], []
mu_odd, P_num_odd, edge_corr_odd = [], [], []

for mu in mu_values:
    Eigenval, Eigenfunctions = Hamiltonian(N, mu, t, delta)
    
    for i in range(2**N):
        state = Eigenfunctions[:, i] 
        particle_number = np.real(np.conjugate(state).T @ N_operador @ state)
        edge_corr = np.real(np.conjugate(state).T @ corr_operador @ state)
    
        parity_val = np.real(np.conjugate(state).T @ P_op @ state)
        
        #  Introducir la información correcta de si es par o impar.
        if parity_val > 0: # Even Parity (+1)
            mu_even.append(mu)
            P_num_even.append(particle_number)
            edge_corr_even.append(edge_corr)
        else:              # Odd Parity (-1)
            mu_odd.append(mu)
            P_num_odd.append(particle_number)
            edge_corr_odd.append(edge_corr)

plt.figure(figsize=(8, 6))
plt.scatter(mu_even, edge_corr_even, color='red', s=4, label='Paridad par (+1)')
plt.scatter(mu_odd, edge_corr_odd, color='blue', s=4, label='Paridad impar (-1)')
plt.axvline(x=2.0, color='black', linestyle='--', label=r'Punto crítico $|\mu| = 2|t|$')
#plt.title('Majorana Edge Correlation across the Many-Body Spectrum')
plt.xlabel('Potencial químico $\mu$', fontsize=12)
plt.ylabel(r'$\langle -i \gamma_1 \gamma_{2N} \rangle$', fontsize=12)
plt.legend()
plt.savefig('Majorana Edge Correlation across the Many-Body Spectrum(Paridad).pdf', format='pdf')
plt.show()
