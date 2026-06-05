'''For this part we are working with the Cerezo de la Roca Paper And the Mclean Paper ... I want to kill myself'''
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time 
# Hamiltoniano de la cadena
Id = np.eye(2,dtype= 'float') 
sx = np.array([[0,1],[1,0]],dtype=complex)
sy = np.array([[0,-1j],[1j,0]],dtype=complex)
sz = np.array([[1,0],[0,-1]],dtype=complex)

#Then the creation and ahnilation ops:
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

#And now the Hamiltonian
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

    
    return H


state_0 = np.array([1,0])
state_1 = np.array([0,1])

def Ry(theta):
    '''Una matriz de rotación 2x2 en Y para un angulo theta'''
    ry = np.array([[np.cos(theta/2), - np.sin(theta/2)],
                   [np.sin(theta/2), np.cos(theta/2)]])
    return ry

def Rz(theta):
    ''' Una rotación en Z (añade la fase compleja)'''
    rz = np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ])
    return rz

def create_initial_state(N):
    '''Crea el estado |000...0> para N qubits'''
    state = state_0
    for _ in range(N-1):
        state = np.kron(state,state_0)
    return state

def sigle_qubit_gate(gate,target_q,N):

    Id = np.eye(2)

    # Empezamos con la gate (si apuntamos a la primera) o la identidad 2x2

    if target_q == 1:
        operator = gate
    else:
        operator = Id

    for j in range(2,N + 1):
        if j == target_q:
            operator = np.kron(operator, gate)
        else:
            operator = np.kron(operator, Id)
    return operator

'''Ahora definimos las puertas'''

CNOT = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,0,1],
            [0,0,1,0]])

#Entrelazadas

CNOT_12 = np.kron(CNOT,np.eye(2))
CNOT_23 = np.kron(np.eye(2),CNOT)

def Ansatz(thetas,N=3,layers=2):

    psi = create_initial_state(N)
    ind_param = 0

    #rotaciones iniciales en los tres qubits:
    for q in range(1, N+1):
        psi = sigle_qubit_gate(Ry(thetas[ind_param]),q, N) @ psi
        psi = sigle_qubit_gate(Rz(thetas[ind_param + 1]),q, N) @ psi
        ind_param += 2

    for layer in range(layers):
        #entrelazamiento
        psi = CNOT_12 @ psi
        psi = CNOT_23 @ psi
    
        for q in range(1, N+1):
            #más rotaciones
            psi = sigle_qubit_gate(Ry(thetas[ind_param]),q, N) @ psi
            psi = sigle_qubit_gate(Rz(thetas[ind_param + 1]),q, N) @ psi
            ind_param += 2
    
    return psi

#Función de costo

def cost_function(thetas, H):
    psi_prueba = Ansatz(thetas,N=3,layers=2)
    energia = np.real(np.conjugate(psi_prueba).T @ H @ psi_prueba)
    return energia

N = 3
mu_test = 0.5
t= -1
delta = 1
H_matrix = Hamiltonian(N,mu_test,t,delta)
exact_energy, _ = np.linalg.eigh(H_matrix)
#Necesitamos 18 parametros para N = 3 y layers = 2
params = 18 
best_energy = float('inf')
for attempt in range(5):
    print('Intento:',attempt+1)
    initial_thetas = np.random.uniform(0, 2*np.pi, params)
    
    
    result = minimize(cost_function, 
                      initial_thetas, 
                      args=(H_matrix,), 
                      method='COBYLA') 
    
    if result.fun < best_energy:
        best_energy = result.fun

print(abs(best_energy-exact_energy[0]))
'''Ahora utilizamos los ángulos óptimos y realizamos mediciones para el número de particulas (empecemos solo por el ground state)'''

def Number_op(N):
    dim = 2**N
    N_op = np.zeros((dim,dim),dtype=complex)
    # El operador tiene la forma de la sumatoria de c_dag_j * c_j
    for j in range(1,N+1):
        N_op += c_plus(N,j)@c_minus(N,j)
    return N_op

def build_correlation_operator(N):
    #Majorana izq
    gamma_left = -1j*(c_minus(N,1) - c_plus(N,1)) #notar que usamos 1 puesto que es el primer sitio de la cadena

    #Majorana der
    gamma_right = c_minus(N,N) + c_plus(N,N) # N pues estamos en el último sitio de la cadena
    
    correlation = -1j*(gamma_left @ gamma_right)

    return correlation

N = 3
t= -1
delta = 1

mu_values = np.linspace(0,3,20)
P_numbers = []
P_numbers2 = []
params = 18 
operator = Number_op(N)
corr_op = build_correlation_operator(N)

for mu in mu_values:

    best_energy = float('inf')
    H_matrix = Hamiltonian(N,mu,t,delta,corr_op)
    _, eigenfunc = np.linalg.eigh(H_matrix)
    optimal_thetas = None
    
    for attempt in range(3):
        initial_thetas = np.random.uniform(0, 2*np.pi, params)
        
        
        result = minimize(cost_function, 
                        initial_thetas, 
                        args=(H_matrix,), 
                        method='COBYLA') 
        
        if result.fun < best_energy:
            best_energy = result.fun
            optimal_thetas = result.x
    
    state = Ansatz(optimal_thetas, N=N, layers = 2)
    particle_number = np.real(np.conjugate(state).T @ operator @ state)
    P_numbers.append(particle_number)

    particle_number2 = np.real(np.conjugate(eigenfunc[0]).T @operator @ eigenfunc[0])
    P_numbers2.append(particle_number2)


plt.scatter(mu_values,P_numbers,s=4,color='blue', label='VQE')
plt.scatter(mu_values,P_numbers2, s=4, color='red',label='EDG' ) #Exact diagonalization
plt.axvline(x=2.0, color='black', linestyle='--', label=r'Punto crítico $|\mu| = 2|t|$')
#plt.title('Particle number')
plt.xlabel('Chemical Potential $\mu$', fontsize=12)
plt.ylabel('Particle number  $\langle \hat{N} \u27E9$ ', fontsize=12)
plt.legend()
plt.savefig('Número de particulas(VQE).pdf', format='pdf')
plt.show()
    

