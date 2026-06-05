from qiskit.circuit import QuantumCircuit, ParameterVector
import numpy as np

'''Hamiltoniano de la cadena'''
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


N = 3
mu_test = 0.5
t= -1
delta = 1
H_matrix = Hamiltonian(N,mu_test,t,delta)

#Revisa si puedes cambiar este al otro que tienes en numpy
def build_qiskit_ansatz(N=3, layers=2):
    """Builds the quantum circuit for the hardware-efficient Ansatz."""
    # Create a circuit with N qubits
    qc = QuantumCircuit(N)
    
    # We need 2 angles (Ry, Rz) per qubit per layer, plus the initial layer
    num_params = N * 2 * (layers + 1)
    thetas = ParameterVector('θ', num_params)
    
    param_idx = 0
    
    # Initial Rotations
    for q in range(N):
        qc.ry(thetas[param_idx], q)
        qc.rz(thetas[param_idx+1], q)
        param_idx += 2
        
    # Repeating Entanglement and Rotation Layers
    for layer in range(layers):
        # Entanglement (CNOTs between 0-1 and 1-2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        
        # More Rotations
        for q in range(N):
            qc.ry(thetas[param_idx], q)
            qc.rz(thetas[param_idx+1], q)
            param_idx += 2
            
    return qc, thetas


from qiskit.quantum_info import SparsePauliOp

# SparsePauli toma la matriz de numpy y la traduce al lenguaje del Hardware del QC
qubit_hamiltonian = SparsePauliOp.from_operator(H_matrix)

print(qubit_hamiltonian)

from qiskit.primitives import StatevectorEstimator as Estimator
from scipy.optimize import minimize

# Initialize the circuit and the measurement engine
ansatz_circuit, theta_params = build_qiskit_ansatz(N=3, layers=2)
estimator = Estimator()

def qiskit_cost_function(angle_values):
    """
    This function sends the job to the quantum simulator using the new V2 API. I hate you IBM for changing V1 and costing me a week
    """
    # 1. Package everything into a single "PUB" tuple
    pub = (ansatz_circuit, qubit_hamiltonian, angle_values)
    
    # 2. Pass a list of PUBs to the estimator
    job = estimator.run([pub])
    
    # 3. Retrieve the result
    result = job.result()
    
    # 4. In V2, we access the Expectation Value (evs) from the data of the first pub
    energy = result[0].data.evs
    
    # The new API sometimes returns a 0D array, so we use float() to ensure 
    # the scipy optimizer just gets a plain Python number.
    return float(energy)

# 18 parameters for N=3, layers=2
initial_angles = np.random.uniform(0, 2*np.pi, 18)

print("Sending jobs to Qiskit...")
result = minimize(qiskit_cost_function, 
                  initial_angles, 
                  method='COBYLA', # COBYLA is preferred for noisy hardware
                  options={'maxiter': 30})

print("VQE Energy Found:", result.fun)

from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as RuntimeEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# 1. Log in and find a computer
service = QiskitRuntimeService(channel="ibm_quantum_platform", token="kn_Ho35AK2S49tmL0qMV3JLDnke70UMgCv3Uhb8pLtTT")
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=3)
print(f"Connected to {backend.name}")

# 2. V2 REQUIREMENT: Transpile the theoretical circuit to physical hardware (ISA)
print("Compiling circuit to physical hardware layout...")
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(ansatz_circuit)

# 3. V2 REQUIREMENT: Update the Hamiltonian to match the new physical wiring
isa_hamiltonian = qubit_hamiltonian.apply_layout(isa_circuit.layout)

# 4. Initialize the hardware Estimator using 'mode'
hardware_estimator = RuntimeEstimator(mode=backend)

def hardware_cost_function(angle_values):
    """
    Sends the compiled ISA circuit and ISA Hamiltonian to the real chip.
    """
    # Use the V2 PUB format with our compiled ISA objects
    pub = (isa_circuit, isa_hamiltonian, angle_values)
    job = hardware_estimator.run([pub])
    
    result = job.result()
    # Extract the expectation value
    energy = result[0].data.evs
    
    return float(energy)

# 5. Run the VQE on the real Quantum Computer!
initial_angles = np.random.uniform(0, 2*np.pi, 18)

print("Starting hardware optimization. This may take hours depending on the queue...")
result = minimize(hardware_cost_function, 
                  initial_angles, 
                  method='COBYLA', 
                  options={'maxiter': 25}) # Keep this low for sanity!

print("========================================")
print(f"Real Hardware VQE Energy Found: {result.fun:.6f}")
print("========================================")