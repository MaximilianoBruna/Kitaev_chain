import numpy as np
from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
import scipy.sparse.linalg as sla

def build_kitaev_hamiltonian(N, mu, t, delta):
    """
    Construye el Hamiltoniano de la Cadena de Kitaev 1D.
    N: Número de sitios
    mu: Potencial químico
    t: Amplitud de salto (hopping)
    delta: Parámetro de gap superconductor
    """
    hamiltonian = FermionOperator()
    
    for i in range(N):
        # Término de potencial químico: -mu * (c^\dagger_i c_i)
        hamiltonian += FermionOperator(f'{i}^ {i}', -mu)
        
        if i < N - 1:
            # Términos de salto (hopping): -t * (c^\dagger_i c_{i+1} + c^\dagger_{i+1} c_i)
            hamiltonian += FermionOperator(f'{i}^ {i+1}', -t)
            hamiltonian += FermionOperator(f'{i+1}^ {i}', -t)
            
            # Términos de emparejamiento (superconductividad): delta * c_i c_{i+1} + h.c.
            hamiltonian += FermionOperator(f'{i} {i+1}', delta)
            hamiltonian += FermionOperator(f'{i+1}^ {i}^', np.conj(delta))
    
    return hamiltonian

# Parámetros del sistema (Régimen Topológico: |mu| < 2t)
N_sites = 3     # Número de sitios (equvale a 3 qubits)
mu = 0.4        # Potencial químico
t = 2.0         # Hopping
delta = 1.0    # Gap superconductor

# Construir el operador fermiónico
H_fermion = build_kitaev_hamiltonian(N_sites, mu, t, delta)
print("--- Hamiltoniano Fermiónico ---")
print(H_fermion)

# Configurar NumPy para que la matriz se vea limpia
np.set_printoptions(precision=3, suppress=True, linewidth=120)

print("\n" + "="*60)
print(f" MATRIZ DEL HAMILTONIANO (8x8 para 3 sitios)")
print("="*60)
# Convertimos el operador a matriz dispersa y luego a densa para imprimir.
H_matrix = get_sparse_operator(H_fermion).toarray()
print(H_matrix.real)


print("\n" + "="*60)
print(" ENERGÍAS MÁS BAJAS (Diagonalización Exacta)")
print("="*60)
eigenvalues, eigenvectors = sla.eigsh(get_sparse_operator(H_fermion), k=4, which='SA')
for i, val in enumerate(eigenvalues):
    print(f" Estado {i}: \t {val:>8.4f}")


print("\n" + "="*60)
print(" HAMILTONIANO EN QUBITS (Mapeo Jordan-Wigner)")
print("="*60)
H_qubit = jordan_wigner(H_fermion)

# Iteramos sobre el diccionario interno de OpenFermion para darle formato
for term, coefficient in H_qubit.terms.items():
    # Si la tupla está vacía, representa el operador Identidad
    if not term:
        pauli_string = "I"
    else:
        # Unimos las letras (X, Y, Z) con sus índices
        pauli_string = " ".join([f"{pauli}{idx}" for idx, pauli in term])
    
    coef_str = f"{coefficient.real:>5.2f}"
    print(f" {coef_str} * [ {pauli_string:^10} ]")

print("############################################################\n")

