import numpy as np
import matplotlib.pyplot as plt
from openfermion.ops import FermionOperator
from openfermion.linalg import get_sparse_operator
import scipy.linalg as la

# --- FUNCIÓN DEL HAMILTONIANO ---
def build_kitaev_hamiltonian(N, mu, t, delta):
    hamiltonian = FermionOperator()
    for i in range(N):
        hamiltonian += FermionOperator(f'{i}^ {i}', -mu)
        if i < N - 1:
            hamiltonian += FermionOperator(f'{i}^ {i+1}', -t)
            hamiltonian += FermionOperator(f'{i+1}^ {i}', -t)
            hamiltonian += FermionOperator(f'{i} {i+1}', delta)
            hamiltonian += FermionOperator(f'{i+1}^ {i}^', np.conj(delta))
    return hamiltonian

N_sites = 6 #Como mucho mueve este hasta 8
t = -1.0
delta = -1.0

mu_values = np.linspace(0.0, 4.0, 50)

print("\n" + "="*60)
print(" CÁLCULO DEL ESPECTRO DE ENERGÍA (DIAGRAMA DE FASES)")
print("="*60)

# Listas para guardar los primeros 4 niveles de energía. Se pueden plotear más pero cambia el N_sites o va a tirar error.
energies_0 = []
energies_1 = []
energies_2 = []
energies_3 = []

for mu in mu_values:
    H_fermion = build_kitaev_hamiltonian(N_sites, mu, t, delta)
    H_matrix = get_sparse_operator(H_fermion, n_qubits=N_sites).toarray()
    
    
    eigenvalues = la.eigvalsh(H_matrix) 
    
    # Guardamos las 4 energías más bajas
    energies_0.append(eigenvalues[0])
    energies_1.append(eigenvalues[1])
    energies_2.append(eigenvalues[2])
    energies_3.append(eigenvalues[3])


plt.figure(figsize=(10, 6))

plt.plot(mu_values, energies_0, label='$E_0$ (Estado Fundamental)', color='blue', linewidth=2)
plt.plot(mu_values, energies_1, label='$E_1$ (1er Estado Excitado)', color='orange')
plt.plot(mu_values, energies_2, label='$E_2$', color='green')
plt.plot(mu_values, energies_3, label='$E_3$', color='red')

critical_mu = 2.0 * abs(t)
plt.axvline(x=critical_mu, color='black', linestyle='--', label=f'Punto Crítico $\mu = {critical_mu}$')

plt.xlabel('Potencial Químico ($\mu$)', fontsize=12)
plt.ylabel('Energía', fontsize=12)
plt.title(f'Espectro de Energía de la Cadena de Kitaev ($N={N_sites}$ sitios)', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()