import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# --- MATRIZ DE BOGOLIUBOV-DE GENNES (BdG) ---
def build_bdg_matrix(N, mu, t, delta):
    """Construye la matriz BdG 2Nx2N para la cadena de Kitaev."""
    H0 = np.zeros((N, N))
    Delta_mat = np.zeros((N, N))
    
    for i in range(N):
        H0[i, i] = -mu
        if i < N - 1:
            H0[i, i+1] = -t
            H0[i+1, i] = -t
            Delta_mat[i, i+1] = delta
            Delta_mat[i+1, i] = -delta
            
    H_BdG = np.block([
        [H0, Delta_mat],
        [-np.conj(Delta_mat), -np.conj(H0)]
    ])
    return H_BdG

# =====================================================================
#             Decaimiento Exponencial de Modos de Majorana
# =====================================================================
N_values = list(range(3, 25))
mu = 0.5
t = 1.0
delta = 1.0
energy_splittings = []

for N in N_values:
    H_BdG = build_bdg_matrix(N, mu, t, delta)
    
    # Calculamos todos los valores propios de la matriz BdG
    eigenvalues = np.linalg.eigvalsh(H_BdG)
    
    # Por simetría partícula-hueco, el espectro es simétrico respecto a cero.
    # La separación de energía del estado fundamental de muchos cuerpos equivale
    # al valor propio positivo más pequeño del espectro BdG.
    min_energy = np.min(np.abs(eigenvalues))
    energy_splittings.append(min_energy)

# --- CREACIÓN DEL GRÁFICO ---
plt.figure(figsize=(8, 5))
plt.plot(N_values, energy_splittings, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
plt.yscale('log') 
plt.xlabel('Número de sitios (N)', fontsize=12)
plt.ylabel('Separación de Energía $\Delta E$', fontsize=12)
plt.title('Decaimiento Exponencial de Modos de Majorana', fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
#plt.savefig('majorana_decoherence.png', dpi=300)
plt.show()



# Parámetros fijos
N = 30  # Usamos una cadena más larga (30 sitios) para ver mejor los bordes
t = 1.0
delta = 1.0

# =====================================================================
#       ESPECTRO DE ENERGÍA Y TRANSICIÓN DE FASE TOPOLÓGICA
# =====================================================================
mu_values = np.linspace(0, 4.0, 100)
all_eigenvalues = []

for mu in mu_values:
    H = build_bdg_matrix(N, mu, t, delta)
    eigenvalues = np.linalg.eigvalsh(H)
    all_eigenvalues.append(eigenvalues)

all_eigenvalues = np.array(all_eigenvalues)

plt.figure(figsize=(12, 5))

# Gráfico 1: Espectro de Energía
plt.subplot(1, 2, 1)
for i in range(2 * N):
    # Graficamos cada nivel de energía en función de mu
    plt.plot(mu_values, all_eigenvalues[:, i], alpha=0.5, linewidth=0.8)

plt.axvline(x=2.0, color='red', linestyle='--', label=r'Punto Crítico $|\mu| = 2t$')
plt.title('Espectro de Energía (Matriz BdG)', fontsize=14)
plt.xlabel('Potencial Químico $\mu$', fontsize=12)
plt.ylabel('Energía $E$', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# =====================================================================
#           LOCALIZACIÓN ESPACIAL DE LOS MODOS DE MAJORANA
# =====================================================================

mu_topologico = 0.0
H_topo = build_bdg_matrix(N, mu_topologico, t, delta)
evals, evecs = np.linalg.eigh(H_topo)

# Los modos de Majorana corresponden a las energías más cercanas a cero.
# En una matriz de 2Nx2N, estos estados están en el medio del espectro (índices N-1 y N)
idx_zero_1 = N - 1
idx_zero_2 = N

# Extraemos los vectores propios (columnas de la matriz)
psi_1 = evecs[:, idx_zero_1]
psi_2 = evecs[:, idx_zero_2]

# La densidad de probabilidad en cada sitio i es la suma de la 
# componente de partícula y de hueco al cuadrado: |u_i|^2 + |v_i|^2
prob_density_1 = np.abs(psi_1[:N])**2 + np.abs(psi_1[N:])**2
prob_density_2 = np.abs(psi_2[:N])**2 + np.abs(psi_2[N:])**2

# Gráfico 2: Densidad de Probabilidad
plt.subplot(1, 2, 2)
sitios = np.arange(1, N + 1)
plt.bar(sitios, prob_density_1, color='blue', alpha=0.6, label='Modo Cero 1')
plt.bar(sitios, prob_density_2, color='orange', alpha=0.5, label='Modo Cero 2')

plt.title(f'Localización Espacial ($\mu={mu_topologico}$)', fontsize=14)
plt.xlabel('Sitio de la cadena $i$', fontsize=12)
plt.ylabel('Densidad de Probabilidad $|\psi|^2$', fontsize=12)
plt.xticks(np.arange(0, N+1, 5))
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
#plt.savefig('analisis_clasico_kitaev.png', dpi=300)
plt.show()

