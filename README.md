# Simulación Clásica y Cuántica de la Cadena de Kitaev: Fases Topológicas y Modos de Majorana

![Estado del Proyecto](https://img.shields.io/badge/Estado-En%20progreso-yellow)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-IBM_Quantum-purple)

Este repositorio contiene el código fuente, los gráficos de resultados y el informe final del proyecto de investigación computacional sobre la **Cadena de Kitaev unidimensional ($N=3$)**. El objetivo de este proyecto es simular la aparición de Modos Cero de Majorana (MZMs) mediante métodos clásicos (Diagonalización Exacta) y algoritmos cuánticos (Variational Quantum Eigensolver - VQE), demostrando su viabilidad en simuladores y procesadores cuánticos reales (NISQ).

**Autor:** Maximiliano Bruna  
**Institución:** Universidad Técnica Federico Santa María (UTFSM) - FIS-205  

---

## Descripción del Proyecto

La cadena de Kitaev es un modelo teórico fundamental para la computación cuántica topológica tolerante a fallos. En este proyecto, se evita depender de "cajas negras" prefabricadas:
1. **Diagonalización Exacta (ED):** Construcción manual del Hamiltoniano de Bogoliubov-de Gennes (BdG) y uso de la transformación de Jordan-Wigner para obtener el espectro de muchos cuerpos.
2. **VQE Clásico:** Desarrollo de un Eigensolver Cuántico Variacional "from scratch" implementando un Ansatz parametrizado y optimizando la energía del estado fundamental mediante el optimizador clásico COBYLA.
3. **VQE en Hardware Cuántico (IBM):** Adaptación del circuito para su transpilación y ejecución en computadores cuánticos reales utilizando Qiskit Runtime Primitives y el manejo de sesiones (Sessions) *(Fase actualmente en progreso)*..

Los resultados confirman el cierre del gap de energía, la degeneración del estado fundamental en el régimen topológico ($|\mu| < 2|t|$), la conservación de la paridad macroscópica con números de partículas fraccionarios, y las firmas de correlación de borde no locales que caracterizan a los fermiones de Majorana.

---

## Estructura del Repositorio

El repositorio está organizado de la siguiente manera:

* **`Kitaev_chain_main.py`**: Script principal de simulación clásica. Construye y diagonaliza exactamente el Hamiltoniano BdG. Incluye el cálculo del espectro de energía, el número de partículas y las correlaciones de borde separadas por paridad (par/impar). (en caso de no poder ver el informe en Github, descargarlo y debería verse normal. En caso contrario avisar inmediatamente a mbruna@usm.cl)
* **`VQE.py`**: Implementación matricial (basada en NumPy y SciPy) del algoritmo VQE. Define los operadores de espín (Pauli), la transformación de Jordan-Wigner y minimiza la función de costo iterativamente.
* **`VQE_for_QC.py`**: Script preparado para infraestructura cuántica real. Utiliza `qiskit.circuit` y `qiskit_ibm_runtime` para adaptar el Hamiltoniano teórico a la topología física (ISA) del hardware cuántico de IBM.
* **`Informe_Kitaev_Chain-4.pdf`**: Documento detallado tipo paper (formato Physical Review) que consolida la metodología, las matemáticas subyacentes, y el análisis físico de los resultados obtenidos.
* **`/images`**: Carpeta que contiene las gráficas vectoriales generadas a partir de las simulaciones, fundamentales para el análisis de la transición de fase topológica. Incluye:
  * `Espectro de Energía muchos cuerpos(cadena completa).pdf`
  * `Espectro de energía muchos cuerpos (Paridad).pdf`
  * `Número de particulas.pdf` / `Número de particulas(VQE).pdf`
  * `Correlación de los majorana.pdf`
  * `Majorana Edge Correlation across the Many-Body Spectrum(Paridad).pdf`

---

## Requisitos e Instalación

Para ejecutar los scripts locales (`Kitaev_chain_main.py` y `VQE.py`), necesitas las siguientes librerías de Python:

```bash
pip install numpy scipy matplotlib

```
Para el script cuántico (VQE_for_QC.py), es necesario instalar Qiskit y configurar tus credenciales de IBM Quantum dentro del propio código:

```bash
pip install qiskit qiskit-ibm-runtime
```
```bash
service = QiskitRuntimeService(channel="ibm_quantum_platform", token="TOKEN_AQUÍ")
```
**Aviso de Dependencias:** Durante el desarrollo de este proyecto, se detectaron posibles conflictos de versiones entre dependencias clásicas y las herramientas de simulación cuántica. Por ello, **se recomienda encarecidamente crear un entorno virtual limpio de Conda** antes de instalar las librerías.

### 1. Creación del Entorno Virtual (Recomendado)
Se puede crear y activar un entorno llamado `kitaev_env` (recomendamos Python 3.10 para máxima compatibilidad con Qiskit) usando los siguientes comandos en tu terminal:

```bash
conda create -n kitaev_env python=3.10
conda activate kitaev_env
```

---
## Uso y Ejecución

1. **Simulación Clásica: Diagonalización Exacta (ED):**
Para correr la solución exacta basada en la matriz de Bogoliubov-de Gennes (BdG) de muchos cuerpos y mapear la física exacta del sistema, ejecuta:
```bash
python Kitaev_chain_main.py
```
* Dinámica Interna *: El script calcula las energías y los estados propios barriendo el potencial químico (μ). Para cada estado, evalúa el valor esperado del operador de número de partículas y el operador de correlación de borde de Majorana.
* Clasificación por Paridad ($Z^2$) *: El código proyecta cada autoestado sobre el operador de paridad fermiónica. Clasifica automáticamente los resultados en dos arreglos independientes: Paridad Par (+1) y Paridad Impar (-1). Esto permite una visualización limpia en el gráfico final sin cruces caóticos de líneas, evidenciando analíticamente cómo la degeneración del estado fundamental se vuelve perfecta en la fase topológica ($∣μ∣<2∣t∣$).

2. **Algoritmo VQE Clásico (Optimización con Ansatz Local):**
Para evaluar el desempeño del algoritmo cuántico variacional utilizando una representación matricial exacta en la máquina local
```bash
python VQE.py
```
* Arquitectura del Ansatz: Construye un circuito parametrizado profundo con un diseño estructural de dos capas cuánticas (layers=2) para un sistema de $N=3$ espines/qubits. Aplica rotaciones locales $R_y(θ)$ y $R_z(θ)$ combinadas con entrelazamiento no local mediante compuertas CNOT, controlando un total de 18 parámetros angulares independientes.
* Mitigación de Mínimos Locales: Para evitar quedar atrapado en mínimos locales (problema recurrente debido a los Barren Plateaus), el loop ejecuta 3 intentos aleatorios independientes (attempts=3) para cada valor de μ. El optimizador clásico COBYLA se encarga de minimizar la función de costo calculando los valores esperados energéticos paso a paso.
* Salida: Grafica el número de partículas promedio comparando la curva obtenida mediante la optimización cuántica variacional heurística frente al resultado teórico de la diagonalización clásica.

3. **Ejecución Cuántica Real: Qiskit Runtime Primitives (IBM QPU):**

Para adaptar y transpolar el circuito variacional con el objetivo de ejecutarlo sobre procesadores cuánticos NISQ reales en la nube de IBM Quantum:
```bash
python VQE_for_QC.py
```
* Compilación y Topología Física (ISA): A diferencia de la simulación teórica, este código implementa restricciones de hardware real. Utiliza generate_preset_pass_manager (con un nivel de optimización básico) para traducir las compuertas lógicas abstractas del Ansatz al mapa físico de acoplamientos (líneas físicas de CNOT) del chip seleccionado. Del mismo modo, el Hamiltoniano cuántico se reestructura automáticamente usando .apply_layout().
* Uso Eficiente de Sesiones (Qiskit Runtime Sessions): Para la optimización variacional con hardware real, el script inicializa el primitivo RuntimeEstimator configurado en el backend físico. El algoritmo está programado para enviar la tupla de datos (circuito compilado, Hamiltoniano mapeado y parámetros clásicos) en formato PUB de la V2 de Qiskit.
* Recomendación operativa: Para reducir los tiempos de cola en el clúster cuántico de IBM, se sugiere envolver este bucle iterativo de COBYLA dentro de un gestor de contexto with Session(backend=backend) as session:, garantizando que la QPU procese secuencialmente los 25 pasos máximos programados (maxiter=25) sin volver a formarse en la cola general en cada iteración.

## Referencias Principales
Este proyecto fue fuertemente inspirado y guiado por la literatura científica clave en el área de la materia condensada topológica y los algoritmos variacionales cuánticos:

* Kitaev, A. Y. (2001). Unpaired Majorana fermions in quantum wires. Physics-Uspekhi.
* Rančić, M. J. (2023). Exactly solving the Kitaev chain and generating Majorana-zero-modes out of noisy qubits.
* McClean, J. R., et al. (2016). The theory of variational hybrid quantum-classical algorithms.
* Cerezo, M., et al. (2021). Variational quantum algorithms. Nature Reviews Physics.


Ain't nobody here but us chickens

