# Simulación Clásica y Cuántica de la Cadena de Kitaev: Fases Topológicas y Modos de Majorana

![Estado del Proyecto](https://img.shields.io/badge/Estado-En%20progreso-yellow)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
![Qiskit](https://img.shields.io/badge/Qiskit-IBM_Quantum-purple)

Este repositorio contiene el código fuente, los gráficos de resultados y el informe final del proyecto de investigación computacional sobre la **Cadena de Kitaev unidimensional ($N=3$)**. El objetivo de este proyecto es simular la aparición de Modos Cero de Majorana (MZMs) mediante métodos clásicos (Diagonalización Exacta) y algoritmos cuánticos (Variational Quantum Eigensolver - VQE), demostrando su viabilidad en simuladores y procesadores cuánticos reales (NISQ).

**Autor:** Maximiliano Bruna  
**Institución:** Universidad Técnica Federico Santa María (UTFSM) - FIS-205  

---

## Descripción del Proyecto

La cadena de Kitaev es un modelo teórico fundamental para la computación cuántica topológica tolerante a fallos. En este proyecto, evitamos depender de "cajas negras" prefabricadas y construimos la física desde los cimientos:
1. **Diagonalización Exacta (ED):** Construcción manual del Hamiltoniano de Bogoliubov-de Gennes (BdG) y uso de la transformación de Jordan-Wigner para obtener el espectro de muchos cuerpos.
2. **VQE Clásico:** Desarrollo de un Eigensolver Cuántico Variacional "from scratch" implementando un Ansatz parametrizado y optimizando la energía del estado fundamental mediante el optimizador clásico COBYLA.
3. **VQE en Hardware Cuántico (IBM):** Adaptación del circuito para su transpilación y ejecución en computadores cuánticos reales utilizando Qiskit Runtime Primitives y el manejo de sesiones (Sessions).

Los resultados confirman el cierre del gap de energía, la degeneración del estado fundamental en el régimen topológico ($|\mu| < 2|t|$), la conservación de la paridad macroscópica con números de partículas fraccionarios, y las firmas de correlación de borde no locales que caracterizan a los fermiones de Majorana.

---

## Estructura del Repositorio

El repositorio está organizado de la siguiente manera:

* **`Kitaev_chain_main.py`**: Script principal de simulación clásica. Construye y diagonaliza exactamente el Hamiltoniano BdG. Incluye el cálculo del espectro de energía, el número de partículas y las correlaciones de borde separadas por paridad (par/impar).
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

## 🛠️ Requisitos e Instalación

Para ejecutar los scripts locales (`Kitaev_chain_main.py` y `VQE.py`), necesitas las siguientes librerías de Python:

```bash
pip install numpy scipy matplotlib



Ain't nobody here but us chickens

