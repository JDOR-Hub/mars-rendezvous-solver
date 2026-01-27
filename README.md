# Rendezvous en Marte - Mecánica Celeste

**Cálculo de maniobra de rendezvous alrededor de Marte**  
**Autor:** juandospinor@gmail.com

---

## Descripción

Este proyecto implementa el cálculo de una **maniobra de rendezvous** alrededor de Marte, utilizando mecánica celeste y resolución del **problema de Lambert**.  
El código calcula:

- Posiciones iniciales y finales de una cápsula en órbita.
- Transformaciones al **plano de Lambert**.
- Resolución geométrica y numérica del problema de Lambert.
- Velocidades en el **sistema inercial**.
- **ΔV requeridos** para acoplamiento entre la estación espacial y el vehículo tripulado.
- Visualización de órbitas en 2D y solución geométrica de Lambert.

Todo se realiza en **unidades canónicas** basadas en Marte.

---

## Requisitos

- Python 3.9+
- Librerías Python:
  - `numpy`
  - `matplotlib`
  - `scipy`

Instalación recomendada:

```bash
pip install numpy matplotlib scipy
