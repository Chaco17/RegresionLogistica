# Tarea 2: Regresión logística — Clasificación de calidad de vinos

**Instituto Tecnológico de Costa Rica** · Inteligencia Artificial

Implementación de **regresión logística con PyTorch** para clasificar vinos tintos como **BUENO** o **MALO** a partir de sus características fisicoquímicas.

## Estudiantes

| Nombre | Carné |
|---|---|
| Sebastián Chacón Muñoz | 2023221908 |
| Daniel Pulido Castagno | |
| Sebastián Muñoz | |

## Dataset

[Red Wine Quality (Cortez et al., 2009)](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009): 1599 vinos tintos, 11 variables fisicoquímicas y una calificación `quality` (0–10).

La variable objetivo se binariza como `target = 1 (BUENO)` si `quality >= 7`, y `0 (MALO)` en otro caso. La justificación está en el notebook.

## Contenido del notebook

1. **Descarga y carga del dataset**
2. **Análisis exploratorio (EDA):** histogramas, boxplots, diagramas de dispersión, matriz de correlación, selección de máximo 6 features y división train / validación / test con *stratified sampling*
3. **Implementación de la regresión logística** en PyTorch y 10 entrenamientos variando *learning rate*, *batch size* y *epochs*
4. **Evaluación:** curvas de pérdida, análisis de overfitting y cuadro comparativo (accuracy, precision, recall, F1-score, AUC)
5. **Análisis de resultados:** mejor modelo, matriz de confusión y métricas finales sobre el set de prueba

## Estructura del repositorio

```
.
├── data/
│   └── winequality-red.csv   # Dataset
├── resultados/               # Gráficos y tablas generadas
├── main.ipynb                # Notebook principal de la tarea
├── requirements.txt          # Dependencias
└── README.md
```

## Ejecución

```bash
pip install -r requirements.txt
jupyter notebook main.ipynb
```

El notebook usa la GPU automáticamente si hay una disponible (CUDA); si no, se ejecuta en CPU. Para usar GPU, instale la versión de PyTorch con CUDA siguiendo las instrucciones de https://pytorch.org/get-started/locally/, por ejemplo:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

## Referencias

- P. Cortez, A. Cerdeira, F. Almeida, T. Matos y J. Reis, "Modeling wine preferences by data mining from physicochemical properties", *Decision Support Systems*, vol. 47, no. 4, pp. 547–553, 2009.
- UCI Machine Learning Repository, [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality).
