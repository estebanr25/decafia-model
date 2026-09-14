# DECAFIA — Bot de WhatsApp e inferencia ONNX (YOLOv8m)

> Detección automática de enfermedades y plagas en hojas de café mediante
> visión por computadora. Este repositorio contiene el **bot de WhatsApp
> en producción** y el **servicio de inferencia ONNX** que lo respalda.
>
> Para el pipeline de entrenamiento, el manifiesto de procedencia del
> dataset y los resultados completos, ver
> [estebanr25/decafia-research](https://github.com/estebanr25/decafia-research).

---

## Clases detectadas

El modelo detecta **tres clases**. Las hojas sanas no son una clase de
detección: se representan como imágenes con etiquetas vacías y un veredicto
sano se deriva de la **ausencia de detecciones por encima del umbral de
confianza (0.50)**.

| ID | Clase | Agente causal | AP50 (test) |
|---:|-------|--------------|------------:|
| 0 | `roya` | *Hemileia vastatrix* | 86.19% |
| 1 | `coco` / vaquita | Gorgojos Curculionidae (*Compsus* sp. / *Epicaerus* sp.) | 96.38% |
| 2 | `minador` | *Leucoptera coffeella* | 94.49% |

> La clase `coco` corresponde a daño foliar por defoliación causada por
> gorgojos de la familia Curculionidae, géneros *Compsus* y *Epicaerus*.
> Referencia taxonómica: Constantino et al. (2013), *Manual del Cafetero
> Colombiano*, Vol. 2, pp. 261–306, Cenicafé.
> DOI: [10.38141/cenbook-0026_25](https://doi.org/10.38141/cenbook-0026_25)

---

## Resultados — conjunto de prueba (349 imágenes, 1 948 instancias)

| Métrica | Valor |
|---------|------:|
| mAP50 | **92.35%** |
| mAP50-95 | **72.93%** |

Las métricas se calculan al umbral de confianza óptimo por clase (F1-máximo).

---

## Dataset

- **2 315 imágenes** de campo real
- Finca en El Socorro, Santander, Colombia (alt. ~1 500 m)
- **12 794 instancias** anotadas en formato YOLO
- Split: 1 618 train / 348 val / 349 test
- 1 113 imágenes de fondo (hojas sanas, etiqueta vacía)

El dataset está publicado en Zenodo bajo licencia **CC BY 4.0**:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19931903.svg)](https://doi.org/10.5281/zenodo.19931903)

---

## Arquitectura

- **Modelo base:** YOLOv8m (~26 M parámetros)
- **Épocas:** 100 (paciencia 20; parada temprana)
- **Optimizador:** AdamW, lr₀ = 0.01, decaimiento lineal hasta 1 × 10⁻⁵
- **Semilla:** 42
- **Hardware:** NVIDIA RTX 5070 Laptop GPU (8 GB VRAM, Blackwell sm_120), CUDA 12.8
- **Export:** ONNX para inferencia multiplataforma

---

## Descargar modelo

El modelo exportado (`.onnx`) y los pesos PyTorch (`.pt`) se distribuyen
junto con el dataset en el registro de Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19931903.svg)](https://doi.org/10.5281/zenodo.19931903)

---

## Citar este trabajo

```bibtex
@dataset{rosas2026decafia,
  author    = {Rosas Ruiz, Luis Esteban and
               Salom Medina, Andrey Fernando and
               Barrero Pérez, Jaime Guillermo},
  title     = {{DECAFIA}: dataset e inferencia YOLOv8m para detección
               de enfermedades en hojas de café},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19931903},
  url       = {https://doi.org/10.5281/zenodo.19931903}
}
```

---

## Repositorios relacionados

| Repositorio | Contenido |
|-------------|-----------|
| [estebanr25/decafia-research](https://github.com/estebanr25/decafia-research) | Pipeline de entrenamiento, manifiesto de procedencia, figuras del paper |
| [estebanr25/decafia-app](https://github.com/estebanr25/decafia-app) | Aplicación Android Flutter (inferencia sin conexión) |
| [estebanr25/decafia-landing](https://github.com/estebanr25/decafia-landing) | Sitio web del proyecto — <https://decaf-ia.netlify.app> |
| [estebanr25/decafia-webcam](https://github.com/estebanr25/decafia-webcam) | Demo en navegador con webcam |
| [estebanr25/decafia](https://github.com/estebanr25/decafia) | Repositorio archivado (tesis 2024, Mask R-CNN) — superado por decafia-research |

---

## Licencia

MIT License — ver [LICENSE](LICENSE.txt)

El dataset y los pesos del modelo se distribuyen bajo
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
