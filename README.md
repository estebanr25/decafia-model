\# DECAFIA – Modelo YOLOv8m



> Detección automática de enfermedades en hojas de café mediante visión por computadora.



\## Enfermedades detectadas



| Clase | AP50 |

|-------|------|

| Roya del cafeto (\*H. vastatrix\*) | 88.2% |

| Cochinilla (\*P. jamaicensis\*) | 94.1% |

| Minador de la hoja (\*L. coffeella\*) | 94.9% |

| Planta sana | — |



\## Resultados globales (test set, 286 imágenes)



| Métrica | Valor |

|---------|-------|

| mAP50 | \*\*92.4%\*\* |

| mAP50-95 | \*\*76.6%\*\* |

| Precision | 90.5% |

| Recall | 85.8% |

| F1 | 88.0% |



\## Dataset



\- \*\*2,786 imágenes\*\* de campo real

\- Finca en El Socorro, Santander, Colombia (alt. \~1,500 m)

\- \*\*12,810 instancias\*\* anotadas con SAM polygon masks via VIA

\- Split: 1,951 train / 549 val / 286 test



> El dataset completo no está disponible públicamente debido a derechos institucionales (UIS).

> Para solicitar acceso académico: \[esteban.rosas@correo.uis.edu.co](mailto:esteban.rosas@correo.uis.edu.co)



\## Arquitectura



\- \*\*Modelo base:\*\* YOLOv8m (\~26M parámetros)

\- \*\*Épocas:\*\* 100 (mejor en época 98)

\- \*\*Optimizador:\*\* AdamW, lr=0.01

\- \*\*Hardware:\*\* NVIDIA RTX 5070 8GB, CUDA 12.8

\- \*\*Export:\*\* ONNX para inferencia multiplataforma



\## Descargar modelo



El modelo exportado (`.onnx`) está disponible en Hugging Face:



👉 \[huggingface.co/estebanr25/decafia](https://huggingface.co/estebanr25/decafia)



\## Citar este trabajo

```bibtex

@article{rosas2025decafia,

&#x20; title={DECAFIA: Deep Learning-Based Detection of Coffee Leaf Diseases},

&#x20; author={Rosas Ruiz, Luis Esteban and Salom Medina, Andrey Fernando},

&#x20; journal={Sensors},

&#x20; year={2025},

&#x20; publisher={MDPI}

}

```



\## Licencia



MIT License — ver \[LICENSE](LICENSE)

