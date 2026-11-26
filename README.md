# CNN zur Bildklassifikation

Ein Convolutional Neural Network, von Grund auf in PyTorch gebaut (ohne Transfer Learning), unterscheidet echte Blumenfotos von KI-generierten Bildern. Die Aufgabe lief als Kaggle-Wettbewerb.

Gruppenprojekt mit Adam im Kurs **Machine Learning 2** (4. Semester, Bachelor Künstliche Intelligenz, HAW Landshut, SoSe 2026).

## Vorgehen

- **Daten:** Klassenverteilung und Bildeigenschaften, Bild-Cache für schnelles Training, eigene Dataset-Klasse, Augmentation gegen Overfitting
- **Modell:** eigene CNN-Architektur mit Dimensions- und Parametertabelle und Gewichts-Initialisierung
- **Training:** Loss, Optimizer und Learning-Rate-Schedule, Lernkurven
- **Evaluation:** Konfusionsmatrix, Fehleranalyse, Konfidenz des Modells
- **Submission:** Vorhersage für die 402 Testbilder im Kaggle-Format

## Ergebnis

| | Konservative Augmentation | Blumenspezifische Augmentation |
|---|---:|---:|
| Beste Validation-Accuracy | 0,746 | **0,783** |
| Train-Accuracy am Ende | 1,000 | 1,000 |

Das Validierungsset hat nur 189 Bilder, der Unterschied zwischen beiden Läufen ist also nicht belastbar. Beide Modelle overfitten. Als nächste Schritte sind im Notebook k-Fold Cross-Validation und ein kleineres Modell beschrieben.

## Inhalt

| Datei | Inhalt |
|---|---|
| `real_vs_ai_classification.ipynb` | das Notebook mit Daten, Modell, Training und Evaluation |
| `praesentation.pptx` | die Projektpräsentation |
| `best_model.pt` | die Gewichte des besten Modells |
| `submission.csv` | die Kaggle-Abgabe |
| `Project_data/` | die Label-Dateien. Die Bilder selbst (über 900 MB) sind nicht im Repository |

## Ausführen

```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn pillow jupyter
jupyter notebook real_vs_ai_classification.ipynb
```

Zum Ausführen müssen die Bilder in `Project_data/train/` und `Project_data/test/` liegen.
