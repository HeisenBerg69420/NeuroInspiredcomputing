# NeuroInspiredcomputing

Bewegungserkennung mit OpenCV und eine getrennte CNN-Basis zur Bildklassifikation.
Die Kamera erkennt bewegte Bereiche; sie verwendet das CNN bisher nicht zur Klassifikation.

## Installation

Für eine neue Umgebung eignet sich Python 3.12. Die vorhandene `.venv` verwendet
Python 3.16 als Vorabversion und bleibt unverändert. Im Projektordner:

```sh
python3.12 -m venv .venv-312
.venv-312/bin/python -m pip install -r requirements.txt
source .venv-312/bin/activate
```

Der explizite Interpreter im Installationsbefehl verhindert, dass Pakete in
der alten `.venv` landen. In VS Code unter **Python: Select Interpreter**
`.venv-312/bin/python` auswählen, falls noch die alte Umgebung aktiv ist.

`requirements.txt` enthält Kamera-, Trainings- und ONNX-Abhängigkeiten.
Für PyTorch mit CUDA unter Linux/Windows gegebenenfalls zuerst die passende
Variante nach der [offiziellen Anleitung](https://pytorch.org/get-started/locally/) installieren.

## Dateien

| Datei | Aufgabe |
| --- | --- |
| `main.py` | Startbefehle `camera`, `train`, `export` |
| `camera.py` | Optical Flow / Frame-Differenz, Konturen und NMS |
| `model.py` | CNN-Architektur `ProkrastCNN` |
| `config.py` | Trainingsvorgaben und projektbezogene Pfade |
| `data.py` | Bilder laden und reproduzierbar in Training/Validierung aufteilen |
| `train.py` | Training, Validierung, Early Stopping und Checkpoints |
| `export.py` | Checkpoint laden und als ONNX exportieren |
| `models/` | Generierte Checkpoints, Labels und Trainingsberichte |

Die CNN-Architektur ist eine neu ergänzte Basis mit drei Faltungsschichten.
Die ursprüngliche `model.py` war leer. Alte Gewichte anderer Architekturen
sind mit dieser Basis nicht automatisch kompatibel.

## Kamera

```sh
python main.py camera
# Alternativ: python camera.py
```

In `camera.py` wählt `USE_OPTICAL_FLOW = True` Farnebäck, `False` aktiviert
`absdiff`. `MIN_MOTION_PIXELS` und `MIN_LUMA_DIFFERENCE` steuern die jeweilige
Empfindlichkeit; `MIN_CONTOUR_AREA` filtert kleine Bereiche. Mit `q` beenden.
Die Kamera benötigt die Betriebssystemberechtigung des startenden Terminals/Editors.

## Dataset und Training

Ein Unterordner pro Klasse, mindestens zwei Bilder je Klasse:

```text
data/
  tasse/
    bild1.jpg
    bild2.jpg
  telefon/
    bild1.jpg
    bild2.jpg
```

```sh
python main.py train --dataset ./data --epochs 30 --batch-size 32 --image-size 128
# Alternativ: python train.py --dataset ./data
python main.py train --help
```

Der Loader sortiert die Klassennamen, skaliert Bilder auf die konfigurierte
Größe und wandelt sie in RGB-Tensoren mit Werten von 0 bis 1 um. Pro Klasse
werden ungefähr 20 % der Bilder für die Validierung reserviert (mindestens
ein Bild, mindestens ein weiteres bleibt im Training). Die Aufteilung nutzt
Seed 42. Zusammengehörige Videoaufnahmen vorher so auswählen, dass keine fast
identischen Bilder die Bewertung zwischen Training und Validierung verfälschen.

Training nutzt CUDA, falls verfügbar, sonst Apple MPS oder CPU. Der beste
Checkpoint wird als `models/PoviNet.pt` gespeichert; er enthält Gewichte,
Architekturkennung, Klassennamen und Bildgröße. `training.csv` enthält die
Metriken, `training.png` den Loss-Verlauf. Jeder Lauf überschreibt diese Dateien;
für getrennte Läufe `--output models/versuch-1` verwenden.

## ONNX-Export

```sh
python main.py export
# Eigene Pfade:
python export.py --checkpoint models/versuch-1/PoviNet.pt --output models/versuch-1/PoviNet.onnx
```

Der Export prüft das Ergebnis mit ONNX Runtime und schreibt zusätzlich eine
`.labels.txt` mit der Reihenfolge der Klassen. Eingabe: Float32 mit Form
`[1, 3, image_size, image_size]`, RGB, Werte 0 bis 1. Ausgabe: Klassen-Logits,
keine Wahrscheinlichkeiten. Für Wahrscheinlichkeiten Softmax anwenden.

Das CNN klassifiziert ganze Bilder bzw. zugeschnittene Objekte. Es ist kein
trainierter Detektor für Bounding Boxes und wird nicht automatisch mit der
Bewegungserkennung verbunden.

## Tests

```sh
python -m unittest discover -s tests -v
```

Die Tests verwenden temporäre Testbilder und prüfen Bewegungserkennung,
Datensplit sowie einen kurzen Trainingslauf mit Speichern und Laden.
Sie benötigen keine Kamera und schreiben keine Modelle in den Projektordner.
