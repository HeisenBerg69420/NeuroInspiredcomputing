# NeuroInspiredcomputing

Bewegungserkennung mit OpenCV und AlexNet zur Bildklassifikation.
Die Kamera schneidet erkannte Bewegungsbereiche aus und zeigt pro Boundingbox
das vorhergesagte Label mit Softmax-Score an.

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
| `camera.py` | Optical Flow / Frame-Differenz, Konturen, NMS und Labels pro Box |
| `inference.py` | Klassifikation von Bildern im Speicher mit Checkpoint oder ONNX |
| `model.py` | `AlexNet`, `ProkrastCNN`, Trainingsvorgaben und augmentierter Datenloader |
| `config.py` | Projektpfade und Einstellungen des separaten Datenloaders |
| `data.py` | Bilder laden und reproduzierbar in Training/Validierung aufteilen |
| `train.py` | Training, Validierung, Early Stopping und Checkpoints |
| `export.py` | Checkpoint laden und als ONNX exportieren |
| `models/` | Generierte Checkpoints, Labels und Trainingsberichte |

Standardmäßig wird eine AlexNet-Variante mit den Kanalbreiten
96/256/384/384/256 trainiert. Sie hat fünf Faltungsschichten, ReLUs nach
jeder Faltung, adaptives Pooling auf 6 × 6 und drei Linear-Schichten.
Dies ist keine exakte historische Nachbildung mit LRN und GPU-Gruppierung.
Alternativ bleibt `ProkrastCNN` über `--model prokrastcnn` verfügbar.
Die korrigierte Architektur benötigt neu trainierte Gewichte.

## Kamera

```sh
python main.py camera
# Alternativ: python camera.py
# Eigenen Checkpoint direkt verwenden, ohne ONNX-Export:
python main.py camera --model models/versuch-1/AlexNet.pt
# Oder das exportierte ONNX-Modell samt passender .labels.txt:
python main.py camera --model models/AlexNet.onnx
# Vortrainiertes ImageNet-AlexNet mit 1000 Klassen:
python main.py camera --model models/AlexNet.onnx --imagenet
# Andere Kamera:
python main.py camera --camera-index 1
# Kamera und Bewegungserkennung unabhängig vom Modell testen:
python main.py camera --motion-only
```

Zum Testen muss ein trainiertes Modell vorhanden sein. Standardmäßig lädt die
Kamera `models/AlexNet.pt`, falls vorhanden, sonst `models/AlexNet.onnx`.
Der Checkpoint liefert Klassennamen, Bildgröße und Normalisierung; ONNX nutzt
die zugehörige `.labels.txt` und die Normalisierung mit Mittelwert/Standardabweichung 0.5.
Ohne Gewichte beendet sich das Programm mit einem Hinweis.
`models/AlexNet.labels.txt` enthält die 1000 ImageNet-Klassen in der
Torchvision-Reihenfolge. Diese Datei passt zu einem entsprechend vortrainierten
Modell, nicht automatisch zu einem selbst trainierten AlexNet.
Für das vortrainierte Modell aktiviert `--imagenet` die Skalierung der kurzen
Seite auf 256 Pixel, den mittigen 224-Pixel-Zuschnitt und die ImageNet-Normalisierung.
Die Label-Datei wird vor dem Laden der ONNX-Bibliotheken geprüft.
Startmeldungen zeigen, ob gerade Modellbibliotheken, OpenCV, die Kamera
oder das erste Kamerabild geladen werden. Bleibt eine Meldung stehen,
ist der Start an diesem Schritt noch nicht abgeschlossen.

Halte ein Objekt aus deinen Trainingsklassen in die Kamera und bewege es.
Jede nach NMS verbleibende Box wird aus dem unveränderten Farbbild ausgeschnitten,
von BGR nach RGB konvertiert, auf Modellgröße skaliert und klassifiziert.
Das Modell wird einmal beim Start geladen. Die Inferenz läuft auf der CPU.
Es klassifiziert jede Box in jedem Frame; bei vielen Boxen kann die Bildrate sinken.
Auch unbekannte Objekte erhalten eine der trainierten Klassen; der Score ist
keine Garantie für eine korrekte Erkennung. Stillstehende Objekte erzeugen
mit dieser Bewegungserkennung keine dauerhaften Boxen.

In `camera.py` wählt `USE_OPTICAL_FLOW = True` Farnebäck, `False` aktiviert
`absdiff`. `MIN_MOTION_PIXELS` und `MIN_LUMA_DIFFERENCE` steuern die jeweilige
Empfindlichkeit; `MIN_CONTOUR_AREA` filtert kleine Bereiche. Mit `q` beenden.
Die Kamera benötigt die Betriebssystemberechtigung des startenden Terminals/Editors.

### Start bleibt beim Laden der Bibliotheken stehen

Wenn auch `import numpy` oder `import cv2` lange hängt, kann die Python-Umgebung
ausgelagerte Dateien enthalten. Im Projekt wurden solche Dateien mit dem
macOS-Dateiflag `dataless` gefunden; Python wartet dann beim Import auf deren
lokalen Inhalt. Das betrifft auch ein ausgelagertes ONNX-Modell.

Im Finder für den Projektordner **Geladen behalten** auswählen, sofern diese
Option angeboten wird. Alternativ die Python-Umgebung außerhalb eines
Cloud-synchronisierten Ordners neu anlegen. Nach dem lokalen Laden einen
noch laufenden Start mit `Ctrl+C` beenden und erneut starten.

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
python main.py train --model alexnet --dataset ./data --epochs 30 --batch-size 32 --image-size 256
# Alternativ: python train.py --dataset ./data
python main.py train --help
```

Der Loader sortiert die Klassennamen, skaliert Bilder auf die konfigurierte
Größe und normalisiert RGB-Werte von 0 bis 1 mit Mittelwert 0.5 und
Standardabweichung 0.5 auf den Bereich −1 bis 1. Das Training verwendet
zusätzlich Augmentationen. Die Aufteilung reserviert standardmäßig 15 %
der Bilder zufällig für die Validierung und nutzt Seed 42. Die Vorgaben
stehen in `model.TrainingConfig`. AlexNet benötigt hier mindestens 67 Pixel
pro Seite; für den ONNX-Export ist die getestete Bildgröße 256 empfohlen. Zusammengehörige Videoaufnahmen vorher so auswählen, dass keine fast
identischen Bilder die Bewertung zwischen Training und Validierung verfälschen.

Training nutzt CUDA, falls verfügbar, sonst Apple MPS oder CPU. Der beste
Checkpoint wird als `models/AlexNet.pt` gespeichert (bei ProkrastCNN als
`models/PoviNet.pt`); er enthält Gewichte,
Architekturkennung, Klassennamen und Bildgröße. `training.csv` enthält die
Metriken, `training.png` den Loss-Verlauf. Jeder Lauf überschreibt diese Dateien;
für getrennte Läufe `--output models/versuch-1` verwenden.

## ONNX-Export

```sh
python main.py export
# Eigene Pfade:
python export.py --checkpoint models/versuch-1/AlexNet.pt --output models/versuch-1/AlexNet.onnx
```

Der Export prüft das Ergebnis mit ONNX Runtime und schreibt zusätzlich eine
`.labels.txt` mit der Reihenfolge der Klassen. Eingabe: Float32 mit Form
`[1, 3, image_size, image_size]`, RGB, auf −1 bis 1 normalisierte Werte. Ausgabe: Klassen-Logits,
keine Wahrscheinlichkeiten. Für Wahrscheinlichkeiten Softmax anwenden.

Das CNN klassifiziert ganze Bilder bzw. zugeschnittene Objekte. In der Kamera
liefert die Bewegungserkennung die Boundingboxes und das CNN deren Labels.

## Bilder mit ONNX klassifizieren

```sh
.venv-312/bin/python test.py --model models/AlexNet.onnx bild.jpg
.venv-312/bin/python test.py --model models/AlexNet.onnx ./testbilder --top-k 3
```

`test.py` gibt pro Bild das vorhergesagte Label und den Softmax-Score aus.
Ordner werden einschließlich Unterordnern nach Bildern durchsucht. Die Labels
kommen standardmäßig aus `models/AlexNet.labels.txt`, das beim Export entsteht;
mit `--labels` lässt sich eine andere Datei angeben. Ihre Reihenfolge muss den
Ausgabeklassen des Modells entsprechen.

Die Bildgröße wird aus der ONNX-Eingabe gelesen. Bei dynamischen Bildgrößen
`--image-size 256` ergänzen. Erwartet werden ein RGB-Float32-Eingabetensor im
NCHW-Format und eine Ausgabe mit Klassen-Logits. Die Standardnormalisierung
ist wie im Training `(RGB / 255 - 0.5) / 0.5`. Andere Modelle können mit
`--mean` und `--std` angepasst werden. Die Inferenz läuft über ONNX Runtime
auf der CPU und benötigt kein PyTorch.

## Tests

```sh
python -m unittest discover -s tests -v
```

Die Tests verwenden temporäre Testbilder und prüfen Bewegungserkennung,
Datensplit sowie einen kurzen Trainingslauf mit Speichern und Laden.
Zusätzlich prüfen sie Kameraausschnitte, BGR/RGB-Konvertierung, Labels und
Scores mit einem kleinen ONNX-Testmodell sowie die Checkpoint-Inferenz.
Sie benötigen keine Kamera und schreiben keine Modelle in den Projektordner.
