FORSCHUNGSMATRIZEN – ORIGINALGETREUE ILLUSTRATOR-SVG MIT VBA-ANBINDUNG
===================================================================

WAS IST DIESMAL ANDERS?
----------------------
Matrix 1 basiert DIREKT auf deiner selbst in Illustrator bearbeiteten SVG.
Die Ausgabegrafik ist mit den Standard-Quellenzahlen pixelidentisch zur Vorlage.

Jeder Quadrantenkopf besteht aus GENAU EINEM <text>-Objekt mit formatierten
<tspan>-Abschnitten – wie in deinem gelieferten SVG-Beispiel:
- Matrix 1: oben links 2, oben rechts 2, unten links 2, unten rechts 3 Zeilen.
- Matrix 2: alle vier Quadrantenköpfe genau 2 Zeilen.
- Überschrift: dunkel, fett; Beschreibung: kleiner, hellblau, normal.
- Eine Illustrator-Hauptebene Layer_1, keine vier zusätzlichen Header-Layer.
- Die blauen Kreise bleiben einheitlich.

Für Matrix 2 wurden einige lange Überschriften sinngemäß gekürzt, damit
Überschrift und Beschreibung tatsächlich in je EINE Zeile passen.

QUELLEN UND AUTOMATIK
--------------------
Die Zahlen in eckigen Klammern im Bild sind kein echtes Citavi-Feld.
Das VBA-Makro liest die Nummern aus Word, schreibt matrizen.json,
führt Python aus und ersetzt die zugeordneten SVGs. Die Textstruktur
bleibt bei jeder neuen Ausgabe erhalten.

ZIELORDNER
----------
D:\OLD\Sync\Projekte\Masterarbeit\Project\PoC4\word_anlagen\Forschungsmatrizen_Word_VBA_Automatik

Alle Dateien aus diesem ZIP DIREKT in diesen Ordner kopieren,
NICHT in einen zusaetzlichen Unterordner.

WICHTIG BEIM ERSETZEN
---------------------
- Vorher Sicherheitskopie von matrizen.json machen. Bereits aktualisierte
  Citavi-Nummern nicht versehentlich durch die Beispielwerte ueberschreiben.
- Mindestens diese Dateien muessen zusammen vorliegen:
  forschungsmatrizen.py
  matrizen.json
  Vorlage_Matrix_1_Illustrator.svg  <-- unbedingt mitkopieren!
- Den VBA-Code musst du NICHT neu importieren, wenn dein vorhandenes
  Matrizen_Alles_Aktualisieren bereits mit dem festen Pfad funktioniert.
  Die Moduldatei im ZIP dient als Rueckfallkopie.

PYTHON
------
Die SVG-Erzeugung funktioniert NUR MIT PYTHON 3, ohne Zusatzbibliotheken.
Die PNG-Erzeugung ist optional und verwendet CairoSVG, wenn installiert:
  py -3 -m pip install -r requirements.txt

Manuell neue SVGs erstellen:
  py -3 forschungsmatrizen.py

In Word fuer einen Test ohne neue Citavi-Nummern:
  Matrizen_Nur_Bilder_Aktualisieren
Fuer die volle Aktualisierung:
  Matrizen_Alles_Aktualisieren

Illustrator: Oeffne Matrix_1_...svg beziehungsweise Matrix_2_...svg.
Im SVG ist jeder Quadrantenkopf EIN Textobjekt mit formatierten Textabschnitten.
Die Darstellung kann leicht anders aussehen, wenn Arial nicht verfuegbar ist.

TESTS
-----
Automatisch geprueft:
- Matrix 1: exakt 2/2/2/3 Textzeilen in 4 Textobjekten.
- Matrix 2: exakt 2/2/2/2 Textzeilen in 4 Textobjekten.
- Gleiche SVG-Hauptebene Layer_1 wie die originale Illustrator-Datei.
- Eine numerische Citavi-Aenderung in matrizen.json wird in BEIDEN SVGs aktualisiert.
- Matrix 1 mit originalen Zahlen pixelgenau identisch zur Nutzervorlage.

Grenze: Das VBA-Makro wurde nicht in deiner Windows-Word-/Citavi-Installation ausgefuehrt.
