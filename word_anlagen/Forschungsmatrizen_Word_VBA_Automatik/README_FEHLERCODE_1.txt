FORSCHUNGSMATRIZEN – REPARATUR FUER PYTHON-FEHLERCODE 1
========================================================

Der im Word-Makro verwendete feste Ordner lautet:
D:\OLD\Sync\Projekte\Masterarbeit\Project\PoC4\word_anlagen\Forschungsmatrizen_Word_VBA_Automatik

Die haeufige Ursache wurde im alten Paket gefunden:
Das VBA-Makro schreibt matrizen.json ueber ADODB.Stream. Dabei kann
UTF-8-BOM (Bytefolge EF BB BF) am Anfang entstehen. Python las bisher mit
encoding='utf-8' und konnte dann wegen 'Unexpected UTF-8 BOM' abbrechen.
Der beiliegende Renderer liest nun mit 'utf-8-sig' (BOM und ohne BOM).

AENDERUNGEN
-----------
1. forschungsmatrizen.py: akzeptiert JSON mit und ohne UTF-8-BOM.
2. Matrizen_Word_Automatik.bas: zeigt bei erneutem Fehler die letzten
   Zeilen aus render_protokoll.txt unmittelbar in Word an.
3. Beide Matrizen bleiben gleich (zentrierte Titel, blaue Kreise).
4. Gleiche Dateinamen, Word-Bild-Markierungen und fester Windows-Pfad.

SCHRITT FUER SCHRITT
-------------------
1. Im bestehenden Ordner zuerst eine Sicherung der Datei matrizen.json
   machen. Wichtig: Ihre AKTUELLEN Citavi-Nummern duerfen nicht durch
   die mitgelieferte Musterdatei ersetzt werden.
2. Aus dem ZIP die Datei forschungsmatrizen.py in den obigen Ordner
   kopieren und die alte Datei ueberschreiben.
3. Word: Alt+F11. Das bisherige VBA-Modul Matrizen_Word_Automatik
   entfernen und die neue Matrizen_Word_Automatik.bas importieren.
   Alte Bildmarkierungen im Word-Dokument bleiben erhalten.
4. Word: Alt+F8 > Matrizen_Nur_Bilder_Aktualisieren. Dies prueft
   den Renderer, ohne die Citavi-Quellenzahlen neu einzulesen.
5. Wenn erfolgreich: Word: Alt+F8 > Matrizen_Alles_Aktualisieren.

WENN PYTHON TROTZDEM FEHLER MELDET
---------------------------------
Windows PowerShell im obigen Ordner oeffnen:

  py -3 --version
  py -3 -m pip install matplotlib
  py -3 forschungsmatrizen.py --config matrizen.json --ausgabe .

Danach die Datei render_protokoll.txt mit Notepad oeffnen. Die letzte
Fehlermeldung ist entscheidend:

* 'Unexpected UTF-8 BOM': Pruefen, ob die neue .py-Datei verwendet wird.
* 'No module named matplotlib': py -3 -m pip install matplotlib
* 'No suitable Python runtime' oder 'py nicht gefunden': Python 3
  installieren und das Kommando PYTHON_COMMAND im VBA-Modul anpassen.
* 'FileNotFoundError': Dateipfad und Dateinamen kontrollieren.
* 'JSONDecodeError' an anderer Stelle: matrizen.json auf gueltige
  JSON-Syntax kontrollieren, z.B. mit Python json.load.

WICHTIG
-------
- Word F9 aktualisiert keine Python-Bilder. Verwende das VBA-Makro.
- Citavi-Verweise innerhalb der Bilder sind Darstellungstext.
- Das Python-Skript wurde lokal mit BOM und ohne BOM erfolgreich getestet.
- Die konkrete Word-/Citavi-Umgebung konnte hier nicht getestet werden.

DATEIEN
-------
Matrizen_Word_Automatik.bas (korrigiert)
forschungsmatrizen.py (korrigiert)
matrizen.json (MUSTER! bei bestehender Installation NICHT ersetzen)
requirements.txt
Beide Matrizen als SVG und PNG
