# Abbildung 3.4 – Minimalistische MLOps-Architektur

Diese Variante ergänzt den bestehenden Entwurf; die ursprünglichen Dateien bleiben unverändert.

## Gestaltung
- Lesefluss von links nach rechts: Arbeitsplatz → SLURM → Training → MLflow/Monitoring.
- Drei Funktionsbereiche in der On-Premise-Kubernetes-/Rancher-Umgebung.
- Zwei Pfeiltypen: durchgezogen = Steuerung/Ausführung; gestrichelt = gerichteter Datenfluss.
- Die Rückführung vom ES-Controller zum Training bezeichnet die laufende Stop-/Weiter-Entscheidung; im Live-Shadow-Modus wird lediglich eine Empfehlung protokolliert.
- Arbeitsplatz-Piktogramm bezeichnet die Akteurrolle; die übrigen Rechtecke stellen logische Dienste oder Abläufe dar, nicht zwingend separate physische Rechner.

## Inhaltliche Zuordnung
- SLURM plant und startet den Job auf einem Worker.
- Training/Validierung liefert Qualitätsinformationen; Messung erfasst Qualitäts-/Energiemetriken.
- Der Controller wertet die bis zur aktuellen Epoche verfügbaren Beobachtungen aus.
- MLflow dient Run-Tracking und Artefakten.
- Exporter stellt Jobmetriken bereit; Prometheus ruft sie ab; Grafana zeigt abgefragte Zeitreihen.
- Versuchs-ID, SLURM-Job-ID und MLflow-Run-ID ermöglichen die Zusammenführung.

## Einordnung
Die Grafik beschreibt eine **logische Architektur**, kein exaktes Netzwerkschaltbild und keine Garantie für persistente Speicher. Die tatsächliche Datenhaltung und Zugriffswege werden durch die Deployment-Konfiguration bestimmt.

**Bildunterschrift:**
Abbildung 3.4: Konzeptionelle Architektur der MLOps-Experimentierumgebung mit SLURM-gesteuertem Training, energieadaptiver Trainingssteuerung und getrennten Datenpfaden für Experiment-Tracking und Monitoring. Quelle: Eigene Darstellung.

**Dateien:**
- `MLOps_Architektur_3_4_minimal.svg` – für Word/LaTeX als skalierbare Vektorgrafik
- `MLOps_Architektur_3_4_minimal.graphml` – in yEd bearbeitbar
