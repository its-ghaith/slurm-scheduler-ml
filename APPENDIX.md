# Technischer Anhang: PoC4

Stand: 8. September 2026

Dieses Repository enthaelt den reproduzierbaren technischen Anhang der Masterarbeit. Laufzeitdaten, Zugangsdaten, Caches und lokale Arbeitsdateien gehoeren nicht in die Abgabe.

## Empfohlener Inhalt des Anhangs

- `README.md`: Architektur, Versuchsablaeufe und Betriebsanleitung
- `controller_benchmark/`: Controller, Benchmark-Logik und Auswertung
- `slurm/`: SLURM-Jobs, Training und Energieerfassung
- `scripts/`: reproduzierbare Start-, Analyse-, Backup- und Deploy-Skripte
- `grafana/`: provisionierte Datenquelle, Dashboard-Provider und Dashboard-Exporte
- `mlflow/`: Inventar und Konventionen fuer Experimente und Runs
- `prometheus/`: Scrape-Konfiguration
- `rancherConfigs/slurm-stack.yaml`: Kubernetes-Ressourcen ohne lokale Kubeconfig
- `results/`: wissenschaftliche Ergebnisartefakte; grosse Rohdaten und Caches sind ausgeschlossen
- `tests/`: automatisierte Pruefungen
- `docs/`: ergaenzende Betriebs- und Thesis-Dokumente

## Nicht in den Anhang aufnehmen

- `.git/`, virtuelle Umgebungen und `__pycache__`
- `rancherConfigs/main.yaml` sowie andere lokale oder geheime Kubeconfigs
- `.env` und sonstige Zugangsdaten
- MLflow-Backend-Datenbanken, Artefakt-Stores und lokale `mlruns/`
- Grafana-Laufzeitdaten
- lokale Codex-, QA-, Download- und temporaere Arbeitsverzeichnisse

## Beobachtbarkeit

Grafana wird aus den JSON-Dateien unter `grafana/dashboards/` provisioniert. Der Dashboard-Katalog und die Zuordnung zum Anhang stehen in `grafana/README.md`.

MLflow bleibt das Laufzeitsystem fuer Parameter, Metriken und Artefakte. Der Anhang enthaelt bewusst nur das Experimentinventar und die Namens-/Tag-Konventionen unter `mlflow/`; dadurch werden keine unnoetig grossen oder potenziell sensiblen Laufzeitdaten verteilt.

## Abgabekontrolle

Vor dem Packen sollten alle Dashboard-JSONs parsebar sein, alle UIDs eindeutig sein und `git status --short --ignored` keine versehentlich unignorierten Laufzeitdaten zeigen. Die lokale Kubeconfig darf nie Bestandteil des Archivs sein.
