# Grafana

## Versionierter Bestand

Alle reproduzierbaren Dashboard-Exporte liegen in `dashboards/`. `provisioning/datasources/datasource.yml` definiert Prometheus mit der stabilen UID `prometheus`; `provisioning/dashboards/dashboards.yml` laedt die Exporte in den Ordner `SLURM Energy`.

| Datei | Dashboard | Einordnung |
| --- | --- | --- |
| `slurm-energy-overview.json` | SLURM Energy Overview | Kernansicht |
| `slurm-energy-overview-zero-on-idle.json` | SLURM Energy Overview (Zero on Idle) | alternative Kernansicht |
| `controller-benchmark.json` | Controller Benchmark - Stability and Generalisation | Benchmark |
| `rapec-g-v4-live-shadow-cv-nv.json` | RAPEC-G v4 Real Live-Shadow: CV and Non-CV Scratch | Zentrales Thesis-Ergebnisdashboard |
| `energy-guard-study.json` | Generalized Energy Guard - Cross-Stage 20% Validation | Studie |
| `stage-a.json` | Stage A - Seed Generalisation | Studie |
| `stage-b.json` | Stage B - Scenario Generalisation | Studie |
| `8-run.json` | 9 Run - Complete Re-Run | Vergleichsstudie; historischer Dateiname/UID bleibt fuer stabile Links erhalten |
| `7-run.json` | 7 Run | historischer Vergleich |
| `60-run.json` | 60 Run | historischer Vergleich |
| `rapec-g-v4-live-shadow-cv-nv.json` | Thesis Results - Energy-Adaptive MLOps | kuratiertes Hauptdashboard fuer den Anhang |

## Live-Ordnung

Die historischen und studienspezifischen provisionierten Dashboards liegen im Grafana-Ordner `SLURM Energy`. Das kuratierte Hauptdashboard liegt unter `01 Thesis Results`. Nicht provisionierte Entwicklungsdashboards liegen getrennt unter `99 Development - Not in Appendix` und tragen die Tags `development` und `not-in-appendix`.

Diese Trennung vermeidet, dass Explorationsstaende mit den im Anhang dokumentierten Dashboards verwechselt werden. Dashboard-UIDs werden nicht umbenannt, da sie Bestandteil stabiler Links sind.

## Pflege

1. Dashboard in Grafana bearbeiten.
2. JSON exportieren und die vorhandene Datei mit gleicher UID aktualisieren.
3. Instanzspezifische Felder entfernen: `id` muss `null` sein; Secrets oder absolute lokale Pfade duerfen nicht enthalten sein.
4. JSON-Syntax und eindeutige UIDs pruefen.
5. Provisionierung neu ausrollen und die Prometheus-Abfragen kontrollieren.

Grafana-Laufzeitdaten gehoeren nicht ins Repository. Nur Provisioning und exportierte JSON-Dashboards werden versioniert.

## Zentrales Thesis-Dashboard

`rapec-g-v4-live-shadow-cv-nv.json` kombiniert den historischen 60-Run-Messkontext mit den
generalisierten RAPEC-G-v4-Live-Shadow-Ergebnissen. Es gliedert sich in:

1. globale Energie-, Kosten- und CO2-Kennzahlen,
2. Messvergleich, Phasen und Epoch-Verhalten,
3. paarweise RAPEC-G-/Standard-ES-/Full100-Mittelwerte,
4. Dataset-spezifische Stop-, Qualitaets-, Regret-, Overhead- und Energievergleiche,
5. Energy-Quality-Frontier und Energie je normalisiertem Qualitaetsprozentpunkt sowie
6. task-stratifizierte Generalisierung mit 95-%-Bootstrap-Konfidenzintervallen.

Alle Panels besitzen eine Grafana-Beschreibung. Das Info-Symbol am Paneltitel dokumentiert
Formel, Einheit, Datenquelle und Interpretation. Die Rohqualitaetsansicht erzeugt nur fuer
tatsaechlich vorhandene und ausgewaehlte Metriken eigene Unterdiagramme; inkompatible Skalen
werden nicht zusammengelegt.

Das Dashboard wird reproduzierbar erzeugt mit:

```powershell
python controller_benchmark/dashboard/build_thesis_results_dashboard.py `
  --output grafana/dashboards/rapec-g-v4-live-shadow-cv-nv.json `
  --default-run-id 20260812T143435Z-live-shadow-dev `
  --default-curve-case carpk-aerial-vehicle
```

## RAPEC-G-v4 Energy-Quality-Daten

Das zentrale Thesis-Dashboard enthaelt zwei aus den realen Full100-Trajektorien abgeleitete Ansichten:

- `Full100 Quality-Energy Curve with RAPEC-G and ES Stop Points`: eine reale Full100-Kurve mit markierten kontrafaktischen Stopppunkten.
- `GPU Energy per +1 Normalized Quality Point by Dataset`: Energieaufwand je zusaetzlichem normalisierten Qualitaetsprozentpunkt fuer RAPEC-G v4, Standard ES und Full100.

Die kompakten Epoch-Metriken werden aus den unveraenderten `epoch_summary_job_*.json`-Dateien eines Live-Shadow-Snapshots erzeugt:

```powershell
python controller_benchmark/export_thesis_live_shadow_metrics.py `
  --snapshot <snapshot> `
  --output <thesis-live-shadow.prom>
```

Der Export enthaelt die gemessene GPU-Energie, den taskunabhaengigen `quality_score` und vorhandene Rohqualitaetsmetriken. Nicht vorhandene Rohmetriken werden nicht als Nullwerte exportiert. Das Dashboard verwendet fuer taskuebergreifende Quotienten ausschliesslich den normalisierten Quality Score.
