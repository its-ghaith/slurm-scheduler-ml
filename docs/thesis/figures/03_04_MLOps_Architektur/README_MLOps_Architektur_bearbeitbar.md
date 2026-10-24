# MLOps-Architektur – bearbeitbare yEd-Fassung

Ergänzung zu Abbildung 3.4 für die Masterarbeit. Die ursprünglichen Dateien bleiben bestehen.

- `MLOps_Architektur_3_4_bearbeitbar.graphml`: in yEd vollständig bearbeitbare Gruppen, Knoten, Piktogramm-Grundformen und Pfeile.
- `MLOps_Architektur_3_4_bearbeitbar.svg`: Vektorvorschau für das Textdokument.

Konzept: Arbeitsplatz außerhalb Kubernetes, darin drei funktionale Bereiche Steuerung, Ausführung und Tracking/Monitoring. Durchgezogene Pfeile kennzeichnen Steuerung und Ausführung; gestrichelte Pfeile Datenflüsse. MLflow und Prometheus sind **nicht direkt miteinander verbunden**; die Datenkorrelation erfolgt über Versuchs-, SLURM-Job- und MLflow-Run-IDs. Der Live-Shadow-Hinweis ist bewusst nicht in der Grafik.

Personen-/Computer-Piktogramme sind bewusst als eigenständige yEd-Grundformen umgesetzt und bleiben veränderbar; es werden keine externen Bildressourcen vorausgesetzt. Die Grafik zeigt logische Verantwortlichkeiten und keine vollständige physische Netzwerktopologie.

Bildunterschrift: *Konzeptionelle Architektur der MLOps-Experimentierumgebung. Quelle: Eigene Darstellung.*
