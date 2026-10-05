# MLflow

MLflow speichert die Laufzeitdaten im Kubernetes-Deployment und nicht im Repository. Fuer den technischen Anhang werden deshalb nur Inventar, Konventionen und die Logging-Implementierung dokumentiert.

## Kuratierung

Alle aktiven Experimente tragen folgende Tags:

- `project=PoC4`
- `appendix.group`: sortierbare Versuchsgruppe
- `appendix.scope`: `primary`, `supporting` oder `development`
- `retention=keep`
- `curation.date=2026-09-08`

Der Stand vom 8. September 2026 umfasst 13 aktive Experimente mit insgesamt 356 Runs. Es wurden keine Experimente oder Runs geloescht.

## Gruppen

- `01-live-shadow-v4`: zwei primaere Live-Shadow-Experimente, 48 Runs
- `02-generalisation`: drei unterstuetzende Generalisierungsexperimente, 113 Runs
- `03-controller-benchmark`: sieben unterstuetzende Benchmark-Experimente, 186 Runs
- `99-development`: ein Entwicklungsexperiment, 9 Runs; nicht als Kernergebnis des Anhangs verwenden

Das vollstaendige Inventar steht in `experiments.csv`.

## Reproduzierbarkeit und Datenschutz

- Experimentnamen bleiben stabil, weil Trainingsskripte sie referenzieren koennen.
- Backend-Datenbank und Artefakt-Store werden nicht in Git aufgenommen.
- Vor einer Abgabe sind Artefakte auf personenbezogene Daten, Tokens, Hostnamen und absolute lokale Pfade zu pruefen.
- Fuer eine Langzeitarchivierung sollte der MLflow-Store separat gesichert und mit Zugriffsrechten sowie Pruefsumme dokumentiert werden.
- Die aktuelle Kubernetes-Konfiguration nutzt fluechtigen `emptyDir`-Speicher. Ein Pod-Neustart kann daher MLflow-Daten verlieren; fuer die finale Archivierung ist ein externer Export oder ein PersistentVolume erforderlich.
