# 1. Ziel

Das Projekt untersucht, wie ein Machine-Learning-Training automatisch beendet werden kann, sobald der erwartbare zusätzliche Qualitätsgewinn den weiteren Energieverbrauch nicht mehr rechtfertigt. Im Unterschied zu klassischem Early Stopping soll die Entscheidung nicht nur auf einer stagnierenden Validierungsmetrik beruhen. Sie soll den bisherigen Qualitätsverlauf, das Rauschen der Lernkurve, die Unsicherheit zukünftiger Verbesserungen, die pro Epoche gemessene Energie und den Rechenaufwand des Controllers berücksichtigen.

Der praktische Beitrag besteht aus zwei zusammengehörenden Teilen:

- einer reproduzierbaren MLOps- und Messumgebung auf Kubernetes/Rancher mit SLURM, MLflow, Prometheus, Grafana, CodeCarbon und direkter GPU-Leistungsmessung;
- einer austauschbaren, task-unabhängigen Controller- und Benchmark-Plattform für energieadaptives Early Stopping.

Der zuletzt verfolgte wissenschaftliche Ansatz ist RAPEC-G: ein Controller erhält unabhängig von der ML-Aufgabe pro Epoche einen zu maximierenden `quality_score` in `[0,1]`, die gemessene Epochenenergie, die Laufzeit und weitere verfügbare Trainingssignale. Er darf weder Datasetnamen noch Tasknamen, Modellnamen oder die Kennzeichnung Scratch/Pretrained zur Auswahl eines festen Profils verwenden. Aufgabenspezifische Originalmetriken bleiben dennoch gespeichert und werden zusätzlich berichtet.

Die zentrale Forschungsfrage lautet sinngemäß:

> Kann ein online arbeitender, task-unabhängiger und energieadaptiver Early-Stopping-Controller über unterschiedliche ML-Aufgaben, Datasets und Initialisierungen hinweg relevante Trainingsenergie einsparen, ohne einen nicht vertretbaren Verlust der Modellqualität zu verursachen?

Die wissenschaftliche Aussage muss auf den tatsächlich geprüften Geltungsbereich begrenzt bleiben. Abgedeckt sind iterative Trainingsverfahren mit einer Validierungsmetrik pro Epoche beziehungsweise Checkpoint, messbarer Trainingsenergie und speicherbarem Modellzustand. Nicht abgedeckt sind geschlossene Trainingsdienste ohne Zwischenmetriken, nicht iterative Verfahren oder Aufgaben ohne reproduzierbare Validierung.

# 2. Aktueller Stand

## Praktischer Gesamtzustand

Der praktische Teil ist funktional weitgehend fertig. Die Infrastruktur kann Trainingskampagnen asynchron an Rancher übergeben, serverseitig über SLURM ausführen, Energie- und Qualitätswerte erfassen, Controllerentscheidungen speichern, Ergebnisse nach Prometheus exportieren und in Grafana darstellen. Nach erfolgreicher Übergabe kann der lokale Rechner ausgeschaltet werden. Statusskripte und eine optionale öffentliche Statusseite zeigen Run-ID, Job, Case, Laufzeit, letzte Epoche und Qualität an.

Die wiederverwendbare Plattform liegt unter `controller_benchmark/`. Neue Controller implementieren eine gemeinsame Schnittstelle und können ohne Änderung der Mess- und Dashboard-Pipeline eingebunden werden. Baselines werden in einem persistenten, prüfsummenverifizierten Katalog gespeichert, damit Full100 nicht für jede Controlleridee neu trainiert werden muss.

## Finaler Benchmarkumfang

Der vollständige archivierte Entwicklungsstand enthält 48 Full100-Lernkurven. Der Archivlauf `20260816T142848Z-live-shadow-48-archive` ist mit `48/48` Fällen als `COMPLETED` gespeichert. Er setzt sich aus acht Taskfamilien, drei Datasets beziehungsweise Umgebungen je Task und zwei Initialisierungsarten zusammen:

| Taskfamilie | Datasets oder Umgebungen |
|---|---|
| Objekterkennung aus Luftbildern | CARPK, VEDAI, VisDrone |
| Bildklassifikation | CIFAR-10, CIFAR-100, Tiny ImageNet |
| Semantische Segmentierung | Oxford-IIIT Pet, Pascal VOC 2012, UAVid |
| Textklassifikation | GoEmotions, MultiEURLEX, MARC Multilingual |
| Sprachmodellierung | WikiText-103, LM1B-Shard, C4-English-Shard |
| Zeitreihenprognose | Traffic Hourly, Electricity Hourly, Solar 10-Minute |
| Empfehlungssysteme | MovieLens 25M, Amazon Books 5-Core, Amazon Electronics 5-Core |
| Reinforcement Learning | MinAtar Seaquest, MinAtar Asterix, MinAtar Breakout |

Für jeden Fall liegen Scratch- und Pretrained-Varianten vor. Scratch bedeutet zufällige Initialisierung. Pretrained bedeutet einen Source-Domain-Warmstart mit formkompatiblen Gewichten aus einem anderen Dataset derselben Taskfamilie. Die Energie des Source-Pretrainings wird separat behandelt und nicht der Energie des Zieltrainings zugerechnet.

## Mess- und Vergleichsprotokoll

Im Live-Shadow-Verfahren wird ein Dataset physisch genau einmal bis Full100 trainiert. Standard Early Stopping und RAPEC beobachten online denselben Epochenstrom und sehen zu Epoche `e` nur Informationen bis einschließlich `e`. Beim ersten Stoppsignal wird die virtuelle Controlleransicht eingefroren. Der reale gemeinsame Lauf trainiert weiter, um die Full100-Referenz bereitzustellen.

Die primäre Shadow-Energie ist:

```text
E_shadow,c = E_training_epochs_1_to_stop(c) + E_controller_overhead(c)
```

Preprocessing und Finalisierung werden im Shadowvergleich nicht mehrfach auf virtuelle Controller verteilt, da diese Phasen physisch nur einmal ausgeführt wurden. CodeCarbon-Gesamtjobenergie bleibt eine zusätzliche betriebliche Sicht und darf nicht als virtuelle Energie jedes Controllers dupliziert werden. Live Shadow eignet sich zur schnellen, kontrollierten Controllerentwicklung; für eine starke kausale Endaussage sollte der gewählte Controller zusätzlich in echten Stoppläufen mit mehreren Seeds validiert werden.

## Aktuell stärkster Controller

RAPEC-G v4 ist derzeit der stärkste Hauptkandidat für die Arbeit. Er baut auf der profilfreien RAPEC-Logik auf und ergänzt zwei Mechanismen:

- Eine konservative Standard-ES-Referenz erzeugt ein Dominanzsignal. Wenn Standard ES stoppen würde, darf RAPEC nur weitertrainieren, wenn der beobachtete zusätzliche Nutzen pro Energie die Fortsetzung rechtfertigt.
- Ein task-unabhängiger Burst-Schutz erkennt verzögerte, sprunghafte Qualitätsverbesserungen und verhindert, dass ein temporäres Plateau vorschnell als endgültige Konvergenz interpretiert wird.

RAPEC-G v4 verwendet ausschließlich Signale des aktuellen Laufs. Task-, Dataset-, Modell- und Scratch/Pretrained-Namen beeinflussen die Stopplogik nicht.

## Belastbare 48-Fall-Ergebnisse aus Offline-Replay

Die Datei `rapec-g-v5-48-scratch-pretrained-20260816.json` wertet RAPEC-G v4 und v5 auf denselben 48 aufgezeichneten Full100-Epochenverläufen aus. Es wurden dabei keine neuen Trainingsjobs ausgeführt.

| Controller | Fälle | Mittlere Einsparung vs. Full100 | Mittlere Einsparung vs. Standard ES | Mittlerer Qualitätsverlust | Maximaler Qualitätsverlust | Weniger Energie als Standard ES |
|---|---:|---:|---:|---:|---:|---:|
| RAPEC-G v4 | 48 | 65,27 % | 14,52 % | 2,39 pp | 19,35 pp | 34/48 |
| RAPEC-G v5 | 48 | 63,13 % | 10,55 % | 2,28 pp | 19,35 pp | 29/48 |

Diese Werte zeigen einen klaren durchschnittlichen Energiegewinn, aber keine ausnahmslose Qualitätsgarantie. Besonders der maximale Verlust von 19,35 Prozentpunkten muss in der Arbeit als Ausreißer und Grenze des Verfahrens diskutiert werden. Eine Aussage wie „RAPEC-G v4 verliert niemals mehr als 10 pp“ wäre mit dem 48-Fall-Ergebnis nicht zulässig.

Im älteren 24-Fall-Scratch-Replay erreichte RAPEC-G v4 im Mittel 63,07 % Einsparung gegenüber Full100, 1,83 pp mittleren und 8,86 pp maximalen Qualitätsverlust. In 19 von 24 Fällen war die Energie niedriger als bei Standard ES, in vier Fällen gleich und in einem Fall höher. Die besseren Worst-Case-Werte dieses Teilbenchmarks dürfen nicht auf die spätere 48-Fall-Matrix übertragen werden.

## Testzustand

Am 05.10.2026 wurde die Python-Unittest-Suite lokal ausgeführt:

```text
138 Tests ausgeführt
135 erfolgreich
3 Umgebungsfehler
0 fachliche Assertionsfehler
```

Die drei Fehler betreffen nicht die Controllerlogik, sondern fehlende optionale Pakete im aktuell verwendeten lokalen Runtime-Interpreter: PyTorch für Sobol/RF-ParEGO und SALib für die Sensitivitätsanalyse. Die vollständigen wissenschaftlichen Optimierungsanforderungen stehen in den jeweiligen `requirements.txt`-Dateien.

## Versions- und Sicherungszustand

Der Git-Arbeitsbaum ist nicht sauber. Es existieren viele modifizierte und unversionierte Dateien, darunter der gesamte Ordner `controller_benchmark/`, Ergebnisordner, Dashboards, Dokumentation und Änderungen an der SLURM-Trainingspipeline. Unerwartete fremde Änderungen wurden nicht zurückgesetzt. Vor weiterer Entwicklung ist ein vollständiger Snapshot beziehungsweise ein sauberer Commit dringend erforderlich. Die bisherigen Ergebnisordner und historischen Dashboards dürfen nicht überschrieben oder bereinigt werden.

## Stand der schriftlichen Arbeit

Für die schriftliche Arbeit liegen ein Entwurf, eine kommentierte Feedbackfassung der Betreuerin und eine separate Übergabedatei zur Überarbeitung vor. Das Feedback fordert insbesondere die klare Trennung von Grundlagen, verwandten Arbeiten und eigener Konzeption, eine konsistente Formelnotation, weniger Redundanz sowie vollständige und satzgenaue Quellenangaben.

# 3. Aktive Dateien

Alle relativen Pfade beziehen sich auf:

```text
D:\OLD\Sync\Projekte\Masterarbeit\Project\PoC4
```

## Schriftliche Arbeit und Übergaben

- `D:\OLD\Sync\Projekte\Masterarbeit\Doks\Arbeit\Test_Abgabe\v2.docx`: bisher bekannte bearbeitbare Fassung der Arbeit.
- `D:\OLD\Sync\Projekte\Masterarbeit\Doks\Arbeit\Test_Abgabe\01.09_bis_Metriken des Accuracy-Energy-Trade-offs.pdf`: unveränderter eingereichter Entwurf.
- `C:\Users\ma7am\Downloads\Documents\01.09_bis_Metriken des Accuracy-Energy-Trade-offs_KH.pdf`: kommentierte Feedbackfassung der Betreuerin.
- `handoff.md`: Übergabe speziell für die Überarbeitung des schriftlichen Entwurfs.
- `handoff_Projektkomplatte.md`: diese vollständige Projektübergabe.
- `handoff_für_Schreiben.md`: frühere Schreibübergabe; nicht ungeprüft mit den neueren Dateien vermischen.

## Controller-Plattform

- `controller_benchmark/api.py`: task-unabhängige Controller-Schnittstelle, Beobachtungen und Entscheidungen.
- `controller_benchmark/runtime.py`: Controller-Laufzeitintegration.
- `controller_benchmark/task_adapter.py`: Übersetzung aufgabenspezifischer Metriken in den gemeinsamen Quality Score.
- `controller_benchmark/loader.py`: dynamisches Laden von Controllerplugins.
- `controller_benchmark/controllers/live_shadow_suite.py`: parallele Online-Beobachtung mehrerer Controller in einem Full100-Lauf.
- `controller_benchmark/analysis.py` und `controller_benchmark/analyze_live_shadow.py`: Ergebnisaufbereitung und paarweise Auswertung.

## Für die Endauswertung besonders relevante Controller

- `controller_benchmark/controllers/standard_early_stopping.py`: Referenzbaseline.
- `controller_benchmark/controllers/risk_aware_predictive_energy.py`: RAPEC-v3-Kernlogik.
- `controller_benchmark/controllers/rapec_v3_pf.py`: profilfreie RAPEC-v3-Ablation.
- `controller_benchmark/controllers/preference_conditioned_rapec_v3_pf.py`: optionale Energiepriorität und Schwierigkeitsskala.
- `controller_benchmark/controllers/generalized_rapec_energy5.py`: dünner generalisierter Adapter der Energy5-Linie.
- `controller_benchmark/controllers/generalized_rapec_v2.py`: erste Anpassung an unterschiedliche Kurvenregime.
- `controller_benchmark/controllers/generalized_rapec_v3.py`: verbesserte Non-Vision-Anpassung.
- `controller_benchmark/controllers/generalized_rapec_v4.py`: derzeitiger Hauptcontroller.
- `controller_benchmark/controllers/generalized_rapec_v5.py`: v4 mit zusätzlichem dynamischem Reife-/Plateau-Gate.
- `controller_benchmark/controllers/bayesian_guarded_rapec_v3.py`: RAPEC-v8, also RAPEC-v3 mit leichtgewichtigem Bayesian Safety Guard.
- `controller_benchmark/controllers/online_multi_horizon_pareto.py`: RAPEC-v6.
- `controller_benchmark/controllers/online_bayesian_rapec.py`: RAPEC-v7.
- `controller_benchmark/controllers/risk_constrained_bayesian_multi_horizon.py`: RAPEC-v9.
- `controller_benchmark/controllers/rapec_v10.py`: dynamische profilfreie Warm-up-Variante.
- `controller_benchmark/controllers/lcpfn_quality_baseline.py`: Adapter für den externen LC-PFN-Qualitätsvergleich.

Die Versionsnummern sind nicht als eine einzige lineare Versionskette zu verstehen. RAPEC-v3 bis v10 gehören überwiegend zur früheren Neun-Dataset-Entwicklungslinie. RAPEC-G v1 bis v5 bilden die spätere taskübergreifende Generalisierungslinie. Diese Unterscheidung muss in der Arbeit ausdrücklich erklärt werden.

## Aktive Controllerkonfigurationen

- `controller_benchmark/config/rapec-g-v4-live-shadow-controllers.json`: Standard ES und RAPEC-G v4 für reale Live-Shadow-Validierung.
- `controller_benchmark/config/generalized-live-shadow-v5-offline-controllers.json`: v4/v5-Replaykonfiguration.
- `controller_benchmark/config/live-shadow-controllers.json`: frühere Neun-Dataset-Entwicklungscontroller.
- `controller_benchmark/config/rapec-v3-pf-controller.json`: RAPEC-v3-PF.
- `controller_benchmark/config/rapec-v3-pf-preference-controller.json`: Präferenzcontroller.
- `controller_benchmark/config/rapec-v3-v8-controllers.json`: direkter v3/v8-Vergleich.
- `controller_benchmark/config/lcpfn-reference-alignment.json`: Abgleichbedingungen für LC-PFN.

## Benchmarks, Datasets und Runner

- `controller_benchmark/generalized_manifest.py`: Erzeugung der generalisierten Standard-, Hard- und finalen Szenariomatrizen.
- `controller_benchmark/live_shadow_plan.py`: Planung physischer Full100- und virtueller Shadow-Sichten.
- `controller_benchmark/runners/cross_domain_assets.py`: persistente Datasetbereitstellung.
- `controller_benchmark/runners/cross_domain_tasks.py`: Non-Vision-Trainingsaufgaben.
- `controller_benchmark/runners/standard_image_classification.py`: Bildklassifikation.
- `controller_benchmark/runners/standard_semantic_segmentation.py`: Segmentierung.
- `controller_benchmark/runners/vedai_detection.py` und `visdrone_detection.py`: Objekterkennung.
- `controller_benchmark/runners/classification_metrics.py`: Macro-F1 und Klassifikationsmetriken.
- `controller_benchmark/config/benchmark-v3-nine-dataset-fresh.json`: neun Vision-Datasets mit frischen Full100-Baselines.
- `controller_benchmark/config/benchmark-live-shadow-pretrained-nine-dataset.json`: Pretrained-Visionmatrix.

## Rancher-, SLURM- und Statusausführung

- `controller_benchmark/scripts/submit-thesis-results-48-live-shadow-async.ps1`: finale 48-Fall-Kampagne.
- `controller_benchmark/scripts/submit-rapec-generalized-cross-domain-async.ps1`: generalisierte Cross-Domain-Kampagnen.
- `controller_benchmark/scripts/get-controller-benchmark-status.ps1`: einmalige Statusabfrage.
- `controller_benchmark/scripts/watch-controller-benchmark-progress.ps1`: laufende Status- und Logansicht.
- `controller_benchmark/server/orchestrate.py`: serverseitige sequenzielle Orchestrierung, Wiederholungen und Abschlussanalyse.
- `controller_benchmark/server/partial_publisher.py`: Veröffentlichung partieller Ergebnisse.
- `controller_benchmark/status_web/server.py`: HTML- und Textstatusseite.
- `controller_benchmark/slurm/run-generic-benchmark.slurm`: generischer SLURM-Job.
- `rancherConfigs/slurm-stack.yaml`: Rancher/Kubernetes-SLURM-Stack.

## Energie- und Trainingsmessung

- `slurm/train_repro_phase_tracked.py`: Training, Phasen-, Epochen- und Controllerintegration.
- `slurm/epoch_energy_controller.py`: historische Controllerlogik und frühe Experimente.
- `slurm/gpu_energy_utils.py`: zeitliche Integration der GPU-Leistung.
- `slurm/summarize_gpu_metrics.py`: Gesamtjob-GPU-Auswertung.
- `slurm/summarize_gpu_metrics_phases.py`: Phasenauswertung.
- `slurm/summarize_gpu_metrics_epochs.py`: Epochenenergie.
- `slurm/capture_run_metadata.py`: Hardware-, Lauf- und Reproduzierbarkeitsmetadaten.
- `slurm/export_job_metrics_prom.py`, `export_job_phase_metrics_prom.py` und `export_job_epoch_metrics_prom.py`: Prometheus-Exporte.

## Dashboards

- `grafana/dashboards/controller-benchmark.json`: Controllerbenchmark.
- `grafana/dashboards/rapec-g-v4-live-shadow-cv-nv.json`: RAPEC-G-v4-Dashboard für Vision und Non-Vision.
- `grafana/dashboards/7-run.json`, `60-run.json` und `8-run.json`: getrennte historische Dashboards.
- `grafana/dashboards/stage-a.json`, `stage-b.json` und `energy-guard-study.json`: frühere Generalisierungs- und Guard-Studien.
- `controller_benchmark/dashboard/build_thesis_results_dashboard.py`: finales Thesis-Dashboard.
- `controller_benchmark/dashboard/build_live_shadow_dashboard.py`: Live-Shadow-Dashboard.

Cloudflare-`trycloudflare.com`-Adressen sind temporär. Sie sind kein dauerhaftes Projektergebnis und funktionieren nur, solange der zugehörige Tunnelprozess aktiv ist.

## Maßgebliche Ergebnisdateien

- `results/controller-benchmark/offline-replays/rapec-g-v5-48-scratch-pretrained-20260816.json`: vollständige 48-Fall-Auswertung von v4 und v5.
- `results/controller-benchmark/offline-replays/rapec-g-v5-48-scratch-pretrained-20260816.txt`: lesbare Tabelle derselben Auswertung.
- `results/controller-benchmark/offline-replays/status-page-archive/20260816T142848Z-live-shadow-48-archive/status.json`: abgeschlossener 48/48-Archivstatus.
- `results/controller-benchmark/offline-replays/status-page-archive/20260816T142848Z-live-shadow-48-archive/metrics/`: Epochen- und Shadow-Zusammenfassungen der 48 Fälle.
- `results/controller-benchmark/offline-replays/rapec-g-v4-all-datasets-20260812-rerun.json`: 24-Fall-Scratch-Replay.
- `results/controller-benchmark/offline-replays/rapec-g-v4-pretrained-available-20260814.json`: früherer Pretrained-Teilstand.
- `results/controller-benchmark/offline-replays/thesis-presentation-48.prom`: Prometheus-Metriken für Präsentation und Dashboard.
- `results/controller-optimization/`, `controller-scientific-optimization/`, `controller-v8-sensitivity/`, `controller-v8-scientific-optimization/` und `controller-v9-scientific-optimization/`: getrennte Optimierungs- und Sensitivitätsergebnisse; niemals als reale Trainingsresultate darstellen.

# 4. Changes Made

## MLOps- und Messinfrastruktur

- Kubernetes/Rancher wurde als Plattform für die Dienste und die persistente Ausführung eingerichtet.
- SLURM wurde in die Kubernetes-Umgebung integriert, um Trainingsjobs zu planen, GPU-Ressourcen zuzuordnen und Jobs eindeutig zu adressieren.
- MLflow wurde für Parameter, Metriken, Checkpoints und Artefakte angebunden.
- Prometheus sammelt Job-, Phasen-, Epochen- und Controllerzeitreihen.
- Grafana stellt getrennte historische und aktuelle Dashboards bereit.
- CodeCarbon ergänzt die direkte SLURM/GPU-Messung um eine softwarebasierte Gesamtenergiesicht.
- Die GPU-Leistung wird zeitlich integriert; Energie, Kosten und CO2 werden getrennt nach Messquelle und Systemgrenze ausgewiesen.
- Preprocessing, Initialisierung, Training, Validierung, Controllerberechnung, Finalisierung und Gesamtjob können getrennt betrachtet werden.
- Controller-Overhead wird als Laufzeit und, soweit möglich, über Intel RAPL gemessen; andernfalls wird eine explizit gekennzeichnete CPU-Zeit-Schätzung verwendet.

## Reproduzierbare und asynchrone Experimente

- Trainingsjobs können lokal geplant, als Snapshot hochgeladen und serverseitig sequenziell ausgeführt werden.
- Nach der Übergabe läuft die Orchestrierung unabhängig vom lokalen Rechner weiter.
- Jeder Run besitzt eine Run-ID, Run-Matrix, eingefrorene Controllerkonfiguration und einen Workspace-Snapshot.
- Full100-Baselines werden prüfsummenverifiziert zwischengespeichert und kontrolliert wiederverwendet.
- Alte 7-, 9- und 60-Run-Ergebnisse sowie ihre Dashboards wurden getrennt erhalten.
- Ein öffentliches Statusfrontend und PowerShell-Watcher wurden ergänzt.
- Fehlerhafte oder unvollständige Jobs können gezielt wiederholt werden, ohne bereits abgeschlossene Fälle neu auszuführen.

## Entwicklung vom Ausgangsexperiment zum RAPEC-Ansatz

Das ursprüngliche CARPK-Experiment verglich Full100, Standard Early Stopping, Delta-MAPE, einen Uncertainty-Aware Controller sowie feste 20-, 30- und 50-Epochenbudgets. Dabei zeigten sich drei zentrale Probleme:

- Standard ES stoppte in diesem Szenario häufig gar nicht.
- Der ursprüngliche Uncertainty-Controller stoppte bereits um Epoche 20 und unterschätzte spätere Lernfortschritte deutlich.
- Delta-MAPE lieferte einen besseren Kompromiss, war aber noch an feste Schwellen und die konkrete Lernkurve gebunden.

Darauf folgten Hybrid-Pareto-, Conformal- und Energy-Guard-Studien. Mindestepoche, Auswertungsintervall, Patience, Qualitätsgrenze und Energienutzen wurden schrittweise dynamischer. Gleichzeitig wurde sichtbar, dass ein Controller auf CARPK gut funktionieren kann und bei anderen Datasets, Tasks oder Scratch/Pretrained-Verläufen dennoch scheitert.

## RAPEC-v3 bis RAPEC-v10

- RAPEC-v3 wurde zum wichtigen frühen Referenzcontroller, weil er Qualitätsprognose, Unsicherheit, Energie-Utility, Lernzustand und Patience in einer praktikablen Stopplogik kombinierte. Seine Schwäche waren feste beziehungsweise profilabhängige Regeln.
- RAPEC-v4 und v5 experimentierten mit stärker dynamisierten Schwellen und unterschiedlichen Schutzmechanismen. Die Verbesserungen waren nicht über alle Aufgaben konsistent.
- RAPEC-v6 führte Online-Multi-Horizon-Prognosen für Qualität, Energie und Laufzeit sowie Paretoentscheidungen ein. Die höhere Modellkomplexität erhöhte den Controlleraufwand.
- RAPEC-v7 ergänzte umfassende Online-Bayesian-Inferenz und Model Averaging. Der Controller war wissenschaftlich interessant, aber in der Ausführung teilweise sehr langsam und konnte mehr Zeit beziehungsweise Energie als Full100 verursachen.
- RAPEC-v8 behielt die praktikable v3-Stopplogik und ergänzte einen leichteren Bayesian Safety Guard. Diese Variante gefiel, weil sie Bayes zur Absicherung und nicht als vollständigen Ersatz der bewährten Regel verwendete.
- RAPEC-v9 formulierte die Entscheidung als risikobeschränkte bayesianische Multi-Horizon-Optimierung und trennte die äußere Qualitätsbedingung von der Energieoptimierung. Die zugehörige wissenschaftliche Optimierung war jedoch sehr rechenintensiv und ergab nicht frühzeitig eine überzeugende praktische Verbesserung.
- RAPEC-v10 entfernte taskabhängige Profile aus der Mindestepochenwahl und leitete die Freigabe stärker aus dem aktuellen Lauf ab. Die reale Vier-Fall-Prüfung zeigte jedoch keinen ausreichend guten Energie-Qualitäts-Kompromiss.

## Profilfreie und generalisierte Linie

- RAPEC-v3-PF entfernte die Task-, Dataset-, Modell- und Scratch/Pretrained-Profile aus v3. Die Warm-up-Freigabe wird aus Struktur, Rauschen und beobachtetem Lernfortschritt des aktuellen Laufs abgeleitet.
- RAPEC-v3-PF Energy5 verschob nur die zusätzlichen profilfreien Parameter stärker in Richtung Energieeinsparung. Die gemeinsamen v3-Parameter blieben unverändert.
- Der Präferenzcontroller ergänzte eine optionale zehnstufige Energiepriorität und eine optionale zehnstufige Schwierigkeitseingabe. Ohne Schwierigkeitseingabe wird die Schwierigkeit online aus dem aktuellen Lauf geschätzt.
- RAPEC-G v1 übertrug die profilfreie Logik über Taskadapter auf Non-Vision-Aufgaben.
- RAPEC-G v2 und v3 reagierten auf das Problem, dass viele Non-Vision-Kurven mit generischen Regeln an nahezu denselben Epochen stoppten. Lernregime, Plateau- und Recovery-Signale wurden dynamischer ausgewertet.
- RAPEC-G v4 ergänzte die Standard-ES-Dominanzregel und den Burst-Schutz. Diese Variante ist aktuell der beste durchschnittliche Kompromiss.
- RAPEC-G v5 ergänzte ein dynamisches Reife-Gate. Es senkte den mittleren Qualitätsverlust leicht, verlor aber einen Teil der Energieeinsparung und blieb beim Worst Case unverändert.

## Task-unabhängige Qualitätsdarstellung

- Klassifikation verwendet Macro-F1 statt nur Accuracy.
- Objekterkennung verwendet mAP50-95.
- Segmentierung verwendet mIoU beziehungsweise Dice, wobei die Primärmetrik explizit gespeichert wird.
- Fehlermaße wie NRMSE und RMSE werden monoton in einen zu maximierenden Quality Score transformiert.
- Sprachmodellierung verwendet eine monotone Transformation der Cross-Entropy.
- Reinforcement Learning verwendet einen normalisierten Evaluationsreward.
- Originalmetriken bleiben erhalten, weil eine Differenz von `0,01` im normalisierten Quality Score nicht automatisch in jeder Aufgabe dieselbe fachliche Bedeutung besitzt.

## Benchmarkgeneralisation

- CARPK-Stabilität wurde über mehrere Seeds untersucht.
- CARPK-Szenariogeneralisation variierte Datenmenge, Scratch/Pretrained und Modellgröße.
- Datasetgeneralisation wurde unter anderem mit VEDAI und VisDrone erweitert.
- Taskgeneralisation wurde zunächst über Detektion, Klassifikation und Segmentierung und später über fünf weitere Non-Vision-Taskfamilien erweitert.
- Für die Vision-Entwicklung wurden neun Datasets mit frischen Full100-Baselines ausgeführt.
- Für die finale taskübergreifende Sicht wurden 48 Scratch-/Pretrained-Fälle zusammengeführt.

## Wissenschaftliche Optimierung und Sensitivität

- Pareto- und NSGA-II-Optimierungen wurden als frühe Parameterwahl untersucht.
- Eine wissenschaftlichere verschachtelte Optimierung mit Task-Holdouts, Quality-CVaR, Energiequantilen und Nash-Auswahl wurde implementiert.
- Für RAPEC-v8 wurden Morris-Screening, Sobol-Indizes und ALE-Effekte implementiert.
- Die erste und die fokussierte Sobol-Analyse markierten den Surrogaten als nicht zuverlässig; `min_epochs_floor` erschien zwar als stärkster Parameter, durfte aufgrund der unzureichenden Surrogatgüte nicht als belastbarer globaler Sensitivitätsbefund interpretiert werden.
- Für RAPEC-v9 wurden Sobol, qLogNEHVI und RF-ParEGO unter gemeinsamen Budgets vergleichbar gemacht.
- LC-PFN wurde als externe Quality-only-Baseline vorbereitet. Wegen inkompatibler PyTorch-Anforderungen soll LC-PFN in einem separaten, gepinnten Container evaluiert werden und nicht das gemessene Trainingsimage zur Laufzeit verändern.

## Ergebnis- und Dashboarddarstellung

- Jobpunkte werden numerisch beziehungsweise nach Sequenz sortiert.
- Jobnamen enthalten verständliche Case-IDs.
- SLURM- und CodeCarbon-Sichten werden getrennt dargestellt.
- Energie, Kosten, CO2, Qualität, Epochenzahl, Laufzeit und Phasendauer stehen auf Jobebene zur Verfügung.
- Task-unabhängiger Quality Score und aufgabenspezifische Originalmetriken werden parallel dargestellt.
- Scratch und Pretrained können getrennt ausgewertet werden.
- Historische Dashboards wurden wiederhergestellt, ohne die neuen Benchmarks zu überschreiben.

# 5. Failed Attempts

## Inhaltlich gescheiterte oder unzureichende Controllerideen

- Der ursprüngliche Uncertainty-Aware Controller stoppte zu früh. Er interpretierte kurze, verrauschte Anfangsverläufe als endgültige Sättigung und verpasste große spätere Qualitätsgewinne.
- Standard Early Stopping stoppte bei manchen CARPK-Läufen überhaupt nicht und sparte dann keine Energie. In anderen Taskfamilien stoppte es sehr früh und wurde dadurch selbst zu einer starken Baseline.
- Delta-MAPE war auf CARPK brauchbar, aber feste Schwellen ließen sich nicht überzeugend auf andere Tasks übertragen.
- Mehrere Hybrid-, Pareto- und Guard-Varianten erreichten ein gewünschtes Mindestziel nicht gleichzeitig für alle Stufen.
- RAPEC-v6 und v7 waren rechnerisch zu aufwendig. Multi-Horizon-Fits, Bootstrap beziehungsweise Bayesian Model Averaging konnten den Controller langsamer und energetisch teurer als Full100 machen.
- RAPEC-v8 war methodisch attraktiv, aber die Optimierung und Sensitivitätsauswertung zeigten keine belastbare universelle Parameterwahl. Die Surrogatgüte der Sobol-Analyse war unzureichend.
- RAPEC-v9 erforderte sehr lange qNEHVI-Optimierungsläufe. Der beobachtete Fortschritt führte nicht schnell zu einer praktisch überzeugenden Verbesserung; einzelne Läufe wurden abgebrochen.
- RAPEC-v10 löste das Profilproblem konzeptionell, lieferte im ausgewählten Vier-Fall-Test aber keinen zufriedenstellenden Trade-off.
- RAPEC-G v1 funktionierte bei Computer Vision besser als bei frühen Non-Vision-Datasets. Viele Non-Vision-Kurven stoppten nahezu identisch um dieselben Epochen, weil die Kurvenregime nicht ausreichend differenziert wurden.
- RAPEC-G v5 beseitigte den Worst-Case-Qualitätsverlust von v4 nicht. Es sparte durchschnittlich weniger Energie und schlug Standard ES in weniger Fällen.
- Kein bisheriger Controller ist in allen 48 Fällen gleichzeitig besser als Standard ES und ohne relevanten Qualitätsverlust. Dies ist ein Ergebnis und keine Implementierungspanne.

## Fehlgeschlagene oder problematische Experimente

- Ein frühes Reset-PowerShell-Skript wurde unter Windows PowerShell 5 durch unescaped `||` und fehlerhafte Zeichenkodierung nicht geparst.
- Der erste PEP-Benchmark scheiterte bereits im ersten SLURM-Job; ein korrigierter Lauf kam später bis zu den Dataset- und Taskfällen.
- Ein vollständig abgeschlossener 18-Job-Lauf wurde im Orchestrator nachträglich als `FAILED` markiert, weil die abschließende HTTP-Veröffentlichung mit Status 400 scheiterte. Die Trainingsjobs selbst waren abgeschlossen.
- VEDAI-Full100 scheiterte einmal und wurde separat wiederholt.
- VisDrone-Full100 überschritt bei Epoche 86 das damalige 14.400-Sekunden-Limit. Der Fall wurde anschließend gezielt ohne Wiederholung der vorherigen Jobs neu ausgeführt.
- CIFAR-100 zeigte lange keine Epochenmetriken, weil Datasetvorbereitung beziehungsweise Loader vor dem eigentlichen Trainingsloop Zeit benötigten. Statusanzeige und Runner wurden daraufhin robuster gemacht.
- Die Baseline-Importphase meldete `Expected one Full100 source ... found 0`, obwohl 27 Jobs abgeschlossen waren. Ursache war eine unpassende Zuordnung der Full100-Artefakte zur Manifest-ID.
- Der erste generalisierte Cross-Domain-Lauf verwendete irrtümlich das nur für neun Vision-Datasets gedachte Vorbereitungsskript. Datasets wie Adult oder AG News wurden als ungültige Auswahl abgewiesen. Die Cross-Domain-Assetpipeline wurde danach getrennt implementiert.
- Ein `kubectl cp`-Aufruf für generierte Manifeste scheiterte mit `one of src or dest must be a local file specification`; die Pfad- und Uploadlogik wurde korrigiert.
- Mehrere Grafana-Dashboards zeigten zunächst `No data`, weil Metriknamen, Labels, Datasource-Abfragen oder Run-Filter nicht zusammenpassten.
- Eine Dashboardbereitstellung scheiterte trotz abgeschlossener Jobs mit HTTP 400, weil die Grafana-UID länger als 40 Zeichen war.
- Alte Dashboards wurden zeitweise leer, nachdem neue Metriken beziehungsweise Filter die historischen Serien nicht mehr trafen. Daraufhin wurden historische Metriken getrennt wiederhergestellt.
- Öffentliche `trycloudflare.com`-Links lieferten mehrfach Fehler 1033, sobald der temporäre Tunnelprozess nicht mehr lief oder die Adresse wechselte.
- Das PowerShell-Watchskript versuchte zuerst, die schreibgeschützte Variable `$PID` zu überschreiben.
- Das Statusskript rief einmal eine Methode auf einem Nullwert auf, nachdem ein Run am Timeout gescheitert war.
- Live-Logs waren zeitweise leer, weil der Job noch keine Logdatei angelegt hatte oder Pfad und Dateiname erst nach dem SLURM-Start bekannt waren.

## Fehlgeschlagene oder eingeschränkte Optimierungsversuche

- qLogNEHVI fiel ohne Ninja/C++-Extension auf die deutlich langsamere reine Python-Implementierung zurück.
- Wissenschaftliche Optimierungsläufe wurden unterbrochen und mussten mit gespeicherten Fortschrittsereignissen neu gestartet beziehungsweise fortgesetzt werden.
- Die erste RAPEC-v8-Sobol-Analyse und die fokussierte Wiederholung endeten mit `SurrogateReliable: False`. Die numerischen Sobol-Rangfolgen dürfen deshalb nicht als gesicherte Wirkung der Parameter dargestellt werden.
- Die fokussierte RAPEC-v8-Optimierung meldete hohe Energieeinsparung, aber `QualityFeasible: False`; der gefundene Punkt war somit kein zulässiger Endcontroller.
- Die lokale Kompletttestsuite kann die drei wissenschaftlichen Optimierungstests ohne PyTorch, scikit-learn und SALib nicht ausführen. Die übrigen 135 Tests laufen erfolgreich.

## Methodische Grenzen, die nicht „wegkorrigiert“ werden dürfen

- Offline-Replay verwendet aufgezeichnete Full100-Kurven. Es prüft die Entscheidungslogik, beweist aber nicht, dass ein real gestoppter Lauf exakt dieselbe Systemenergie und Finalisierung hätte.
- Live Shadow kontrolliert Hardwarezustand, Datenreihenfolge und Seed sehr gut, führt aber die späteren Epochen physisch trotzdem aus. Die virtuelle Einsparung ist eine kontrafaktische Controlleransicht.
- Ein einzelner Seed pro finalem 48-Fall-Entwicklungsfall reicht nicht für eine allgemeine statistische Stabilitätsaussage.
- Der normalisierte Quality Score stellt eine gemeinsame Richtung und Skala bereit, aber nicht automatisch dieselbe fachliche Bedeutung einer Differenz über alle Tasks.
- Mittlere Einsparungen dürfen Worst-Case-Verluste nicht verdecken.
- Gesamtjobenergie, Trainingsepochenenergie und Controllerentscheidungsenergie haben unterschiedliche Systemgrenzen und dürfen nicht vermischt werden.

# 6. Welche Controller hat mir gefallen

## RAPEC-v3

RAPEC-v3 war der erste Controller, dessen Verhalten als praktisch überzeugend wahrgenommen wurde. Er kombinierte Prognose, Unsicherheit, Energieeffizienz, Lernzustand und Patience, ohne so komplex wie die späteren vollständigen Bayesian-Multi-Horizon-Varianten zu sein. Das Problem war nicht die Grundidee, sondern die task- und szenarioabhängige Profilwahl.

## RAPEC-v8

RAPEC-v8 gefiel als wissenschaftlich nachvollziehbare Weiterentwicklung von v3: Die bekannte v3-Regel bleibt entscheidend, während ein leichter Bayesian Safety Guard einen zu frühen Stopp vetoieren kann. Damit verbindet v8 die Energieorientierung von v3 mit probabilistischer Unsicherheitsabsicherung. Gegen eine endgültige Auswahl sprachen die instabile Parameteroptimierung, die unzuverlässige globale Sensitivitätsanalyse und nicht durchgehend bessere reale Ergebnisse.

## RAPEC-v3-PF

RAPEC-v3-PF gefiel, weil es das Profilproblem direkt adressiert. Derselbe Qualitäts-, Energie- und Telemetrieverlauf erzeugt unabhängig von Task-, Dataset- oder Modellbezeichnung dieselbe Entscheidung. Im früheren Entwicklungsvergleich wurden ungefähr 36,40 % mittlere Einsparung gegenüber Full100 bei etwa 1,69 Prozentpunkten mittlerem Qualitätsverlust berichtet. Diese Werte stammen aus einem kleineren Entwicklungsvergleich und sind nicht mit dem finalen 48-Fall-Benchmark gleichzusetzen.

## RAPEC-v3-PF Energy5

RAPEC-v3-PF Energy5 wurde ausdrücklich als bevorzugte energieorientierte Variante ausgewählt. Es verändert nicht die gemeinsamen RAPEC-v3-Parameter, sondern macht nur die zusätzlichen profilfreien Warm-up- und Freigabeparameter aggressiver. Im zugehörigen kleineren Entwicklungsvergleich wurden ungefähr 52,71 % mittlere Einsparung gegenüber Full100 bei 3,68 Prozentpunkten mittlerem Qualitätsverlust berichtet. Die Variante war ein wichtiger Ausgangspunkt für RAPEC-G, weil sie einen verständlichen Energie-Qualitäts-Kompromiss ohne feste Taskprofile bot.

## RAPEC-G v4

RAPEC-G v4 ist der aktuell empfohlene Hauptcontroller. Er behält den profilfreien Ansatz bei, funktioniert über Vision- und Non-Vision-Tasks und liefert im vollständigen 48-Fall-Replay den besten mittleren Energiegewinn gegenüber Standard ES unter den zuletzt verglichenen v4/v5-Varianten. Seine Stärke ist die Kombination aus Energie-Utility, Standard-ES-Dominanz und Schutz vor verspäteten Qualitätssprüngen.

Die Auswahl ist dennoch als empirischer Kompromiss und nicht als universelle Überlegenheit zu formulieren. v4 verbraucht in 34 von 48 Fällen weniger Energie als Standard ES, aber nicht in allen Fällen. Der maximale Qualitätsverlust beträgt 19,35 Prozentpunkte. Für die Arbeit sollte v4 deshalb als vorgeschlagener Hauptcontroller, v5 als sicherheitsorientierte Ablation, Standard ES als Baseline und Full100 als Qualitäts-/Energiereferenz dargestellt werden.

## Nicht als Hauptcontroller ausgewählt

- RAPEC-v6 und v7: zu hoher Controlleraufwand.
- RAPEC-v9: wissenschaftlich interessant, aber zu komplex und nicht überzeugend validiert.
- RAPEC-v10: profilfrei, aber praktisch schwächere Ergebnisse im ausgewählten Test.
- RAPEC-G v5: etwas geringerer mittlerer Qualitätsverlust, aber weniger Energiegewinn und unveränderter Worst Case.
- Standard ES: unverzichtbare Baseline, aber kein eigener Forschungsbeitrag.

# 7. Nächste Schritte, nachdem der praktische Teil fertig ist

## 1. Projektstand unveränderbar sichern

- Den gesamten aktuellen Arbeitsbaum einschließlich unversionierter Dateien sichern.
- Einen nachvollziehbaren Git-Commit und ein Tag für den Thesis-Endstand erstellen.
- Ergebnisordner, Konfigurationen, Dashboard-JSON, Run-Matrizen und Controllercode gemeinsam archivieren.
- Für große oder ignorierte Dateien zusätzliche SHA-256-Prüfsummen erzeugen.
- Containerimage, Image-Digest, CUDA-, Treiber-, Python- und Bibliotheksversionen dokumentieren.

## 2. Endgültigen Ergebnisdatensatz festlegen

- `rapec-g-v5-48-scratch-pretrained-20260816.json` als zentrale 48-Fall-Auswertungsquelle einfrieren.
- RAPEC-G v4 als Hauptverfahren, RAPEC-G v5 als Ablation, Standard ES als Baseline und Full100 als Referenz festlegen.
- Jede Tabelle mit dem Auswertungsmodus kennzeichnen: reales Training, reales Stoppen, Live Shadow oder Offline-Replay.
- Best-Checkpoint-Qualität, letzte beobachtete Qualität und Testqualität nicht vermischen.
- Validierungsmetriken als Controllerinput und Testmetriken ausschließlich als finale Evaluation kennzeichnen.

## 3. Statistische Endauswertung erstellen

- Paarweise Energie- und Qualitätsdifferenzen pro Case berechnen.
- Ergebnisse getrennt nach Taskfamilie, Scratch/Pretrained und Dataset ausweisen.
- Mittelwert, Median, Standardabweichung, Konfidenzintervalle und Worst Case berichten.
- Makro- und aggregierte Energieeinsparung unterscheiden.
- Pareto-Diagramme und Energy-to-Target, soweit aus den Kurven bestimmbar, ergänzen.
- Die 19,35-pp-Ausreißer einzeln analysieren und nicht nur im Mittelwert verstecken.
- Falls zeitlich möglich, RAPEC-G v4 für eine kleinere repräsentative Teilmenge als echten Stopp mit mehreren Seeds gegen Standard ES und Full100 validieren. Falls das nicht mehr möglich ist, die Generalisierungsaussage entsprechend einschränken.

## 4. Abbildungen und Tabellen reproduzierbar exportieren

- Alle Thesis-Tabellen direkt aus den eingefrorenen JSON-/CSV-Dateien generieren.
- Für jede Abbildung ein Skript, Quelldatei, Filter und Einheit dokumentieren.
- Energy-vs-Quality-Pareto, Stop-Epochen, Einsparung vs. Standard ES, Scratch-vs-Pretrained und taskweise Ergebnisse darstellen.
- SLURM- und CodeCarbon-Energie nicht in derselben Kennzahl zusammenführen.
- Screenshots nur ergänzend verwenden; die numerischen Resultate müssen aus exportierbaren Dateien stammen.

## 5. Methodikkapitel schreiben

- Systemgrenze und Energiegleichungen eindeutig definieren.
- Taskadapter und Quality-Score-Transformationen formal beschreiben.
- RAPEC-G-v4-Algorithmus als Pseudocode darstellen.
- Dynamische Schwellen, Energie-Utility, Dominanzregel, Burst-Schutz und Patience erklären.
- Klar benennen, welche Teile aus Literatur übernommen und welche im Rahmen dieser Arbeit entwickelt wurden.
- Controller-Overhead und Entscheidungszeit als Bestandteil des Verfahrens dokumentieren.

## 6. Implementierungskapitel schreiben

- Kubernetes/Rancher, SLURM, MLflow, Prometheus, Grafana und CodeCarbon in ihrer konkreten Projektrolle erklären.
- Datenfluss vom Jobstart über Epochenmetrik, Controllerentscheidung, Artefaktspeicherung und Prometheus bis Grafana darstellen.
- Asynchrone Orchestrierung, Baselinecache, Wiederholungslogik und öffentliche Statusseite dokumentieren.
- Allgemeine Technologiegrundlagen nicht mit der konkreten Systemarchitektur vermischen.

## 7. Experimentdesign und Ergebnisse schreiben

- Entwicklungsexperimente klar von der finalen Evaluation trennen.
- Die historische Controllerentwicklung als Problem-Lösungsfolge darstellen: zu früher Stopp, fehlender Stopp, Profile, Overhead, Non-Vision-Kurven, v4-Dominanz und Burst-Schutz.
- Den 48-Fall-Benchmark vollständig beschreiben, ohne die 48 Fälle als 48 unabhängige wissenschaftliche Replikationen auszugeben.
- Ergebnisse sowohl gegen Full100 als auch gegen Standard ES berichten.
- Positive und negative Fälle gleichwertig interpretieren.

## 8. Diskussion und wissenschaftlichen Mehrwert formulieren

- Mehrwert gegenüber klassischem ES: Einbezug gemessener Energie, Unsicherheit und task-unabhängiger Qualitätsabbildung.
- Mehrwert gegenüber LC-PFN: nicht nur Qualitätskurvenprognose, sondern energiebezogene Onlineentscheidung und Controllerintegration in eine reale MLOps-Pipeline.
- Mehrwert gegenüber „Energy-efficient Neural Network Training through Runtime Layer Freezing, Model Quantization, and Early Stopping“: Fokus auf eine isolierbare Stopentscheidung, taskübergreifende Qualitätsadapter, reale Epochenenergie und breite Cross-Task-Evaluation.
- Grenzen offen benennen: Replay/Shadow, ein Seed, normalisierte Qualität, Ausreißer, temporäre Cloudflare-URLs und Hardwarebindung der Energiemessung.
- Keine stärkere Generalisierung behaupten, als die acht Taskfamilien und verwendeten Modelle tatsächlich stützen.

## 9. Feedback der Betreuerin umsetzen

- Einleitung, Motivation und Zielsetzung entflechten.
- Grundlagen, verwandte Arbeiten und eigene Konzeption trennen.
- Alle Formelzeichen konsistent definieren und ein Symbolverzeichnis anlegen.
- Energiebegriffe und Systemgrenzen vereinheitlichen.
- Quellen satzgenau zuordnen und das Literaturverzeichnis vervollständigen.
- Redundanzen entfernen und die positiv bewerteten Übergänge erhalten.
- Die Umsetzung der 166 Annotationen mit einer Checkliste nachverfolgen.

## 10. Abschließende Reproduzierbarkeitsprüfung

- Auf einem sauberen Checkout mindestens Planerzeugung, Offline-Replay, Tabellenexport und Dashboardgenerierung testen.
- Die Testabhängigkeiten installieren und die vollständige Suite erneut ausführen.
- Befehle für Setup, PlanOnly, reale Submission, Statusabfrage, Ergebnisdownload und Replay in einer finalen README zusammenführen.
- Prüfen, dass keine geheimen Zugangsdaten, temporären Tunnel-URLs oder persönliche Pfade für die Reproduktion zwingend erforderlich sind.
- Erst danach den praktischen Teil als eingefroren behandeln und nur noch Fehlerkorrekturen mit dokumentierter Auswirkung zulassen.
