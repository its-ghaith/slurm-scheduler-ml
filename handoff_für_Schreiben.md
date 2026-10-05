# 1) Ziel

Der am 01.09.2026 eingereichte Entwurf soll auf Grundlage der Kommentare der Betreuerin wissenschaftlich und strukturell überarbeitet werden. Das Ziel ist kein bloßes Abarbeiten einzelner Randnotizen, sondern eine konsistente Neufassung mit klarem roten Faden:

- Einleitung, Motivation, Zielsetzung und Ergebnisse müssen funktional voneinander getrennt sein.
- Theoretische Grundlagen, verwandte Arbeiten und der eigene wissenschaftliche Beitrag müssen eindeutig unterscheidbar werden.
- Der eigene Ansatz - insbesondere der aufgabenübergreifende Quality Score sowie die probabilistische und energieadaptive Stop-Entscheidung - soll als Konzeption beziehungsweise Methodik der Arbeit erkennbar sein und nicht unbeabsichtigt als allgemeine Grundlage erscheinen.
- Alle Formeln, Variablen, Einheiten und Begriffe müssen eindeutig, konsistent und für externe Lesende nachvollziehbar eingeführt werden.
- Quellen müssen direkt den jeweils belegten Aussagen zugeordnet sein; das Literaturverzeichnis muss vollständig und einheitlich formatiert werden.
- Redundanzen und unnötige technische Details sollen entfernt werden, ohne die für die spätere Methodik erforderlichen Grundlagen zu verlieren.

Das Ergebnis der Überarbeitung soll eine gutachtertaugliche Fassung sein, in der Problemstellung, Forschungslücke, eigener Beitrag, Methodik und spätere Evaluation logisch aufeinander aufbauen.

# 2) Aktueller Stand

Der Entwurf und die kommentierte Fassung umfassen jeweils 52 Seiten. In der Feedbackfassung wurden 166 Annotationen erkannt. Neben sprachlichen und formalen Hinweisen enthält das Feedback mehrere grundlegende Empfehlungen zur wissenschaftlichen Struktur.

Positiv hervorgehoben wurden insbesondere:

- der rote Faden und die Übergänge im bisherigen Grundlagenkapitel,
- die aufgabenübergreifende Einordnung der Qualitätsmetriken als wertvoller Teil der Arbeit,
- einzelne Übergänge von Qualitätsbetrachtung zu Energieeinsatz,
- die grundsätzlich ausreichende Gliederungstiefe an mehreren Stellen.

Die wichtigsten offenen Überarbeitungspunkte sind:

1. **Einleitung und Motivation trennen:** Die Einleitung soll zuerst begründen, warum der Energieverbrauch von ML-Training wissenschaftlich und praktisch relevant ist. Detaillierte Angaben zur Infrastruktur, zum Controller, zu Vergleichsstrategien oder bereits vermuteten Ergebnissen gehören in Zielsetzung, Methodik oder Ergebnisdarstellung. Wiederholte Absätze zur Fahrzeugerkennung und zu Energieproblemen sind zusammenzuführen. Unklare Sammelbegriffe wie „wissenschaftliche Gesellschaft“ oder „ML-Gesellschaft“ sind durch konkrete Akteure oder sachliche Formulierungen zu ersetzen. Begriffe wie „fair“ müssen operationalisiert werden.

2. **Grundlagen, Stand der Forschung und eigene Konzeption trennen:** Mehrere Abschnitte ab der Formalisierung des task-unabhängigen Quality Scores, der probabilistischen Stop-Regel, der Verlustfunktionen und des energieadaptiven Controllers stellen bereits einen eigenen methodischen Beitrag dar. Diese Inhalte sollen vorzugsweise in ein Konzeptions- beziehungsweise Methodikkapitel verschoben und im Grundlagenkapitel nur vorbereitet werden. Diskussionen konkreter Forschungsarbeiten, etwa zu Layer Freezing, Quantisierung oder energiebezogenen Laufzeitinterventionen, gehören in den Stand der Forschung beziehungsweise in „Verwandte Arbeiten“.

3. **Formale Darstellung konsolidieren:** Viele Symbole sind nicht oder erst nach ihrer ersten Verwendung definiert. Teilweise werden dieselben Zeichen für verschiedene Bedeutungen verwendet, beispielsweise `t` für Epoche und physikalischen Zeitpunkt oder `Delta Q` sowohl für den Qualitätsfortschritt als auch für den Verlust eines Controllers. Auch die verschiedenen Energiegrößen (`E_total`, `E_full`, `E_IT`, Job-, Trainings-, Epochen-, Controller- und Inferenzenergie) sind noch nicht in einer einheitlichen Hierarchie dargestellt. Wegen der hohen Anzahl an Formelzeichen ist ein Formel- beziehungsweise Symbolverzeichnis sinnvoll.

4. **Task-unabhängige Qualität präzisieren:** Die Originalmetriken der Aufgaben, ihre Richtung und ihre Transformation in einen gemeinsamen Quality Score müssen sauber unterschieden werden. Zu klären sind insbesondere die Bedeutung von Quality-Score-Punkten, die Herkunft der Normalisierungsgrenzen, die Vergleichbarkeit marginaler Änderungen zwischen Aufgaben und die weiterhin notwendige Berichterstattung der Originalmetriken. Dieser Teil wurde als wertvoller Beitrag bewertet, benötigt jedoch eine vollständig erklärte Notation und eine klare Zuordnung zur eigenen Methodik.

5. **Early Stopping und Prognosen belegen:** In den Abschnitten zu klassischem Early Stopping, Grenzen geduldsbasierter Regeln, Learning Curve Extrapolation und probabilistischen Entscheidungen fehlen teilweise Quellen oder die Quelle ist nicht eindeutig der belegten Aussage zugeordnet. Abkürzungen wie LC-PFN sind beim ersten Auftreten auszuschreiben. Partielle beziehungsweise unvollständige Lernkurven, Evidenz, Stabilitätsprüfungen, Verlustfunktionen und Schwellenwerte müssen erklärt oder als eigene Designentscheidungen kenntlich gemacht werden.

6. **Energieabschnitt straffen und ordnen:** Der Green-AI-Abschnitt wiederholt frühere Aussagen und soll gekürzt werden. Zuerst ist eine eindeutige Energiehierarchie einzuführen, danach folgen Systemgrenzen und Messverfahren. Die Beziehung zwischen IT-Energie, Rechenzentrumsenergie, PUE, Jobenergie, Trainingsenergie, Epochenenergie, Controllerenergie, Idle-Anteil und Inferenzenergie muss mathematisch und begrifflich widerspruchsfrei sein. Erwartete zukünftige Energie darf nicht mit bereits gemessener Energie vermischt werden. Die CO2-Formel mit zeitabhängiger Emissionsintensität ist zu prüfen.

7. **MLOps-Grundlagen von der konkreten Implementierung abgrenzen:** Die allgemeine Funktion von Kubernetes, SLURM, MLflow, Prometheus und Grafana kann im Grundlagenkapitel bleiben. Die konkrete Komponentenarchitektur, Job-/Phasen-/Epochenzuordnung und projektspezifische Datenflüsse gehören jedoch in das Konzeptions- oder Implementierungskapitel. Für externe Lesende sind Abkürzungen wie SLURM bei der ersten Verwendung zu erklären.

8. **Redundanz und Detailtiefe prüfen:** Formeln sollen nur dort stehen, wo sie für spätere Herleitungen tatsächlich benötigt werden. Mehrere Abschnitte enthalten bereits Methodik oder wiederholen zuvor eingeführte Aussagen. Der Stil ist zu vereinheitlichen; insbesondere soll nicht punktuell Fettdruck eingesetzt werden, wenn er im übrigen Text nicht als Gestaltungsmittel verwendet wird.

9. **Quellenarbeit und Literaturverzeichnis korrigieren:** Quellen sollen am Ende des konkreten Satzes stehen, den sie belegen, nicht pauschal am Absatzende. Fehlende Belege sind zu ergänzen. Doppelte Einträge sind zu entfernen. Alle Einträge benötigen einen einheitlichen Stil und - soweit verfügbar - Autorenschaft, Titel, Jahr, Publikationsorgan, Band/Ausgabe, Seiten, Verlag, ISBN, DOI sowie einen stabilen Link oder ein Repository.

10. **Formale Kleinigkeiten beheben:** Auf dem Titelblatt soll „Externe Betreuung: Tilman Hartwig“ ergänzt werden. Im Inhaltsverzeichnis sind fehlerhafte Leerzeichen unter anderem vor „Epochen“ sowie bei 2.2.5, 2.4.1 und 2.6.1 zu korrigieren. Das Inhaltsverzeichnis und weitere Verzeichnisse sind abschließend vollständig zu aktualisieren.

# 3) Aktive Dateien

- `D:\OLD\Sync\Projekte\Masterarbeit\Doks\Arbeit\Test_Abgabe\01.09_bis_Metriken des Accuracy-Energy-Trade-offs.pdf`  
  Eingereichter, unveränderter Entwurf vom 01.09.2026. Diese Datei dient als Referenz für den Stand vor dem Feedback.

- `C:\Users\ma7am\Downloads\Documents\01.09_bis_Metriken des Accuracy-Energy-Trade-offs_KH.pdf`  
  Kommentierte Fassung der Betreuerin. Sie enthält die maßgeblichen Annotationen und darf bei der Überarbeitung nicht überschrieben werden.

- `D:\OLD\Sync\Projekte\Masterarbeit\Doks\Arbeit\Test_Abgabe\v2.docx`  
  Bisher bekannte bearbeitbare Fassung der Arbeit. Vor der Überarbeitung ist zu prüfen, ob sie inhaltlich exakt dem eingereichten PDF entspricht. Änderungen sollten in einer neuen Revisionskopie erfolgen, damit der bisherige Stand erhalten bleibt.

- `D:\OLD\Sync\Projekte\Masterarbeit\Project\PoC4\handoff.md`  
  Diese Übergabedatei mit Ziel, Arbeitsstand, aktiven Dateien und priorisierten nächsten Schritten.

- `D:\OLD\Sync\Projekte\Masterarbeit\Project\PoC4\tmp\pdfs\feedback_annotations.tsv`  
  Technische Arbeitsdatei mit den extrahierten Annotationen, Seitenzahlen, markierten Textausschnitten und Kommentaren. Sie dient nur der Nachverfolgung und ist kein Bestandteil der Arbeit.

# 4) Nächste Schritte

1. **Bearbeitbare Ausgangsfassung verifizieren und sichern:** Prüfen, ob `v2.docx` dem eingereichten PDF entspricht. Anschließend eine separate Revisionsdatei anlegen; weder der ursprüngliche Entwurf noch die kommentierte PDF werden verändert.

2. **Makrostruktur vor Einzelkorrekturen festlegen:** Für jeden bestehenden Absatz entscheiden, ob er zu Einleitung, Motivation, Zielsetzung, theoretischen Grundlagen, verwandten Arbeiten, eigener Konzeption, Implementierung, Evaluation oder Diskussion gehört. Erst danach sprachliche Detailkorrekturen durchführen.

3. **Einleitung und Motivation neu ordnen:** Die Relevanzkette knapp herstellen: zunehmender ML-Einsatz -> steigender Ressourcenbedarf -> ökologische, wirtschaftliche und wissenschaftliche Folgen -> Bedarf an energieeffizienten Trainingsentscheidungen -> konkrete Forschungslücke. Infrastruktur, Controllerdetails und Ergebnisse aus diesem Teil entfernen und an die passenden späteren Kapitel verschieben.

4. **Kapitel 2 bereinigen:** Nur etablierte und belegte Grundlagen beibehalten. Den eigenen Quality-Score-Ansatz, die probabilistische Stop-Logik, die energieadaptive Entscheidungsfunktion, Verlustfunktionen, Schwellenwertlogik und projektspezifische Messzuordnung in das Methodik-/Konzeptionskapitel verschieben. Konkrete Forschungsansätze und deren Grenzen in „Verwandte Arbeiten“ einordnen.

5. **Notation neu aufsetzen:** Eine Symboltabelle erstellen und anschließend alle Formeln dagegen prüfen. Empfohlen ist eine klare Trennung zwischen Epochenindex, physikalischer Zeit, Stop-Epoche, maximaler Epochenzahl, gemessener Energie, prognostizierter Energie, Qualitätsänderung und Qualitätsverlust. Jedes Symbol muss unmittelbar vor oder nach seiner ersten Formel erklärt werden.

6. **Energiehierarchie definieren:** Eine zentrale Gleichung beziehungsweise Abbildung erstellen, die Job-, Preprocessing-, Initialisierungs-, Trainings-, Validierungs-, Controller-, Finalisierungs- und gegebenenfalls Inferenzenergie verbindet. Danach PUE, Idle-Korrektur, Kosten und CO2 konsistent darauf beziehen. Brutto-, Netto-, gemessene und erwartete Energie getrennt benennen.

7. **Quality-Score-Konzept als eigenen Beitrag ausarbeiten:** Für jede Aufgabenklasse Originalmetrik, Optimierungsrichtung, Transformation, Wertebereich und Interpretation dokumentieren. Explizit erklären, warum gleiche Quality-Score-Differenzen nicht automatisch dieselbe fachliche Bedeutung besitzen und weshalb Originalmetriken parallel berichtet werden.

8. **Quellen- und Literaturprüfung durchführen:** Satzweise kontrollieren, welche Behauptung durch welche Quelle gestützt wird. Fehlende Quellen ergänzen, Literaturdiskussionen in das richtige Kapitel verschieben und das Literaturverzeichnis nach einem einheitlichen Stil vervollständigen. Dubletten entfernen und DOI, ISBN sowie stabile Links ergänzen.

9. **Redundanz- und Verständlichkeitsprüfung:** Wiederholte Definitionen und Aussagen zusammenführen. Alle Abkürzungen und Fachbegriffe beim ersten Auftreten ausschreiben. Übergänge ergänzen, wo die Betreuerin logische Sprünge markiert hat. Die positive Erzählstruktur des bisherigen Grundlagenkapitels soll erhalten bleiben.

10. **Formale Endkontrolle:** Titelblatt, Inhaltsverzeichnis, Überschriftenabstände, Verzeichnisse, Gleichungsnummern, Querverweise und Zitierstil aktualisieren. Danach die neue PDF vollständig gegen alle 166 Annotationen prüfen und für jeden Kommentar dokumentieren, ob er umgesetzt, begründet anders gelöst oder noch offen ist.
