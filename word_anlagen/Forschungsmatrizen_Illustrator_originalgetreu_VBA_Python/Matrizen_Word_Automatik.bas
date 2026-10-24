Attribute VB_Name = "Matrizen_Word_Automatik"
Option Explicit

' Word Desktop fuer Windows, Citavi, Python 3.
' Alles liegt DIREKT in diesem Ordner (KEIN weiterer Unterordner):
' D:\OLD\Sync\Projekte\Masterarbeit\Project\PoC4\word_anlagen\Forschungsmatrizen_Word_VBA_Automatik
'
' Die TAG_*-Konstanten sind mit frueheren Versionen kompatibel.
' Falls ein frueheres Modul mit gleichem Namen existiert, dieses
' vor dem Import des neuen Moduls im VBA-Editor entfernen.

Private Const MATRIX_FOLDER As String = "D:\OLD\Sync\Projekte\Masterarbeit\Project\PoC4\word_anlagen\Forschungsmatrizen_Word_VBA_Automatik"
Private Const PYTHON_COMMAND As String = "py -3"
Private Const SVG_1 As String = "Matrix_1_Qualitaetsprognose_und_Energiebezug.svg"
Private Const SVG_2 As String = "Matrix_2_Modelllernen_und_probabilistische_Prognose.svg"
Private Const TAG_1 As String = "FM_MATRIX_1 | Qualitaetsprognose und Energiebezug"
Private Const TAG_2 As String = "FM_MATRIX_2 | Modelllernen und probabilistische Qualitaetsprognose"

' ============================================================
' HAUPTMAKRO: aktuelle Citavi-Nummern -> JSON -> Python -> Word
' ============================================================
Public Sub Matrizen_Alles_Aktualisieren()
    Dim doc As Document
    Dim keys As Variant, i As Long, key As String, nummer As String
    Dim jsonFile As String, json As String, logFile As String, folder As String
    Dim bild1 As Boolean, bild2 As Boolean

    On Error GoTo Fehler
    Set doc = ActiveDocument
    folder = Matrix_Ordner()
    jsonFile = folder & "\matrizen.json"
    logFile = folder & "\render_protokoll.txt"

    If Not Dateien_Vorhanden() Then
        Err.Raise vbObjectError + 101, , _
            "Im festen Ordner fehlen matrizen.json oder forschungsmatrizen.py:" & vbCrLf & folder & vbCrLf & _
            "Beide Dateien muessen DIREKT darin liegen."
    End If

    json = Datei_UTF8_Lesen(jsonFile)
    keys = Array("prechelt", "domhan", "reguero", "lcnet", "lcpfn", "zeus")
    For i = LBound(keys) To UBound(keys)
        key = CStr(keys(i))
        nummer = Quelle_Aus_Word(doc, key)
        If Len(nummer) = 0 Then
            Err.Raise vbObjectError + 102, , _
                "Citavi-Quelle '" & key & "' im Literaturverzeichnis nicht eindeutig erkannt." & vbCrLf & _
                "Zuerst Citavi-Literaturverzeichnis aktualisieren. Falls notwendig," & vbCrLf & _
                "den Citavi-Nachweis [n] markieren und das Makro" & vbCrLf & _
                "Matrizen_Quelle_" & key & "_Merken ausfuehren."
        End If
        json = JSON_Quellenwert_Aendern(json, key, nummer)
    Next i

    ' JSON-Sicherung; bei einem Quellenfehler noch KEINE Veraenderung.
    CreateObject("Scripting.FileSystemObject").CopyFile jsonFile, jsonFile & ".bak", True
    Datei_UTF8_Schreiben jsonFile, json

    Python_Rendern folder, logFile

    bild1 = Bild_Mit_Tag_Ersetzen(doc, TAG_1, folder & "\" & SVG_1)
    bild2 = Bild_Mit_Tag_Ersetzen(doc, TAG_2, folder & "\" & SVG_2)

    If bild1 And bild2 Then
        MsgBox "Erfolgreich: Quellenzahlen ermittelt, JSON aktualisiert," & vbCrLf & _
               "Python ausgefuehrt und beide Word-Grafiken ersetzt.", vbInformation
    Else
        MsgBox "Die SVG-Dateien wurden aktualisiert." & vbCrLf & _
               "Matrix 1 in Word: " & IIf(bild1, "ersetzt", "Bild noch nicht zugeordnet") & vbCrLf & _
               "Matrix 2 in Word: " & IIf(bild2, "ersetzt", "Bild noch nicht zugeordnet") & vbCrLf & _
               "Die Grafiken mit den Makros Matrizen_Bild_1_Markieren" & vbCrLf & _
               "und Matrizen_Bild_2_Markieren einmalig zuordnen.", vbExclamation
    End If
    Exit Sub
Fehler:
    MsgBox "Matrizen-Aktualisierung abgebrochen:" & vbCrLf & Err.Description, vbCritical
End Sub

' Optionaler Test: verwendet bereits in JSON gespeicherte Nummern.
' Hilfreich, wenn Citavi-Literaturverzeichnis noch nicht fertig ist.
Public Sub Matrizen_Nur_Bilder_Aktualisieren()
    Dim folder As String, doc As Document, x As Boolean, y As Boolean
    On Error GoTo Fehler
    Set doc = ActiveDocument
    folder = Matrix_Ordner()
    If Not Dateien_Vorhanden() Then
        Err.Raise vbObjectError + 111, , "Python/JSON fehlen: " & folder
    End If
    Python_Rendern folder, folder & "\render_protokoll.txt"
    x = Bild_Mit_Tag_Ersetzen(doc, TAG_1, folder & "\" & SVG_1)
    y = Bild_Mit_Tag_Ersetzen(doc, TAG_2, folder & "\" & SVG_2)
    MsgBox "Bilder neu berechnet. Matrix 1: " & IIf(x, "ersetzt", "nicht zugeordnet") & _
           "; Matrix 2: " & IIf(y, "ersetzt", "nicht zugeordnet"), vbInformation
    Exit Sub
Fehler:
    MsgBox "Fehler beim Neuzeichnen: " & Err.Description, vbCritical
End Sub

' ============================================================
' PFAD: exakt der vom Benutzer angegebene Ordner
' ============================================================
Private Function Matrix_Ordner() As String
    Matrix_Ordner = MATRIX_FOLDER
End Function

Private Function Dateien_Vorhanden() As Boolean
    Dim folder As String
    folder = Matrix_Ordner()
    Dateien_Vorhanden = (Len(Dir$(folder & "\matrizen.json")) > 0) And _
                       (Len(Dir$(folder & "\forschungsmatrizen.py")) > 0)
End Function

Public Sub Matrizen_Diagnose()
    Dim folder As String, m As String
    folder = Matrix_Ordner()
    m = "Fester Ordner:" & vbCrLf & folder & vbCrLf & vbCrLf
    m = m & "matrizen.json: " & IIf(Len(Dir$(folder & "\matrizen.json")) > 0, "OK", "FEHLT") & vbCrLf
    m = m & "forschungsmatrizen.py: " & IIf(Len(Dir$(folder & "\forschungsmatrizen.py")) > 0, "OK", "FEHLT") & vbCrLf
    m = m & "Matrix 1 SVG: " & IIf(Len(Dir$(folder & "\" & SVG_1)) > 0, "OK", "FEHLT") & vbCrLf
    m = m & "Matrix 2 SVG: " & IIf(Len(Dir$(folder & "\" & SVG_2)) > 0, "OK", "FEHLT")
    MsgBox m, vbInformation, "Matrizen-Diagnose"
End Sub

' ============================================================
' PYTHON als lokale CMD-Datei ausfuehren (robust bei Leerzeichen)
' ============================================================
Private Sub Python_Rendern(ByVal folder As String, ByVal logFile As String)
    Dim cmdFile As String, f As Integer, runner As String
    Dim shell As Object, returnCode As Long
    Dim fs As Object, svgA As String, svgB As String

    cmdFile = folder & "\_matrizen_render.cmd"
    runner = "@echo off" & vbCrLf & _
             "cd /d " & Q(folder) & vbCrLf & _
             PYTHON_COMMAND & " " & Q(folder & "\forschungsmatrizen.py") & _
             " --config " & Q(folder & "\matrizen.json") & _
             " --ausgabe " & Q(folder) & " > " & Q(logFile) & " 2>&1" & vbCrLf & _
             "exit /b %ERRORLEVEL%" & vbCrLf

    f = FreeFile
    Open cmdFile For Output As #f
    Print #f, runner;
    Close #f

    Set shell = CreateObject("WScript.Shell")
    ' cmd-Datei mit doppelter Huelle: sicher auch bei Pfaden mit Leerzeichen.
    returnCode = shell.Run("cmd.exe /D /S /C " & Q(Q(cmdFile)), 0, True)
    If returnCode <> 0 Then
        Err.Raise vbObjectError + 115, , _
            "Python meldet Fehlercode " & CStr(returnCode) & "." & vbCrLf & _
            "Protokoll: " & logFile & vbCrLf & _
            "Python-Befehl: " & PYTHON_COMMAND
    End If

    svgA = folder & "\" & SVG_1
    svgB = folder & "\" & SVG_2
    Set fs = CreateObject("Scripting.FileSystemObject")
    If Not fs.FileExists(svgA) Or Not fs.FileExists(svgB) Then
        Err.Raise vbObjectError + 116, , "Python hat nicht beide SVG-Dateien erstellt. Protokoll: " & logFile
    End If
    If fs.GetFile(svgA).Size < 200 Or fs.GetFile(svgB).Size < 200 Then
        Err.Raise vbObjectError + 117, , "Die erzeugten SVG-Dateien sind ungueltig oder leer."
    End If
End Sub

Private Function Q(ByVal wert As String) As String
    Q = Chr$(34) & wert & Chr$(34)
End Function

' ============================================================
' BILDER: bestehende AlternativeText-Markierungen erhalten
' ============================================================
Public Sub Matrizen_Bild_1_Markieren()
    Ausgewaehltes_Bild_Markieren TAG_1
End Sub
Public Sub Matrizen_Bild_2_Markieren()
    Ausgewaehltes_Bild_Markieren TAG_2
End Sub

Private Sub Ausgewaehltes_Bild_Markieren(ByVal tag As String)
    On Error GoTo Fehler
    If Selection.InlineShapes.Count <> 1 Then
        MsgBox "Ein einzelnes Bild markieren. Layout: 'Mit Text in Zeile'.", vbExclamation
        Exit Sub
    End If
    Selection.InlineShapes(1).AlternativeText = tag
    MsgBox "Bild zugeordnet. Die Position bleibt bei spaeteren Updates erhalten.", vbInformation
    Exit Sub
Fehler:
    MsgBox "Bildzuordnung fehlgeschlagen: " & Err.Description, vbCritical
End Sub

Public Sub Matrizen_Bild_1_Einfuegen()
    Bild_Am_Cursor_Einfuegen TAG_1, SVG_1
End Sub
Public Sub Matrizen_Bild_2_Einfuegen()
    Bild_Am_Cursor_Einfuegen TAG_2, SVG_2
End Sub

Private Sub Bild_Am_Cursor_Einfuegen(ByVal tag As String, ByVal name As String)
    Dim img As InlineShape, breite As Single, pfad As String
    On Error GoTo Fehler
    pfad = Matrix_Ordner() & "\" & name
    If Len(Dir$(pfad)) = 0 Then
        Err.Raise vbObjectError + 121, , "Bild nicht vorhanden: " & pfad
    End If
    Set img = ActiveDocument.InlineShapes.AddPicture( _
        FileName:=pfad, LinkToFile:=False, SaveWithDocument:=True, Range:=Selection.Range)
    img.AlternativeText = tag
    img.LockAspectRatio = msoTrue
    breite = ActiveDocument.PageSetup.PageWidth - _
             ActiveDocument.PageSetup.LeftMargin - ActiveDocument.PageSetup.RightMargin
    If img.Width > breite Then img.Width = breite
    MsgBox "Bild eingefuegt und zugeordnet.", vbInformation
    Exit Sub
Fehler:
    MsgBox "Einfuegen fehlgeschlagen: " & Err.Description, vbCritical
End Sub

Private Function Bild_Mit_Tag_Ersetzen(ByVal doc As Document, _
                                      ByVal tag As String, _
                                      ByVal pfad As String) As Boolean
    Dim i As Long, alt As InlineShape, neu As InlineShape, stelle As Range
    Dim widthOld As Single, heightOld As Single
    If Len(Dir$(pfad)) = 0 Then
        Err.Raise vbObjectError + 123, , "SVG nicht vorhanden: " & pfad
    End If
    For i = doc.InlineShapes.Count To 1 Step -1
        Set alt = doc.InlineShapes(i)
        If alt.AlternativeText = tag Then
            widthOld = alt.Width
            heightOld = alt.Height
            Set stelle = alt.Range.Duplicate
            stelle.Collapse wdCollapseStart
            ' Zuerst die NEUE SVG sicher einfuegen; das alte Bild wird
            ' erst nach erfolgreicher Erstellung geloescht.
            Set neu = doc.InlineShapes.AddPicture( _
                FileName:=pfad, LinkToFile:=False, SaveWithDocument:=True, Range:=stelle)
            neu.LockAspectRatio = msoFalse
            neu.Width = widthOld
            neu.Height = heightOld
            neu.LockAspectRatio = msoTrue
            neu.AlternativeText = tag
            alt.Delete
            Bild_Mit_Tag_Ersetzen = True
            Exit Function
        End If
    Next i
    Bild_Mit_Tag_Ersetzen = False
End Function

' ============================================================
' CITAVI: Literaturnummern aus Word lesen, Citavi-Felder nicht editieren
' ============================================================
Private Function Quelle_Aus_Word(ByVal doc As Document, ByVal key As String) As String
    Dim n As String
    n = Quelle_Aus_Literaturverzeichnis(doc, key)
    If Len(n) = 0 Then n = Quelle_Aus_Lesezeichen(doc, key)
    Quelle_Aus_Word = n
End Function

Private Function Titelstueck(ByVal key As String) As String
    Select Case key
        Case "prechelt": Titelstueck = "Early stopping"
        Case "domhan":   Titelstueck = "Speeding up automatic hyperparameter optimization"
        Case "reguero":  Titelstueck = "Energy-efficient neural network training"
        Case "lcnet":    Titelstueck = "Learning curve prediction with Bayesian neural networks"
        Case "lcpfn":    Titelstueck = "Efficient Bayesian Learning Curve Extrapolation"
        Case "zeus":     Titelstueck = "Zeus: Understanding"
    End Select
End Function

Private Function Quelle_Aus_Literaturverzeichnis(ByVal doc As Document, _
                                               ByVal key As String) As String
    Dim p As Paragraph, t As String, nummer As String
    Dim vergleich As String, gefunden As String, count As Long
    Dim re As Object, matches As Object, istBib As Boolean
    Dim marker As Long, anfang As Long

    ' Das letzte Vorkommen der Ueberschrift 'Literaturverzeichnis' verwenden
    ' (das erste kann ein Eintrag im Inhaltsverzeichnis sein).
    For Each p In doc.Paragraphs
        t = Trim$(Replace(Replace(p.Range.Text, vbCr, ""), Chr$(7), ""))
        If LCase$(t) = "literaturverzeichnis" Or LCase$(t) = "bibliography" Then
            anfang = p.Range.End
        End If
    Next p
    If anfang = 0 Then Exit Function

    Set re = CreateObject("VBScript.RegExp")
    re.Pattern = "^\s*\[\s*(\d{1,4})\s*\]"
    re.Global = False
    vergleich = LCase$(Titelstueck(key))

    For Each p In doc.Range(anfang, doc.Content.End).Paragraphs
        t = Replace(Replace(p.Range.Text, vbCr, ""), Chr$(7), "")
        ' Bei Prechelt das Wort 'but when' fordern, um andere Prechelt-Arbeiten
        ' (z. B. Automatic early stopping) nicht zu verwechseln.
        If key = "prechelt" Then
            istBib = (InStr(1, t, "early stopping", vbTextCompare) > 0 And _
                      InStr(1, t, "but when", vbTextCompare) > 0)
        Else
            istBib = (InStr(1, t, vergleich, vbTextCompare) > 0)
        End If
        If istBib Then
            If re.Test(t) Then
                Set matches = re.Execute(t)
                nummer = CStr(matches(0).SubMatches(0))
                count = count + 1
                If Len(gefunden) > 0 And nummer <> gefunden Then Exit Function
                gefunden = nummer
            End If
        End If
    Next p
    If count = 1 Then Quelle_Aus_Literaturverzeichnis = gefunden
End Function

Private Function Quelle_Aus_Lesezeichen(ByVal doc As Document, ByVal key As String) As String
    Dim bookmarkName As String, citation As String
    Dim re As Object
    bookmarkName = "FM_Q_" & UCase$(key)
    If Not doc.Bookmarks.Exists(bookmarkName) Then Exit Function
    citation = doc.Bookmarks(bookmarkName).Range.Text
    Set re = CreateObject("VBScript.RegExp")
    re.Pattern = "\[\s*(\d{1,4})\s*([\];,]|$)"
    If re.Test(citation) Then
        Quelle_Aus_Lesezeichen = CStr(re.Execute(citation)(0).SubMatches(0))
    End If
End Function

Public Sub Matrizen_Quelle_Prechelt_Merken()
    Quellen_Lesezeichen_Setzen "prechelt"
End Sub
Public Sub Matrizen_Quelle_Domhan_Merken()
    Quellen_Lesezeichen_Setzen "domhan"
End Sub
Public Sub Matrizen_Quelle_Reguero_Merken()
    Quellen_Lesezeichen_Setzen "reguero"
End Sub
Public Sub Matrizen_Quelle_LCNet_Merken()
    Quellen_Lesezeichen_Setzen "lcnet"
End Sub
Public Sub Matrizen_Quelle_LCPFN_Merken()
    Quellen_Lesezeichen_Setzen "lcpfn"
End Sub
Public Sub Matrizen_Quelle_Zeus_Merken()
    Quellen_Lesezeichen_Setzen "zeus"
End Sub

Private Sub Quellen_Lesezeichen_Setzen(ByVal key As String)
    Dim bookmarkName As String
    bookmarkName = "FM_Q_" & UCase$(key)
    If Len(Trim$(Selection.Range.Text)) = 0 Then
        MsgBox "Bitte zuerst einen echten Citavi-Nachweis wie [16] markieren.", vbExclamation
        Exit Sub
    End If
    If ActiveDocument.Bookmarks.Exists(bookmarkName) Then ActiveDocument.Bookmarks(bookmarkName).Delete
    ActiveDocument.Bookmarks.Add Name:=bookmarkName, Range:=Selection.Range
    MsgBox "Lesezeichen gesetzt: " & bookmarkName, vbInformation
End Sub

' ============================================================
' JSON/UTF-8: ausschliesslich die sechs Zahlenwerte ersetzen
' ============================================================
Private Function JSON_Quellenwert_Aendern(ByVal json As String, ByVal key As String, _
                                          ByVal number As String) As String
    Dim re As Object, m As Object, matches As Object, repl As String
    Set re = CreateObject("VBScript.RegExp")
    re.Pattern = "(""" & key & """\s*:\s*"")(\d+)("")"
    re.Global = True
    Set matches = re.Execute(json)
    If matches.Count <> 1 Then
        Err.Raise vbObjectError + 151, , "Quellenfeld fehlt/mehrdeutig in JSON: " & key
    End If
    Set m = matches(0)
    repl = CStr(m.SubMatches(0)) & number & CStr(m.SubMatches(2))
    JSON_Quellenwert_Aendern = Left$(json, m.FirstIndex) & repl & _
                              Mid$(json, m.FirstIndex + m.Length + 1)
End Function

Private Function Datei_UTF8_Lesen(ByVal pfad As String) As String
    Dim stream As Object
    Set stream = CreateObject("ADODB.Stream")
    stream.Type = 2
    stream.Charset = "utf-8"
    stream.Open
    stream.LoadFromFile pfad
    Datei_UTF8_Lesen = stream.ReadText(-1)
    stream.Close
End Function

Private Sub Datei_UTF8_Schreiben(ByVal pfad As String, ByVal inhalt As String)
    Dim stream As Object
    Set stream = CreateObject("ADODB.Stream")
    stream.Type = 2
    stream.Charset = "utf-8"
    stream.Open
    stream.WriteText inhalt
    stream.SaveToFile pfad, 2
    stream.Close
End Sub
