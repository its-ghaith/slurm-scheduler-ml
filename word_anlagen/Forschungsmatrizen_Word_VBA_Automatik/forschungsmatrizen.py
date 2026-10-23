#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Originalgetreue, dynamische Illustrator-SVG-Matrizen.

Die hochgeladene, manuell bearbeitete SVG ist die echte Vorlage. Die vier
Quadrantenköpfe bleiben JE EIN <text>-Objekt. Titel und Beschreibung teilen
sich dasselbe Textobjekt und werden über verschachtelte <tspan> formatiert.

Zeilenregel: jeweils Titel 1 Zeile + Beschreibung 1 Zeile;
nur Matrix 1 unten rechts Titel 2 Zeilen + Beschreibung 1 Zeile.

SVG-Erzeugung benötigt nur die Python-Standardbibliothek.
PNG wird optional mit CairoSVG erzeugt, wenn installiert.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
S = lambda t: '{%s}%s' % (NS, t)
KEYS = ('oben_links', 'oben_rechts', 'unten_links', 'unten_rechts')
SOURCE_NAMES = {
    'Domhan et al.': 'domhan',
    'LCNet': 'lcnet',
    'LC-PFN': 'lcpfn',
    'Reguero et al.': 'reguero',
    'Klassische Kriterien nach Prechelt': 'prechelt',
    'Zeus': 'zeus',
}
HEADER_ORIGINS = ((196.45, 107.67), (718.38, 107.67), (250.78, 422.91), (723.72, 422.92))
CENTERS = ((368.1, 107.67), (869.4, 107.67), (368.1, 422.91), (869.4, 422.91))


def content(el: ET.Element) -> str:
    return ''.join(el.itertext()).strip()


def change_element_text(el: ET.Element, value: str) -> None:
    # Standardbeschriftung in Illustrator hat genau ein inneres <tspan>.
    ts = el.findall('.//' + S('tspan'))
    if ts:
        ts[-1].text = value
    else:
        el.text = value


def validate_and_replace_header(el: ET.Element, cfg: dict, key: str, matrix_key: str,
                                original: bool) -> None:
    title = cfg['titel']
    subtitle = cfg['beschreibung']
    titles = title.split('\n')
    subtitles = subtitle.split('\n')
    expected = 2 if matrix_key == 'matrix1' and key == 'unten_rechts' else 1
    if len(titles) != expected or len(subtitles) != 1:
        raise ValueError(f'{matrix_key}/{key}: erwartet {expected} Titelzeile(n) und genau eine Beschreibungszeile; gefunden {len(titles)} und {len(subtitles)}')
    if not original:
        # Matrix 2: EIN SVG-Textobjekt mit zentriertem Anker und tspans.
        el.set('text-anchor', 'middle')
        el.set('transform', 'translate(%.2f %.2f)' % CENTERS[KEYS.index(key)])
        el[:] = []
        for i, line in enumerate(titles):
            parent = ET.SubElement(el, S('tspan'), {'class': 'st14'})
            ET.SubElement(parent, S('tspan'), {'x': '0', 'y': '0' if i == 0 else '22.8'}).text = line
        parent = ET.SubElement(el, S('tspan'), {'class': 'st16'})
        ET.SubElement(parent, S('tspan'), {'x': '0', 'y': '17.76' if len(titles) == 1 else '40.56'}).text = subtitle
    else:
        # Matrix 1: exakte vom Nutzer gespeicherte Illustrator-Geometrie behalten.
        old_title = ''.join([content(v) for v in el if v.get('class') == 'st14'])
        old_subtitle = ''.join([content(v) for v in el if v.get('class') == 'st16'])
        if old_title != title.replace('\n','') and old_title.strip() != title.replace('\n',' ').strip():
            raise ValueError(f'Für Matrix 1 wurde die Originalüberschrift {key} verändert. Bitte das SVG-Beispiel direkt bearbeiten, damit sein Original-Layout erhalten bleibt.')
        if old_subtitle != subtitle:
            raise ValueError(f'Die Originalbeschreibung für Matrix 1 im Feld {key} stimmt nicht mit matrizen.json überein. Bitte das SVG-Beispiel oder die JSON anpassen.')


def matrix1_refs(root: ET.Element, conf: dict) -> None:
    refs = conf['quellen']
    for el in root.iter(S('text')):
        value = content(el)
        for prefix, key in SOURCE_NAMES.items():
            if re.fullmatch(re.escape(prefix) + r' \[\d+\]', value):
                change_element_text(el, '%s [%s]' % (prefix, refs[key]))
                break


def parent_index(root):
    return {c:p for p in root.iter() for c in p}


def replace_matrix2_items(root: ET.Element, conf: dict) -> None:
    parents = parent_index(root)
    # Alte Einträge und blaue Kreise aus der Illustrator-Kopie entfernen.
    old = list(root.iter(S('text')))
    for el in old[:9]:
        parents[el].remove(el)
    parents = parent_index(root)
    for g in list(root.iter(S('g'))):
        if re.match(r'PathCollection_\d+$', g.get('id', '')):
            parents[g].remove(g)
    # Einträge werden als eigenständige Textobjekte wie in der Vorlage gesetzt.
    # Die vier Quadrantenköpfe bleiben dabei jeweils EIN Objekt.
    target = next(g for g in root.iter(S('g')) if g.get('id') == 'axes_1')
    # Koordinaten des originalen Illustrator-Dokuments (1180.8 × 799.2).
    placements = {
      'oben_links': [(166.32, 235.00), (166.32, 279.00)],
      'oben_rechts': [(665.00, 240.54)],
      'unten_links': [(166.32, 555.42)],
      'unten_rechts': [(665.00, 525.00), (665.00, 585.00)]
    }
    for quadrant, coords in placements.items():
        works = conf['matrix2']['quadranten'][quadrant]['arbeiten']
        if len(coords) != len(works):
            raise ValueError('Falsche Anzahl von Arbeiten für Matrix 2: ' + quadrant)
        for work, (cx, baseline) in zip(works, coords):
            # Punktmittelpunkt 4.6 Pixel oberhalb der Textgrundlinie.
            ET.SubElement(target, S('circle'), {
                'class':'st2', 'cx':str(cx), 'cy':str(baseline - 4.6), 'r':'5.7'})
            main = ET.SubElement(target, S('text'), {'class':'st13', 'transform':'translate(%.2f %.2f)' % (cx + 30.24, baseline)})
            ET.SubElement(main, S('tspan'), {'x':'0','y':'0'}).text = '%s [%s]' % (work['text'], conf['quellen'][work['quelle']])
            detail = work.get('detail','')
            if detail:
                for i,line in enumerate(detail.split('\n')):
                    det = ET.SubElement(target,S('text'), {'class':'st8', 'transform':'translate(%.2f %.2f)' % (cx + 30.24,baseline + 22.03 + i * 13.4)})
                    ET.SubElement(det, S('tspan'), {'x':'0','y':'0'}).text = line


def update_axis_matrix2(root: ET.Element, conf: dict) -> None:
    d = conf['matrix2']
    texts = list(root.iter(S('text')))
    # Beruht auf der Reihenfolge der kopierten Nutzer-Vorlage nach Entfernen der
    # 9 Arbeiten-Textobjekte: 0 y-Achse, 1..2 oben, 3..5 unten, 6..8 X-Achse.
    # Wir verwenden exakte Originalnamen zur sicheren Zuordnung.
    mapping = {
      'Prognose zukünftiger Modellqualität': d['y_achse'],
      'Explizite': 'Prädiktive',
      'Keine': 'Keine',
      'explizite': 'prädiktive',
      'Keine explizite Ziel- oder Bewertungsgröße': d['x_links'],
      'Explizite Ziel- oder Bewertungsgröße': d['x_rechts'],
      'Energie als explizite Ziel- oder Bewertungsgröße': d['x_achse'],
    }
    for t in texts:
        s = content(t)
        if s in mapping:
            change_element_text(t,mapping[s])
    # Oberer 2. Zeilentext, unten 3. Zeilentext; keine Quadrantenköpfe betroffen.
    axes = [t for t in texts if t.get('class') == 'st12']
    assert len(axes) == 5
    change_element_text(axes[1], 'Verteilung')
    change_element_text(axes[4], 'Verteilung')
    # Kurze Y-Endpunktbeschriftungen rechtsbündig VOR der vertikalen Achse.
    # Das vermeidet das Überlappen der Achse bei den neuen Matrix-2-Labels.
    for marker in axes:
        match = re.search(r'translate\([^ ]+ ([^)]+)\)', marker.get('transform',''))
        if match:
            marker.set('transform', 'translate(108.00 %s)' % match.group(1))
            marker.set('text-anchor', 'end')
    # X-Achsen: linker Text bleibt linksbündig. Rechtsbündiger Text wird am
    # rechten Rand verankert statt bei dessen ursprünglichem Startpunkt.
    for t in texts:
        if content(t) == d['x_rechts']:
            t.set('transform','translate(1134.70 710.49)')
            t.set('text-anchor','end')
        elif content(t) == d['x_achse']:
            t.set('transform','translate(630.00 762.10)')
            t.set('text-anchor','middle')


def render(conf: dict, key: str, out_dir: Path, tpl: Path):
    tree = ET.parse(tpl)
    root = tree.getroot()
    headers = [el for el in root.iter(S('text')) if el.get('class') == 'st11']
    assert len(headers) == 4, 'Vorlage enthält nicht genau vier SVG-Textobjekte für Quadrantenköpfe'
    for name, node in zip(KEYS, headers):
        validate_and_replace_header(node, conf[key]['quadranten'][name], name, key, key=='matrix1')
    if key=='matrix1':
        matrix1_refs(root, conf)
    else:
        replace_matrix2_items(root,conf)
        update_axis_matrix2(root,conf)
    dest = out_dir / (conf[key]['dateiname'] + '.svg')
    tree.write(dest, encoding='utf-8', xml_declaration=True)
    # Optionales PNG: die VBA-Automation verwendet SVG, dafür ist kein Cairo nötig.
    try:
        import cairosvg
        cairosvg.svg2png(url=str(dest),write_to=str(dest.with_suffix('.png')),output_width=2362)
    except (ImportError, OSError) as err:
        print('PNG optional übersprungen:',str(err))
    # Strukturelle Überprüfung
    v = ET.parse(dest).getroot()
    hd=[e for e in v.iter(S('text')) if e.get('class')=='st11']
    expected=(2,2,2,3) if key=='matrix1' else (2,2,2,2)
    actual=tuple(sum(1 for _ in e.findall('.//'+S('tspan')) if _.text and _.get('class') is None) for e in hd)
    if len(hd)!=4 or actual!=expected:
        raise AssertionError('Textobjekte / Zeilen nicht korrekt: '+str(actual))
    print(key, 'erstellt:',dest.name,'Textzeilen:',actual)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('matrizen.json'))
    parser.add_argument('--ausgabe',type=Path,default=None)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    base=args.config.parent
    out=args.ausgabe or base
    out.mkdir(parents=True,exist_ok=True)
    template=Path(__file__).with_name('Vorlage_Matrix_1_Illustrator.svg')
    if not template.exists():
        raise FileNotFoundError('Fehlt: Vorlage_Matrix_1_Illustrator.svg im Skriptordner')
    for k in ('matrix1','matrix2'):
        render(config,k,out,template)

if __name__=='__main__':
    main()
