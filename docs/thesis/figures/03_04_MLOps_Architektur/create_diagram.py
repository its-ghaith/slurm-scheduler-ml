"""Create an editable yEd diagram and a geometrically matching SVG preview."""
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree, register_namespace
from html import escape
import json

OUT = Path(__file__).parent
G = 'http://graphml.graphdrawing.org/xmlns'
Y = 'http://www.yworks.com/xml/graphml'
register_namespace('', G)
register_namespace('y', Y)
register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
def el(parent, name, attrs=None, text=None, ns=Y):
    e = SubElement(parent, '{'+ns+'}'+name, {k:str(v) for k,v in (attrs or {}).items()})
    if text is not None: e.text = text
    return e

NAVY='#18354B'; TEAL='#16756C'; GRAY='#50616D'; PALE='#F3F6F8'; LIGHT='#F0F8F6'
nodes=[]; edges=[]
def node(id, x,y,w,h,label,parent=None,fill='#FFFFFF',border=NAVY,size=22,bold=False,group=False,note=False):
    n=dict(id=id,x=x,y=y,w=w,h=h,label=label,parent=parent,fill=fill,border=border,size=size,bold=bold,group=group,note=note)
    nodes.append(n); return n
def edge(id,source,target,points,label='',labelbox=None,data=False):
    edges.append(dict(id=id,source=source,target=target,points=points,label=label,labelbox=labelbox,data=data))

node('title',40,18,1680,54,'Konzeptionelle Architektur der MLOps-Experimentierumgebung',size=30,bold=True,note=True)
node('subtitle',40,79,1680,32,'Auftragssteuerung, Trainingsausführung und Ergebnisdaten',size=19,note=True)
node('onprem',280,140,1440,990,'On-Premise-Umgebung',fill='#FAFBFC',border='#A3AFB7',size=23,bold=True,group=True)
node('kubernetes',305,195,1390,775,'Kubernetes · Betrieb der containerisierten Dienste und Worker',parent='onprem',fill='#F2F5F8',border='#7891A4',size=22,bold=True,group=True)
node('local',35,270,220,104,'Lokaler Rechner\nProgrammstand und\nVersuchskonfiguration',size=21)
node('orchestrator',355,275,260,100,'Versuchssteuerung\nAufträge und Status',parent='kubernetes',size=22)
node('slurm',710,275,340,100,'SLURM-Jobplanung\nRessourcen und Warteschlange',parent='kubernetes',size=21)
node('worker',640,430,500,495,'SLURM-Worker · bereitgestellte Ressourcen',parent='kubernetes',fill='#E7EFF4',border='#7891A4',size=20,bold=True,group=True)
node('job',660,478,460,425,'Temporärer Trainingsjob',parent='worker',fill='#FFFFFF',border='#7891A4',size=21,bold=True,group=True)
node('training',765,540,320,76,'Training und Validierung',parent='job',size=22)
node('observation',765,675,320,78,'Beobachtung und Messung\nQualität · Energie · Zeit',parent='job',fill=LIGHT,border=TEAL,size=21)
node('controller',765,815,320,64,'Early-Stopping-Controller',parent='job',fill=LIGHT,border=TEAL,size=21)
node('mlflow',1250,275,350,100,'MLflow\nRun-Daten und Modellartefakte',parent='kubernetes',size=22)
node('handoff',1220,525,390,100,'Monitoring-Übergabe\nExporter / Pushgateway',parent='kubernetes',size=23)
node('prometheus',1250,710,350,80,'Prometheus\nMonitoring-Zeitreihen',parent='kubernetes',size=22)
node('grafana',1250,855,350,80,'Grafana\nAbfragen und Visualisierung',parent='kubernetes',size=22)
node('storage',355,1000,1255,100,'Persistente Datenhaltung und Archivierung\nProgrammstände · Konfigurationen · Messaufzeichnungen · Controllerentscheidungen\nRun-Daten · Modellartefakte · Monitoring-Zeitreihen · Auswertungen',parent='onprem',fill=LIGHT,border=TEAL,size=21,bold=False)
node('ids',40,1145,1680,30,'Zuordnung: Versuchskennung · SLURM-Job-ID · MLflow-Run-ID; bei Live-Shadow zusätzlich Controllerkennung.',size=19,note=True)
node('shadow',40,1180,1680,30,'Live-Shadow: Stop-Empfehlung wird dokumentiert; das Referenztraining wird fortgesetzt.',size=19,note=True)
node('legend',40,1218,1680,30,'Durchgezogen: Steuerung / Ausführung    |    Gestrichelt: Datenfluss (Pfeilrichtung = Datenrichtung)    |    Logische Gruppierung',size=18,note=True)
node('source',40,1250,1680,28,'Quelle: Eigene Darstellung der in Abschnitt 3.4 beschriebenen Architektur.',size=18,note=True)

edge('submit','local','orchestrator',[(255,325),(355,325)],'Übergabe',(258,278,94,30))
edge('queue','orchestrator','slurm',[(615,325),(710,325)],'reicht ein',(619,278,87,30))
edge('run','slurm','worker',[(880,375),(880,430)],'startet Job',(887,391,120,30))
edge('observe','training','observation',[(925,616),(925,675)],'erfasst',(937,628,94,30),True)
edge('input','observation','controller',[(925,753),(925,815)],'aktuelle Beobachtungen',(770,774,307,28),True)
edge('feedback','controller','training',[(765,847),(704,847),(704,578),(765,578)],'Fortsetzen /\nStop-Empfehlung',(665,624,94,48))
edge('track','job','mlflow',[(1120,555),(1175,555),(1175,325),(1250,325)],'Konfiguration, Qualität\nund Artefakte',(1190,395,300,52),True)
edge('publish','job','handoff',[(1120,605),(1160,605),(1160,575),(1220,575)],'Messwerte und\nJobergebnisse',(1155,646,240,48),True)
edge('scrape','handoff','prometheus',[(1425,625),(1425,710)],'Abruf durch Prometheus',(1437,657,248,30),True)
edge('query','prometheus','grafana',[(1425,790),(1425,855)],'Abfragen durch Grafana',(1437,808,248,30),True)
edge('raw','job','storage',[(925,903),(925,1000)],'Rohdaten und Entscheidungen',(940,945,325,30),True)
edge('manifest','orchestrator','storage',[(485,375),(485,1000)],'Programmstand und\nVersuchskonfiguration',(365,640,240,54),True)
edge('persist_ml','mlflow','storage',[(1600,325),(1665,325),(1665,980),(1500,980),(1500,1000)],'',None,True)
edge('persist_prom','prometheus','storage',[(1600,750),(1640,750),(1640,950),(1545,950),(1545,1000)],'',None,True)

lookup={n['id']:n for n in nodes}
root=Element('{'+G+'}graphml',{'{http://www.w3.org/2001/XMLSchema-instance}schemaLocation':G+' http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd'})
el(root,'key',{'id':'d0','for':'node','yfiles.type':'nodegraphics'},ns=G)
el(root,'key',{'id':'d1','for':'edge','yfiles.type':'edgegraphics'},ns=G)
el(root,'key',{'id':'d2','for':'node','attr.name':'description','attr.type':'string'},ns=G)
graph=el(root,'graph',{'id':'G','edgedefault':'directed'},ns=G)
graphs={None:graph}
for n in nodes:
    a={'id':n['id']}
    if n['group']: a['yfiles.foldertype']='group'
    ne=el(graphs[n['parent']],'node',a,ns=G)
    el(ne,'data',{'key':'d2'},n['label'],ns=G)
    data=el(ne,'data',{'key':'d0'},ns=G)
    if n['group']:
        proxy=el(data,'ProxyAutoBoundsNode'); real=el(proxy,'Realizers',{'active':'0'}); sh=el(real,'GroupNode')
    else: sh=el(data,'ShapeNode')
    el(sh,'Geometry',{'x':n['x'],'y':n['y'],'width':n['w'],'height':n['h']})
    el(sh,'Fill',{'color':n['fill'],'color2':n['fill'],'transparent':str(n['note']).lower()})
    el(sh,'BorderStyle',{'color':n['border'],'type':'line','width':1.5,'hasColor':str(not n['note']).lower()})
    el(sh,'NodeLabel',{'alignment':'left' if n['note'] else 'center','autoSizePolicy':'content','fontFamily':'Arial','fontSize':n['size'],'fontStyle':'bold' if n['bold'] else 'plain','hasBackgroundColor':'false','hasLineColor':'false','modelName':'internal','modelPosition':'tl' if n['note'] else 't' if n['group'] else 'c','textColor':NAVY,'visible':'true'},n['label'])
    el(sh,'Shape',{'type':'rectangle' if n['note'] else 'roundrectangle'})
    if n['group']:
        el(sh,'State',{'closed':'false','closedHeight':60,'closedWidth':220,'innerGraphDisplayEnabled':'false','autoResize':'true'})
        # yEd computes open-group bounds from child bounds plus header and insets.
        l,t,r,b={'job':(105,34,35,24),'worker':(20,22,20,22),'kubernetes':(50,52,85,35),'onprem':(25,25,25,30)}[n['id']]
        el(sh,'Insets',{'bottom':b,'bottomF':b,'left':l,'leftF':l,'right':r,'rightF':r,'top':t,'topF':t})
        el(sh,'BorderInsets',{'bottom':0,'bottomF':0,'left':0,'leftF':0,'right':0,'rightF':0,'top':0,'topF':0})
        graphs[n['id']]=el(ne,'graph',{'id':n['id']+':','edgedefault':'directed'},ns=G)

for e in edges:
    ee=el(graph,'edge',{'id':e['id'],'source':e['source'],'target':e['target']},ns=G)
    data=el(ee,'data',{'key':'d1'},ns=G); line=el(data,'PolyLineEdge')
    s=lookup[e['source']];t=lookup[e['target']]; first=e['points'][0];last=e['points'][-1]
    path=el(line,'Path',{'sx':first[0]-s['x']-s['w']/2,'sy':first[1]-s['y']-s['h']/2,'tx':last[0]-t['x']-t['w']/2,'ty':last[1]-t['y']-t['h']/2})
    for x,y in e['points'][1:-1]:el(path,'Point',{'x':x,'y':y})
    color=TEAL if e['data'] else NAVY
    el(line,'LineStyle',{'color':color,'type':'dashed' if e['data'] else 'line','width':2})
    el(line,'Arrows',{'source':'none','target':'standard'})
    if e['label']:
        x,y,w,h=e['labelbox']
        el(line,'EdgeLabel',{'alignment':'center','backgroundColor':'#FFFFFF','fontFamily':'Arial','fontSize':17 if e['id']!='feedback' else 15,'fontStyle':'plain','hasBackgroundColor':'true','hasLineColor':'false','height':h,'width':w,'x':x-first[0],'y':y-first[1],'modelName':'free','modelPosition':'anywhere','preferredPlacement':'anywhere','ratio':0.5,'textColor':color,'visible':'true'},e['label'])
    el(line,'BendStyle',{'smoothed':'false'})
ElementTree(root).write(OUT/'MLOps_Architektur_3_4.graphml',encoding='utf-8',xml_declaration=True)

svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1760" height="1295" viewBox="0 0 1760 1295">',
'<defs><marker id="control" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="'+NAVY+'"/></marker><marker id="data" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="'+TEAL+'"/></marker></defs>',
'<rect width="1760" height="1295" fill="white"/>']
def textblock(label,x,y,w,h,fs,color=NAVY,bold=False,left=False,top=False):
    lines=label.split('\n'); leading=fs*1.28
    yy=y+fs+7 if top else y+h/2-(len(lines)-1)*leading/2+fs*.34
    xx=x+6 if left else x+w/2
    svg.append(f'<text x="{xx}" y="{yy}" fill="{color}" font-family="Arial, sans-serif" font-size="{fs}" font-weight="{"bold" if bold else "normal"}" text-anchor="{"start" if left else "middle"}">')
    for i,l in enumerate(lines):svg.append(f'<tspan x="{xx}" dy="{0 if i==0 else leading}">{escape(l)}</tspan>')
    svg.append('</text>')
for n in nodes:
    if n['group']:
        svg.append(f'<rect x="{n["x"]}" y="{n["y"]}" width="{n["w"]}" height="{n["h"]}" rx="9" fill="{n["fill"]}" stroke="{n["border"]}" stroke-width="1.5"/>')
        textblock(n['label'],n['x'],n['y'],n['w'],45,n['size'],bold=n['bold'],top=True)
for e in edges:
    color=TEAL if e['data'] else NAVY
    path='M'+' L'.join(f'{x} {y}' for x,y in e['points'])
    svg.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{"7 5" if e["data"] else "none"}" marker-end="url(#{"data" if e["data"] else "control"})"/>')
for n in nodes:
    if n['group']:continue
    if not n['note']:svg.append(f'<rect x="{n["x"]}" y="{n["y"]}" width="{n["w"]}" height="{n["h"]}" rx="7" fill="{n["fill"]}" stroke="{n["border"]}" stroke-width="1.5"/>')
    textblock(n['label'],n['x'],n['y'],n['w'],n['h'],n['size'],bold=n['bold'],left=n['note'])
for e in edges:
    if e['label']:
        x,y,w,h=e['labelbox'];svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="white"/>')
        textblock(e['label'],x,y,w,h,15 if e['id']=='feedback' else 17,TEAL if e['data'] else NAVY)
svg.append('</svg>')
(OUT/'MLOps_Architektur_3_4.svg').write_text('\n'.join(svg),encoding='utf-8')
(OUT/'diagram_model.json').write_text(json.dumps({'nodes':nodes,'edges':edges},ensure_ascii=False,indent=2),encoding='utf-8')
print('Created editable GraphML and SVG:',OUT)
