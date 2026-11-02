"""Standalone editable SVG for thesis sections 4.1–4.2, no GUI dependency.

Evidence: rancherConfigs/slurm-stack.yaml; node assignment is configuration,
not a live cluster snapshot. Scope: training and measurement deployment.
"""
from pathlib import Path
from xml.etree import ElementTree as ET
from PIL import ImageFont

OUT = Path(__file__).resolve().parent
W, H = 1800, 1290
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
root = ET.Element(f'{{{NS}}}svg', {
    'width': str(W), 'height': str(H), 'viewBox': f'0 0 {W} {H}',
    'role': 'img', 'aria-labelledby': 'title description',
})
def el(tag, attrs=None, parent=None, text=None):
    node = ET.SubElement(root if parent is None else parent, f'{{{NS}}}{tag}',
                         {str(k): str(v) for k,v in (attrs or {}).items()})
    if text is not None: node.text = text
    return node
el('title', {'id': 'title'}, text='Technische Bereitstellung der MLOps- und Messumgebung')
el('desc', {'id': 'description'}, text=(
    'Deploymentansicht des Namespace mlops-energy. benchmark-orchestrator, '
    'slurmctld und slurmd sind auf malea-srv01 festgelegt. MLflow, Pushgateway, '
    'Prometheus und Grafana werden ohne festen nodeSelector eingeplant; '
    'Prometheus schließt acds-cpu-01 aus. Der Trainingsprozess mit Controller '
    'liegt im slurmd-Container, der Node Exporter in einem weiteren Container '
    'desselben Pods. Persistente Volumes werden über S1 bis S4 zugeordnet.'))
C = {'ink':'#173449', 'muted':'#4E6576', 'line':'#ADBDC8',
     'soft':'#F2F6FA', 'teal':'#087F80', 'mint':'#F0F7F5', 'white':'#FFFFFF'}
defs=el('defs')
marker=el('marker', {'id':'arrow', 'viewBox':'0 0 10 10', 'refX':'9',
    'refY':'5', 'markerWidth':'8', 'markerHeight':'8', 'orient':'auto'}, defs)
el('path', {'d':'M 0 0 L 10 5 L 0 10 z', 'fill':C['ink']},marker)
fontbase=Path('C:/Windows/Fonts')
issues=[]
def label(x,y,text,size=20,bold=False,color='ink',width=None,anchor='start'):
    face=ImageFont.truetype(str(fontbase/('arialbd.ttf' if bold else 'arial.ttf')),size)
    measured=face.getlength(text)
    if width is not None and measured>width:
        issues.append((text, round(measured), width))
    return el('text', {'x':x,'y':y,'font-family':'Arial, Helvetica, sans-serif',
        'font-size':size,'font-weight':'700' if bold else '400',
        'fill':C.get(color,color),'text-anchor':anchor}, text=text)
def box(x,y,w,h,fill='white',stroke='line',radius=12,dash=None,sw=1.7):
    attrs={'x':x,'y':y,'width':w,'height':h,'rx':radius,
           'fill':C.get(fill,fill),'stroke':C.get(stroke,stroke),'stroke-width':sw}
    if dash: attrs['stroke-dasharray']=dash
    return el('rect', attrs)
def line(points,arrow=False,color='ink',dash=None,sw=2.3):
    attrs={'points':' '.join(f'{x},{y}' for x,y in points),'fill':'none',
           'stroke':C.get(color,color),'stroke-width':sw,'stroke-linejoin':'round'}
    if arrow: attrs['marker-end']='url(#arrow)'
    if dash: attrs['stroke-dasharray']=dash
    return el('polyline',attrs)
def pill(x,y,w,text):
    box(x,y,w,31,'mint','#A4CFC7',radius=7,sw=1)
    label(x+w/2,y+21,text,17,True,'teal',w-12,'middle')

box(0,0,W,H,'white','white',0)
label(40,58,'Technische Bereitstellung der MLOps- und Messumgebung',37,True,width=1720)
label(40,97,'Deploymentansicht · Knoten, Pods, Service-Endpunkte und Speicherzuordnung',22,color='muted',width=1720)

box(270,135,1490,1085,'white','line',16,sw=2)
label(300,176,'Kubernetes-Cluster  /  Namespace mlops-energy',27,True,width=1420)
label(300,207,'Verwaltung über Rancher · Bereitstellung über kubectl',20,color='muted',width=1420)

box(300,230,760,710,'soft','line',12)
label(325,270,'Knoten: malea-srv01',28,True,width=700)
label(325,302,'Feste Platzierung über nodeSelector',20,color='muted',width=700)
box(1080,230,650,710,'mint','line',12,dash='7 5')
label(1105,270,'Weitere Dienst-Pods',28,True,width=590)
label(1105,302,'Knotenwahl durch Kubernetes gemäß Platzierungsregeln',19,color='muted',width=600)

box(32,335,205,150,'white','line',12)
label(52,371,'Arbeitsplatz',25,True,width=165)
label(52,406,'CLI / Skripte',20,color='muted',width=165)
label(52,440,'Code + Konfiguration',16,color='muted',width=165)
line([(237,395),(325,395)],True)
label(280,377,'kubectl',16,color='muted',anchor='middle')

box(325,335,360,150)
label(345,362,'Pod',17,True,'muted',width=315)
label(345,395,'benchmark-orchestrator',25,True,width=315)
label(345,425,'Serverseitige Versuchssteuerung',19,color='muted',width=315)
label(345,460,'S1 RW · S2 RO · S4 RW',19,color='teal',width=315)
box(745,335,290,150)
label(765,362,'Pod',17,True,'muted',width=250)
label(765,395,'slurmctld',25,True,width=250)
label(765,427,'Service: slurmctld:6817',19,color='muted',width=250)
label(765,458,'Spool: emptyDir',18,color='muted',width=250)
line([(685,398),(745,398)],True)
label(715,380,'sbatch',16,color='muted',anchor='middle')
line([(890,485),(890,520),(650,520),(650,555)],True)
label(765,510,'slurmd:6818',17,color='muted',anchor='middle')

box(325,555,710,350,'white','line',12,sw=2)
label(347,588,'Pod: slurmd',25,True,width=660)
box(347,613,450,228,'soft','line',9)
label(367,641,'Container: slurmd',22,True,width=410)
label(367,667,'GPU-Anforderung: nvidia.com/gpu = 1',18,color='muted',width=410)
box(369,686,406,130,'white','#B8C9D3',8)
label(389,716,'Trainingsjob · Python-Prozess',22,True,width=365)
label(389,747,'Training und Validierung',19,color='muted',width=365)
pill(389,768,365,'ES-Controller als Python-Modul')
box(820,613,190,228,'white','line',9)
label(838,641,'Container',17,True,'muted',width=154)
label(838,674,'Node Exporter',20,True,width=154)
label(838,707,'Textfile Collector',17,color='muted',width=154)
label(838,738,'Port 9100',19,color='muted',width=154)
pill(838,768,154,'S3 RO')
label(347,874,'slurmd: S1 RW · S2 RW · S3 RW',19,color='teal',width=655)
label(347,927,'Service des Exporters: slurmd-node-exporter:9100',18,color='muted',width=680)

def pod(x,y,h,title,endpoint):
    box(x,y,600,h)
    label(x+20,y+34,'Pod: '+title,25,True,width=300)
    label(x+580,y+34,endpoint,19,color='muted',anchor='end')
pod(1105,330,135,'MLflow','mlflow:5000')
label(1125,398,'mlflow-backend-pvc  →  /mlflow',19,color='teal',width=550)
label(1125,429,'mlflow-artifacts-pvc  →  /mlruns',19,color='teal',width=550)
pod(1105,482,105,'Pushgateway','pushgateway:9091')
label(1125,554,'pushgateway-data-pvc  →  /data',19,color='teal',width=550)
pod(1105,604,135,'Prometheus','prometheus:9090')
label(1125,672,'prometheus-data-pvc  →  /prometheus',19,color='teal',width=550)
label(1125,706,'Node-Affinity: Knoten ≠ acds-cpu-01',19,color='muted',width=550)
pod(1105,756,150,'Grafana','grafana:3000')
label(1125,824,'ConfigMaps: Datenquelle und Dashboards',19,color='teal',width=550)
label(1125,855,'emptyDir: /var/lib/grafana',19,color='muted',width=550)
label(1125,884,'Laufzeitdaten temporär; Provisioning aus Konfiguration',17,color='muted',width=550)

label(325,987,'PERSISTENTE SPEICHER DER AUSFÜHRUNGSKOMPONENTEN',20,True,'muted',width=1390)
volumes=[
 (325,'S1','Arbeitsbereich','slurmd-workspace-cache-pvc','/workspace-cache','Programmstand, Daten, Ergebnisse'),
 (675,'S2','Jobprotokolle','slurm-logs-pvc','/workspace/logs','SLURM-Ausgaben und Fehlerlogs'),
 (1025,'S3','Messdateien','slurm-energy-metrics-pvc','/workspace/energy_metrics','Rohwerte, Zusammenfassungen, .prom'),
 (1375,'S4','Ergebnisarchiv','thesis-results-archive-pvc','/thesis-archive','Gesicherte Experimentstände'),
]
for x,key,title,pvc,mount,desc in volumes:
    box(x,1010,330,151,'white','#9EC3BC',9)
    pill(x+16,1024,44,key)
    label(x+73,1047,title,22,True,width=240)
    label(x+16,1080,pvc,17,color='teal',width=298)
    label(x+16,1111,mount,18,width=298)
    label(x+16,1142,desc,16,color='muted',width=298)
label(325,1196,'PVC = PersistentVolumeClaim    RW = lesend / schreibend    RO = nur lesend    emptyDir = temporärer Pod-Speicher',18,color='muted',width=1390)

line([(40,1240),(1760,1240)],color='line',sw=1)
label(40,1271,'Eigene Darstellung gemäß Deploymentmanifest · Auszug: Trainings- und Messumgebung · Knotenwahl gemäß Konfiguration',18,color='muted',width=1715)
if issues: raise ValueError(f'Text exceeds allocated width: {issues}')
ET.indent(root,space='  ')
target=OUT/'Technische_Bereitstellung_4_1_neu.svg'
ET.ElementTree(root).write(target,encoding='utf-8',xml_declaration=True)
print(target.resolve())
