from __future__ import annotations

import argparse
import json
from pathlib import Path


DS = {"type": "prometheus", "uid": "prometheus"}


def target(ref: str, expression: str, legend: str) -> dict:
    return {
        "refId": ref,
        "expr": expression,
        "legendFormat": legend,
        "instant": True,
        "range": False,
        "format": "table",
        "editorMode": "code",
        "datasource": DS,
    }


def echarts_panel(
    panel_id: int,
    title: str,
    description: str,
    targets: list[dict],
    script: str,
    y: int,
    height: int = 14,
) -> dict:
    return {
        "id": panel_id,
        "title": title,
        "description": description,
        "type": "volkovlabs-echarts-panel",
        "gridPos": {"x": 0, "y": y, "w": 24, "h": height},
        "datasource": DS,
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {"renderer": "svg", "followTheme": True, "getOption": script},
        "targets": targets,
    }


ENERGY_QUALITY_SCRIPT = r"""
const frames = context?.panel?.data?.series || [];
const vals = f => { const v=f?.values; if(!v)return []; if(typeof v.toArray==='function')return v.toArray(); if(v.buffer)return Array.from(v.buffer); return Array.isArray(v)?v:[]; };
const nf = f => (f.fields||[]).find(x => x.type==='number' || x.name==='Value');
const value = f => { const a=vals(nf(f)); return a.length ? Number(a[a.length-1]) : null; };
const labels = f => nf(f)?.labels || {};
const byEpoch = new Map(); let rapecStop=null, esStop=null, dataset='Dataset';
frames.forEach(f => { const l=labels(f), v=value(f); if(!Number.isFinite(v))return; dataset=l.case_id||dataset;
  if(f.refId==='A'||f.refId==='B'){ const e=Number(l.epoch); if(!Number.isFinite(e))return; const p=byEpoch.get(e)||{epoch:e}; p[f.refId]=v; byEpoch.set(e,p); }
  if(f.refId==='C')rapecStop=v; if(f.refId==='D')esStop=v;
});
const rows=Array.from(byEpoch.values()).filter(p=>Number.isFinite(p.A)&&Number.isFinite(p.B)).sort((a,b)=>a.epoch-b.epoch);
let cumulative=0,best=-Infinity; const curve=rows.map(p=>{cumulative+=p.A; best=Math.max(best,p.B); return {value:[cumulative,best],epoch:p.epoch};});
const pointAt = stop => { const p=curve.find(x=>x.epoch===Number(stop)); return p ? {value:p.value,epoch:p.epoch} : null; };
const markers=[]; const rp=pointAt(rapecStop), ep=pointAt(esStop), fp=curve[curve.length-1];
if(rp)markers.push({...rp,name:'RAPEC-G v4',itemStyle:{color:'#2ca02c'}});
if(ep)markers.push({...ep,name:'Standard ES',itemStyle:{color:'#ff9800'}});
if(fp)markers.push({...fp,name:'Full100',itemStyle:{color:'#1976d2'}});
return {animation:false,tooltip:{trigger:'item',formatter:p=>`${p.seriesName||p.data.name}<br/>${dataset}<br/>Epoch: ${p.data.epoch}<br/>GPU energy: ${p.data.value[0].toFixed(3)} Wh<br/>Best quality: ${p.data.value[1].toFixed(2)} %`},
 legend:{bottom:0},grid:{left:75,right:35,top:45,bottom:65},xAxis:{type:'value',name:'Cumulative Full100 GPU energy (Wh)',nameLocation:'middle',nameGap:45,min:0},yAxis:{type:'value',name:'Best normalized quality (%)',nameLocation:'middle',nameGap:55,min:0,max:100},
 series:[{name:'Full100 trajectory',type:'line',showSymbol:false,data:curve,lineStyle:{width:3,color:'#616161'}},{name:'Stop points',type:'scatter',symbolSize:18,label:{show:true,position:'top',formatter:p=>`${p.data.name} (E${p.data.epoch})`},data:markers}]};
"""


ENERGY_PER_QUALITY_SCRIPT = r"""
const frames=context?.panel?.data?.series||[];
const vals=f=>{const v=f?.values;if(!v)return[];if(typeof v.toArray==='function')return v.toArray();if(v.buffer)return Array.from(v.buffer);return Array.isArray(v)?v:[];};
const nf=f=>(f.fields||[]).find(x=>x.type==='number'||x.name==='Value'); const value=f=>{const a=vals(nf(f));return a.length?Number(a[a.length-1]):null;}; const labels=f=>nf(f)?.labels||{};
const rows=new Map(); frames.forEach(f=>{const l=labels(f),v=value(f),c=l.case_id;if(!c||!Number.isFinite(v))return;const r=rows.get(c)||{case_id:c,task:l.task_type||'unknown'};r[f.refId]=v;rows.set(c,r);});
const data=Array.from(rows.values()).sort((a,b)=>a.case_id.localeCompare(b.case_id));
const whpp=(energy,q,q0)=>{const gain=(q-q0)*100;return Number.isFinite(energy)&&gain>1e-9?energy/gain:null;};
const series=[['RAPEC-G v4','C','D','#2ca02c'],['Standard ES','E','F','#ff9800'],['Full100','A','B','#1976d2']].map(([name,e,q,color])=>({name,type:'bar',itemStyle:{color},data:data.map(r=>{const v=whpp(r[e],r[q],r.G);return {value:v,case_id:r.case_id,energy:r[e],quality:r[q],start:r.G};})}));
return {animation:false,tooltip:{trigger:'item',formatter:p=>{const d=p.data;if(!Number.isFinite(d.value))return `${p.seriesName}<br/>${d.case_id}<br/>Undefined: no positive quality gain`;return `${p.seriesName}<br/>${d.case_id}<br/>Energy: ${d.energy.toFixed(3)} Wh<br/>Start quality: ${(d.start*100).toFixed(2)} %<br/>Stop quality: ${(d.quality*100).toFixed(2)} %<br/><b>${d.value.toFixed(3)} Wh/pp</b>`;}},legend:{bottom:0},grid:{left:75,right:25,top:35,bottom:120},xAxis:{type:'category',data:data.map(r=>r.case_id),axisLabel:{rotate:45}},yAxis:{type:'value',name:'GPU energy (Wh) per +1 quality pp',nameLocation:'middle',nameGap:58,min:0},series};
"""


def selected() -> str:
    return (
        'benchmark_run_id="$benchmark_run_id",controller_id=~"$controllers",'
        'task_type=~"$task_types",stage=~"$stages",case_id=~"$cases"'
    )


def stat(panel_id: int, title: str, expression: str, unit: str, x: int, y: int) -> dict:
    return {
        "id": panel_id,
        "title": title,
        "type": "stat",
        "gridPos": {"x": x, "y": y, "w": 4, "h": 5},
        "datasource": DS,
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "color": {"mode": "thresholds"},
                "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
            },
            "overrides": [],
        },
        "options": {
            "colorMode": "value",
            "graphMode": "none",
            "textMode": "value_and_name",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        },
        "targets": [target("A", expression, "{{controller_id}}")],
    }


def bars(
    panel_id: int,
    title: str,
    expressions: list[tuple[str, str, str]],
    unit: str,
    x: int,
    y: int,
    width: int = 12,
) -> dict:
    return {
        "id": panel_id,
        "title": title,
        "type": "barchart",
        "gridPos": {"x": x, "y": y, "w": width, "h": 10},
        "datasource": DS,
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "color": {"mode": "palette-classic"},
                "custom": {
                    "fillOpacity": 85,
                    "lineWidth": 1,
                    "stacking": {"group": "A", "mode": "none"},
                },
            },
            "overrides": [],
        },
        "options": {
            "xField": "case_id",
            "orientation": "vertical",
            "xTickLabelRotation": 45,
            "stacking": "none",
            "groupWidth": 0.75,
            "barWidth": 0.25,
            "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"},
        },
        "targets": [target(ref, expression, legend) for ref, expression, legend in expressions],
        "transformations": [
            {"id": "sortBy", "options": {"fields": {}, "sort": [{"field": "case_id", "desc": False}]}},
        ],
    }


def variable(
    name: str,
    label: str,
    query_text: str,
    *,
    multi: bool = True,
    default: str = "",
) -> dict:
    return {
        "name": name,
        "label": label,
        "type": "query",
        "datasource": DS,
        "query": {"query": query_text, "refId": f"var-{name}"},
        "definition": query_text,
        "refresh": 1,
        "multi": multi,
        "includeAll": multi,
        "allValue": ".*" if multi else None,
        "current": {
            "selected": True,
            "text": ["All"] if multi else default,
            "value": ["$__all"] if multi else default,
        },
        "options": [],
    }


def build(
    *,
    title: str = "Live Shadow Controllers - Nine Dataset Development",
    uid: str = "controller-live-shadow-nine-dataset",
    tags: list[str] | None = None,
    default_run_id: str = "",
    default_curve_case: str = "",
) -> dict:
    filters = selected()
    panels: list[dict] = [
        {
            "id": 1,
            "title": "Live Shadow Measurement Scope",
            "type": "text",
            "gridPos": {"x": 0, "y": 0, "w": 24, "h": 3},
            "options": {
                "mode": "markdown",
                "content": (
                    "One real Full100 trajectory per dataset (seed 0). Controllers decide live but cannot stop the shared training. "
                    "Counterfactual total = cumulative epoch GPU energy at the stop signal + separately attributed controller CPU energy. "
                    "Preprocessing and finalization are intentionally not claimed as complete lifecycle energy in this development benchmark."
                ),
            },
        },
        stat(2, "Mean Energy Saving", 'live_shadow_mean_energy_saving_fraction{benchmark_run_id="$benchmark_run_id",controller_id=~"$controllers"} * 100', "percent", 0, 3),
        stat(3, "Mean Quality Regret", 'live_shadow_mean_quality_regret{benchmark_run_id="$benchmark_run_id",controller_id=~"$controllers"} * 100', "percent", 4, 3),
        stat(4, "Energy Success", 'live_shadow_energy_success_rate{benchmark_run_id="$benchmark_run_id",controller_id=~"$controllers"} * 100', "percent", 8, 3),
        stat(5, "Quality Within 10 pp", 'live_shadow_quality_success_rate{benchmark_run_id="$benchmark_run_id",controller_id=~"$controllers"} * 100', "percent", 12, 3),
        stat(6, "Joint Success", 'live_shadow_joint_success_rate{benchmark_run_id="$benchmark_run_id",controller_id=~"$controllers"} * 100', "percent", 16, 3),
        stat(7, "Controller Overhead", 'live_shadow_total_controller_overhead_energy_wh{benchmark_run_id="$benchmark_run_id",controller_id=~"$controllers"}', "watth", 20, 3),
        bars(
            10,
            "Stop Epoch by Dataset and Controller",
            [("A", f"live_shadow_controller_stop_epoch{{{filters}}}", "{{controller_id}}")],
            "none",
            0,
            8,
            24,
        ),
        bars(
            11,
            "Full100 and Quality at Shadow Stop",
            [
                ("A", f"live_shadow_full100_best_quality{{{filters}}} * 100", "Full100 / {{controller_id}}"),
                ("B", f"live_shadow_best_quality_at_stop{{{filters}}} * 100", "Stop / {{controller_id}}"),
            ],
            "percent",
            0,
            18,
            12,
        ),
        bars(
            12,
            "Quality Regret by Dataset",
            [("A", f"live_shadow_quality_regret{{{filters}}} * 100", "{{controller_id}}")],
            "percent",
            12,
            18,
            12,
        ),
        bars(
            13,
            "Full100, Training-to-Stop, and Counterfactual Total Energy",
            [
                ("A", f"live_shadow_full100_training_energy_wh{{{filters}}}", "Full100 / {{controller_id}}"),
                ("B", f"live_shadow_training_energy_to_stop_wh{{{filters}}}", "Training to stop / {{controller_id}}"),
                ("C", f"live_shadow_counterfactual_total_energy_wh{{{filters}}}", "Training + controller / {{controller_id}}"),
            ],
            "watth",
            0,
            28,
            24,
        ),
        bars(
            14,
            "Controller Overhead Energy Only",
            [("A", f"live_shadow_controller_overhead_energy_wh{{{filters}}}", "{{controller_id}} / {{energy_measurement_method}}")],
            "watth",
            0,
            38,
            12,
        ),
        bars(
            15,
            "Controller Compute Time Only",
            [("A", f"live_shadow_controller_compute_seconds{{{filters}}}", "{{controller_id}}")],
            "s",
            12,
            38,
            12,
        ),
        bars(
            16,
            "Energy Saving by Dataset",
            [("A", f"live_shadow_energy_saving_fraction{{{filters}}} * 100", "{{controller_id}}")],
            "percent",
            0,
            48,
            24,
        ),
        echarts_panel(
            17,
            "Full100 Quality-Energy Curve with RAPEC-G and ES Stop Points",
            "Zeigt die reale Full100-Lernkurve des einzeln ausgewählten Datasets. X ist die kumulierte, gemessene GPU-Energie (Summe der Epoch-Energie in Wh), Y die bis dahin beste normalisierte Qualität. RAPEC-G v4 und Standard ES erscheinen ausschließlich als markierte, kontrafaktische Stopppunkte auf derselben realen Kurve; Full100 ist der Endpunkt. Es werden keine Controller-Kurven simuliert.",
            [
                target("A", 'thesis_live_shadow_epoch_gpu_energy_wh{benchmark_run_id="$benchmark_run_id",case_id="$curve_case"}', "Epoch GPU energy"),
                target("B", 'thesis_live_shadow_epoch_quality_score{benchmark_run_id="$benchmark_run_id",case_id="$curve_case"} * 100', "Normalized quality"),
                target("C", 'live_shadow_controller_stop_epoch{benchmark_run_id="$benchmark_run_id",case_id="$curve_case",controller_id="rapec-g-v4"}', "RAPEC-G v4 stop"),
                target("D", 'live_shadow_controller_stop_epoch{benchmark_run_id="$benchmark_run_id",case_id="$curve_case",controller_id="standard-es-10"}', "Standard ES stop"),
            ],
            ENERGY_QUALITY_SCRIPT,
            58,
        ),
        echarts_panel(
            18,
            "GPU Energy per +1 Normalized Quality Point by Dataset",
            "Vergleicht den durchschnittlichen GPU-Energieaufwand je zusätzlichem normalisierten Qualitätsprozentpunkt. Formel: Energie bis zum Stop inklusive Controller-Overhead / ((beste Qualität am Stop − Qualität in Epoche 1) × 100). Für Full100 wird die vollständige Trainingsenergie verwendet. Kleinere Werte sind günstiger; bei keinem positiven Qualitätsgewinn bleibt der Wert bewusst undefiniert.",
            [
                target("A", 'live_shadow_full100_training_energy_wh{benchmark_run_id="$benchmark_run_id",controller_id="rapec-g-v4",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases"}', "Full100 energy"),
                target("B", 'live_shadow_full100_best_quality{benchmark_run_id="$benchmark_run_id",controller_id="rapec-g-v4",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases"}', "Full100 quality"),
                target("C", 'live_shadow_counterfactual_total_energy_wh{benchmark_run_id="$benchmark_run_id",controller_id="rapec-g-v4",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases"}', "RAPEC-G energy"),
                target("D", 'live_shadow_best_quality_at_stop{benchmark_run_id="$benchmark_run_id",controller_id="rapec-g-v4",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases"}', "RAPEC-G quality"),
                target("E", 'live_shadow_counterfactual_total_energy_wh{benchmark_run_id="$benchmark_run_id",controller_id="standard-es-10",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases"}', "Standard ES energy"),
                target("F", 'live_shadow_best_quality_at_stop{benchmark_run_id="$benchmark_run_id",controller_id="standard-es-10",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases"}', "Standard ES quality"),
                target("G", 'thesis_live_shadow_epoch_quality_score{benchmark_run_id="$benchmark_run_id",epoch="1",case_id=~"$cases",task_type=~"$task_types",stage=~"$stages"}', "Initial quality"),
            ],
            ENERGY_PER_QUALITY_SCRIPT,
            72,
        ),
    ]
    descriptions = {
        2: "Makro-Mittel der relativen Energieeinsparung über die ausgewählten Datasets. Formel je Dataset: (Full100-Energie − kontrafaktische Stop-Energie) / Full100-Energie × 100.",
        3: "Makro-Mittel des normalisierten Qualitätsverlusts gegenüber Full100 über die ausgewählten Datasets. Positive Werte bedeuten geringere Qualität am Stopppunkt.",
        4: "Anteil der ausgewählten Dataset-Controller-Paare, welche das definierte Energieeinsparungsziel erreichen.",
        5: "Anteil der ausgewählten Dataset-Controller-Paare, deren normalisierter Qualitätsverlust höchstens 10 Prozentpunkte beträgt.",
        6: "Anteil der Paare, die Energie- und Qualitätsziel gleichzeitig erfüllen.",
        7: "Summe der separat attributierten Controller-Energie in Wh für die ausgewählten Controller und Datasets.",
        10: "Kontrafaktische Stop-Epoche je Dataset. RAPEC-G v4 und Standard ES beobachten dieselbe reale Full100-Trajektorie.",
        11: "Beste normalisierte Qualität von Full100 und beste bis zum jeweiligen Stopppunkt beobachtete Qualität, jeweils in Prozent.",
        12: "Normalisierter Quality Regret je Dataset: (Full100-Qualität − Qualität am Stop) × 100. Positive Werte sind Qualitätsverluste.",
        13: "Gemessene Full100-GPU-Energie, gemessene GPU-Energie bis zum Stop und kontrafaktische Gesamtenergie einschließlich Controller-Overhead.",
        14: "Separat attributierte Overhead-Energie von RAPEC-G v4 und Standard ES je Dataset; Training ist nicht enthalten.",
        15: "Reine Rechenzeit der Stop-Entscheidungen von RAPEC-G v4 und Standard ES je Dataset; Trainingszeit ist nicht enthalten.",
        16: "Relative Energieeinsparung gegenüber Full100 je Dataset. Positive Werte bedeuten Einsparung, negative Werte Mehrverbrauch.",
    }
    for panel in panels:
        if panel.get("type") != "row" and "description" not in panel:
            panel["description"] = descriptions.get(
                panel.get("id"),
                "Methodische Einordnung und Messumfang dieses Panels sind im Dashboard dokumentiert.",
            )
    return {
        "annotations": {"list": []},
        "editable": True,
        "graphTooltip": 1,
        "id": None,
        "uid": uid,
        "title": title,
        "tags": tags
        or ["controller-benchmark", "live-shadow", "nine-datasets", "development"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "30s",
        "time": {"from": "now-30d", "to": "now"},
        "panels": panels,
        "templating": {
            "list": [
                variable("benchmark_run_id", "Run", "label_values(live_shadow_controller_stop_epoch, benchmark_run_id)", multi=False, default=default_run_id),
                variable("controllers", "Controllers", 'label_values(live_shadow_controller_stop_epoch{benchmark_run_id="$benchmark_run_id"}, controller_id)'),
                variable("task_types", "Task Types", 'label_values(live_shadow_controller_stop_epoch{benchmark_run_id="$benchmark_run_id",controller_id=~"$controllers"}, task_type)'),
                variable("stages", "Stages", 'label_values(live_shadow_controller_stop_epoch{benchmark_run_id="$benchmark_run_id",controller_id=~"$controllers"}, stage)'),
                variable("cases", "Datasets", 'label_values(live_shadow_controller_stop_epoch{benchmark_run_id="$benchmark_run_id",controller_id=~"$controllers"}, case_id)'),
                variable("curve_case", "Quality-Energy Dataset", 'label_values(thesis_live_shadow_epoch_quality_score{benchmark_run_id="$benchmark_run_id"}, case_id)', multi=False, default=default_curve_case),
            ]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--title", default="Live Shadow Controllers - Nine Dataset Development"
    )
    parser.add_argument("--uid", default="controller-live-shadow-nine-dataset")
    parser.add_argument("--tag", action="append", dest="tags")
    parser.add_argument("--default-run-id", default="")
    parser.add_argument("--default-curve-case", default="")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            build(
                title=args.title,
                uid=args.uid,
                tags=args.tags,
                default_run_id=args.default_run_id,
                default_curve_case=args.default_curve_case,
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
