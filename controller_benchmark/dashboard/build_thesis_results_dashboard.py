from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from controller_benchmark.case_labels import CASE_LABELS


DS = {"type": "prometheus", "uid": "prometheus"}
ROOT = Path(__file__).resolve().parents[2]


def target(ref: str, expression: str, legend: str = "") -> dict:
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


def row(panel_id: int, title: str, description: str, y: int) -> dict:
    return {
        "id": panel_id,
        "title": title,
        "description": description,
        "type": "row",
        "collapsed": False,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "panels": [],
    }


def stat(
    panel_id: int,
    title: str,
    description: str,
    expression: str,
    unit: str,
    x: int,
    y: int,
    width: int = 6,
    decimals: int | None = 2,
) -> dict:
    defaults: dict = {
        "unit": unit,
        "color": {"mode": "thresholds"},
        "thresholds": {
            "mode": "absolute",
            "steps": [{"color": "green", "value": None}],
        },
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    return {
        "id": panel_id,
        "title": title,
        "description": description,
        "type": "stat",
        "gridPos": {"x": x, "y": y, "w": width, "h": 5},
        "datasource": DS,
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "colorMode": "value",
            "graphMode": "none",
            "justifyMode": "auto",
            "orientation": "auto",
            "textMode": "value_and_name",
            "reduceOptions": {
                "calcs": ["lastNotNull"],
                "fields": "",
                "values": False,
            },
        },
        "targets": [target("A", expression, title)],
    }


def bars(
    panel_id: int,
    title: str,
    description: str,
    expressions: list[tuple[str, str, str]],
    unit: str,
    x: int,
    y: int,
    width: int = 12,
    height: int = 11,
) -> dict:
    colors = ["#2E8B57", "#E69F00", "#1976D2", "#7A3FB0"]
    specs = [
        {"ref": ref, "name": legend, "color": colors[index % len(colors)]}
        for index, (ref, _expr, legend) in enumerate(expressions)
    ]
    unit_label = {
        "none": "Epoch",
        "percent": "%",
        "percentpoint": "Percentage points",
        "watth": "Wh",
        "s": "Seconds",
    }.get(unit, unit)
    script = (FRAME_JS + r"""
const specs = __SPECS__;
const rows = new Map();
frames.forEach(frame => {
  const labels=frameLabels(frame), value=frameValue(frame), refId=frameRef(frame), caseId=labels.case_id;
  if (!caseId || !refId || !Number.isFinite(value)) return;
  const row=rows.get(caseId)||{case_id:caseId,code:labels.case_code||caseId.slice(0,3).toUpperCase(),fullName:labels.case_name||caseId,order:Number(labels.case_order)||999};
  row[refId]=value; rows.set(caseId,row);
});
const data=Array.from(rows.values()).sort((a,b)=>a.order-b.order||a.code.localeCompare(b.code));
const series=specs.map(spec=>({name:spec.name,type:'bar',itemStyle:{color:spec.color},data:data.map(row=>({value:row[spec.ref],code:row.code,fullName:row.fullName,case_id:row.case_id}))}));
return {animation:false,tooltip:{trigger:'item',formatter:p=>`${p.seriesName}<br/><b>${p.data.code} — ${p.data.fullName}</b><br/>Case ID: ${p.data.case_id}<br/>Value: ${Number(p.data.value).toFixed(3)} __UNIT__`},legend:{bottom:0},grid:{left:80,right:25,top:35,bottom:90},xAxis:{type:'category',data:data.map(row=>row.code),axisLabel:{rotate:45}},yAxis:{type:'value',name:'__UNIT__',nameLocation:'middle',nameGap:60,axisLine:{onZero:true}},series};
""").replace("__SPECS__", json.dumps(specs)).replace("__UNIT__", unit_label)
    return echarts(
        panel_id,
        title,
        description + " Achsen verwenden eindeutige Drei-Zeichen-Kürzel; der Tooltip enthält den vollständigen Namen.",
        [target(ref, expr, legend) for ref, expr, legend in expressions],
        script,
        x,
        y,
        width,
        height,
    )


def echarts(
    panel_id: int,
    title: str,
    description: str,
    targets: list[dict],
    script: str,
    x: int,
    y: int,
    width: int = 24,
    height: int = 14,
) -> dict:
    chart_targets = copy.deepcopy(targets)
    for chart_target in chart_targets:
        # Instant time-series frames retain Prometheus labels on the numeric
        # value field, which the ECharts adapter needs for dynamic grouping.
        chart_target["format"] = "time_series"
    return {
        "id": panel_id,
        "title": title,
        "description": description,
        "type": "volkovlabs-echarts-panel",
        "gridPos": {"x": x, "y": y, "w": width, "h": height},
        "datasource": DS,
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {
            "renderer": "svg",
            "followTheme": True,
            "getOption": script,
        },
        "targets": chart_targets,
    }


def variable(
    name: str,
    label: str,
    query_text: str,
    *,
    multi: bool = True,
    default: str = "",
    hide: int = 0,
) -> dict:
    current = (
        {"selected": True, "text": ["All"], "value": ["$__all"]}
        if multi
        else {"selected": True, "text": default, "value": default}
    )
    return {
        "name": name,
        "label": label,
        "type": "query",
        "hide": hide,
        "datasource": DS,
        "query": {"query": query_text, "refId": f"var-{name}"},
        "definition": query_text,
        "refresh": 1,
        "sort": 1,
        "multi": multi,
        "includeAll": multi,
        "allValue": ".*" if multi else None,
        "current": current,
        "options": [],
    }


def custom_variable(name: str, label: str, values: list[str], default: str) -> dict:
    return {
        "name": name,
        "label": label,
        "type": "custom",
        "hide": 0,
        "query": ",".join(values),
        "current": {"selected": True, "text": default, "value": default},
        "options": [
            {"selected": value == default, "text": value, "value": value}
            for value in values
        ],
    }


def constant_variable(name: str, value: str) -> dict:
    """A hidden immutable scope so this appendix dashboard cannot show old runs."""
    return {
        "name": name,
        "label": "Current Thesis Run",
        "type": "constant",
        "hide": 2,
        "query": value,
        "current": {"selected": True, "text": value, "value": value},
        "options": [{"selected": True, "text": value, "value": value}],
    }


FRAME_JS = r"""
const frames = context?.panel?.data?.series || [];
const arr = value => {
  if (value == null) return [];
  if (typeof value.toArray === 'function') return value.toArray();
  if (value.buffer) return Array.from(value.buffer);
  return Array.isArray(value) ? value : [];
};
const fieldsOf = frame => (frame.fields && frame.fields.length) ? frame.fields : (frame.schema?.fields || []);
const numberFieldIndex = frame => {
  const fields = fieldsOf(frame);
  const index = fields.findIndex(field => field.type === 'number' || field.name === 'Value');
  return index >= 0 ? index : (fields.length > 1 ? 1 : -1);
};
const frameValue = frame => {
  const index = numberFieldIndex(frame);
  if (index < 0) return null;
  const local = frame.fields || [];
  const values = local[index]?.values != null ? arr(local[index].values) : arr(frame.data?.values?.[index]);
  return values.length ? Number(values[values.length - 1]) : null;
};
const frameLabels = frame => fieldsOf(frame)[numberFieldIndex(frame)]?.labels || {};
const frameRef = frame => frame.refId || frame.schema?.refId || null;
"""

CASE_CATALOG_JS = "\nconst caseCatalog = " + json.dumps(
    {
        case_id: {"code": code, "fullName": full_name, "order": order}
        for case_id, (code, full_name, order) in CASE_LABELS.items()
    },
    ensure_ascii=False,
) + r""";
const caseMeta = caseId => caseCatalog[caseId] || {code:String(caseId||'UNK').slice(0,3).toUpperCase(),fullName:String(caseId||'Unknown'),order:999};
const caseColor = order => `hsl(${((Number(order)||1)-1)*137.508%360}, 68%, 48%)`;
"""
OPERATIONAL_FRAME_JS = FRAME_JS + CASE_CATALOG_JS


PHASE_DURATION_SCRIPT = OPERATIONAL_FRAME_JS + r"""
const specs=[['A','Preprocessing & Initialization','#4C78A8'],['B','Training','#F58518'],['C','Finalization & Evaluation','#54A24B'],['D','Other Instrumented Work','#B279A2']];
const rows=new Map();
frames.forEach(frame=>{const labels=frameLabels(frame),value=frameValue(frame),refId=frameRef(frame),caseId=labels.benchmark_case_id;if(!caseId||!refId||!Number.isFinite(value))return;const meta=caseMeta(caseId),row=rows.get(caseId)||{case_id:caseId,jobId:labels.job_id,...meta};row[refId]=value;rows.set(caseId,row);});
const data=Array.from(rows.values()).sort((a,b)=>a.order-b.order);
const series=specs.map(([ref,name,color])=>({name,type:'bar',stack:'duration',itemStyle:{color},data:data.map(row=>({value:row[ref],code:row.code,fullName:row.fullName,case_id:row.case_id,jobId:row.jobId,phase:name}))}));
return {animation:false,tooltip:{trigger:'item',formatter:p=>`<b>${p.data.code} — ${p.data.fullName}</b><br/>Job ${p.data.jobId}<br/>Phase: ${p.data.phase}<br/>Duration: ${Number(p.data.value).toFixed(3)} s`},legend:{type:'scroll',bottom:0},grid:{left:80,right:25,top:35,bottom:90},xAxis:{type:'category',data:data.map(row=>row.code),axisLabel:{rotate:45}},yAxis:{type:'value',name:'Duration (s)',nameLocation:'middle',nameGap:60,min:0},series};
"""


PHASE_SCRIPT = OPERATIONAL_FRAME_JS + r"""
const rows = new Map();
frames.forEach(frame => {
  const labels = frameLabels(frame), value = frameValue(frame);
  if (!Number.isFinite(value) || !labels.job_id) return;
  const key = labels.job_id;
  const caseId=labels.benchmark_case_id, meta=caseMeta(caseId);
  const row = rows.get(key) || {job: `Job ${key}`, id: Number(key),case_id:caseId,...meta};
  row[frameRef(frame)] = value;
  rows.set(key, row);
});
const data = Array.from(rows.values()).sort((a,b) => a.id - b.id);
return {
  animation: false,
  tooltip: {trigger:'axis',formatter:items=>{const d=items[0]?.data||{};return `<b>${d.code} — ${d.fullName}</b><br/>${d.job}<br/>`+items.map(item=>`${item.marker}${item.seriesName}: ${Number(item.value).toFixed(3)}`).join('<br/>');}},
  legend: {bottom: 0},
  grid: {left:75,right:80,top:45,bottom:85},
  xAxis: {type:'category',data:data.map(row=>row.code),axisLabel:{rotate:45}},
  yAxis: [
    {type: 'value', name: 'GPU energy (Wh)', nameLocation: 'middle', nameGap: 52, min: 0},
    {type: 'value', name: 'Quality (%)', min: 0, max: 100},
    {type: 'value', name: 'Epochs', offset: 55, min: 0}
  ],
  series: [
    {name:'GPU energy (Wh)',type:'bar',yAxisIndex:0,data:data.map(row=>({value:row.A,...row})),itemStyle:{color:'#2E8B57'}},
    {name:'Quality (%)',type:'line',yAxisIndex:1,data:data.map(row=>({value:row.B,...row})),symbolSize:7,itemStyle:{color:'#7A3FB0'}},
    {name:'Epoch count',type:'line',yAxisIndex:2,data:data.map(row=>({value:row.C,...row})),symbolSize:7,lineStyle:{type:'dashed'},itemStyle:{color:'#5E5E5E'}}
  ]
};
"""


SIGNED_DIFF_SCRIPT = OPERATIONAL_FRAME_JS + r"""
const data = frames.map(frame => {
  const labels = frameLabels(frame), value = frameValue(frame);
  const caseId=labels.benchmark_case_id,meta=caseMeta(caseId);
  return {job:`Job ${labels.job_id}`,id:Number(labels.job_id),value,scenario:labels.scenario,strategy:labels.comparison_strategy,case_id:caseId,...meta};
}).filter(row => Number.isFinite(row.value)).sort((a,b) => a.id - b.id);
return {
  animation: false,
  tooltip: {formatter:p=>`<b>${p.data.code} — ${p.data.fullName}</b><br/>${p.data.job}<br/>${p.data.scenario||''}<br/>${p.data.strategy||''}<br/><b>SLURM − CodeCarbon: ${p.data.value.toFixed(3)} Wh</b>`},
  grid: {left: 75, right: 25, top: 35, bottom: 90},
  xAxis: {type:'category',data:data.map(row=>row.code),axisLabel:{rotate:45}},
  yAxis: {type: 'value', name: 'Signed difference (Wh)', nameLocation: 'middle', nameGap: 55},
  series: [{type: 'bar', data: data.map(row => ({...row, itemStyle: {color: row.value >= 0 ? '#2E8B57' : '#C23B4A'}})), markLine: {silent: true, symbol: 'none', lineStyle: {color: '#777', width: 2}, data: [{yAxis: 0, name: 'Equal'}]}}]
};
"""


def epoch_job_script(y_axis_name: str, tooltip_unit: str) -> str:
    return (OPERATIONAL_FRAME_JS + r"""
const datasets=new Map();
frames.forEach(frame=>{const labels=frameLabels(frame),value=frameValue(frame),caseId=labels.benchmark_case_id,epoch=Number(labels.epoch);if(!caseId||!Number.isFinite(epoch)||!Number.isFinite(value))return;const meta=caseMeta(caseId),dataset=datasets.get(caseId)||{case_id:caseId,jobId:labels.job_id,...meta,points:[]};dataset.points.push({value:[epoch,value],epoch,jobId:labels.job_id,case_id:caseId,...meta});datasets.set(caseId,dataset);});
const ordered=Array.from(datasets.values()).sort((a,b)=>a.order-b.order);
const series=ordered.map(dataset=>({name:dataset.code,type:'line',showSymbol:false,symbolSize:5,lineStyle:{width:1.6,color:caseColor(dataset.order)},itemStyle:{color:caseColor(dataset.order)},data:dataset.points.sort((a,b)=>a.epoch-b.epoch)}));
return {animation:false,tooltip:{trigger:'item',formatter:p=>`<b>${p.data.code} — ${p.data.fullName}</b><br/>Job ${p.data.jobId}<br/>Epoch: ${p.data.epoch}<br/>Value: ${Number(p.data.value[1]).toFixed(4)} __TOOLTIP_UNIT__`},legend:{type:'scroll',bottom:0,data:ordered.map(dataset=>dataset.code)},grid:{left:80,right:25,top:35,bottom:95},xAxis:{type:'value',name:'Epoch',nameLocation:'middle',nameGap:35,min:1},yAxis:{type:'value',name:'__Y_AXIS__',nameLocation:'middle',nameGap:60,min:0},series};
""").replace("__Y_AXIS__", y_axis_name).replace("__TOOLTIP_UNIT__", tooltip_unit)


ENERGY_QUALITY_CURVE_SCRIPT = FRAME_JS + r"""
const datasets = new Map();
frames.forEach(frame => {
  const labels = frameLabels(frame), value = frameValue(frame);
  if (!Number.isFinite(value) || !labels.case_id) return;
  const caseId=labels.case_id;
  const dataset=datasets.get(caseId)||{case_id:caseId,code:labels.case_code||caseId.slice(0,3).toUpperCase(),fullName:labels.case_name||caseId,order:Number(labels.case_order)||999,epochs:new Map(),rapecStop:null,esStop:null};
  const refId = frameRef(frame);
  if (refId === 'A' || refId === 'B') {
    const epoch = Number(labels.epoch); if (!Number.isFinite(epoch)) return;
    const point = dataset.epochs.get(epoch) || {epoch}; point[refId] = value; dataset.epochs.set(epoch, point);
  }
  if (refId === 'C') {dataset.rapecStop=value;dataset.rapecReason=labels.stop_reason||'not reported';}
  else if (refId === 'D') {dataset.esStop=value;dataset.esReason=labels.stop_reason||'not reported';}
  else if (refId !== 'A' && refId !== 'B') dataset[refId]=value;
  datasets.set(caseId,dataset);
});
const ordered=Array.from(datasets.values()).sort((a,b)=>a.order-b.order||a.code.localeCompare(b.code));
const curves=[],rapecMarkers=[],esMarkers=[],fullMarkers=[];
ordered.forEach(dataset=>{
  const rows=Array.from(dataset.epochs.values()).filter(p=>Number.isFinite(p.A)&&Number.isFinite(p.B)).sort((a,b)=>a.epoch-b.epoch);
  let cumulative=0,best=-Infinity;
  const curve=rows.map(p=>{cumulative+=p.A;best=Math.max(best,p.B);return{value:[cumulative,best],epoch:p.epoch,code:dataset.code,fullName:dataset.fullName,case_id:dataset.case_id};});
  if(!curve.length)return;
  curves.push({name:dataset.code,type:'line',showSymbol:true,symbol:'circle',symbolSize:8,data:curve,lineStyle:{width:2},itemStyle:{opacity:0},emphasis:{focus:'series',lineStyle:{width:4},itemStyle:{opacity:1}}});
  const pointAt=stop=>curve.find(point=>point.epoch===Number(stop));
  const rapec=pointAt(dataset.rapecStop),es=pointAt(dataset.esStop),full=curve[curve.length-1];
  dataset.rapecPoint=rapec;
  dataset.esPoint=es;
  if(rapec)rapecMarkers.push({...rapec,stop:'RAPEC-G v4',energySaving:dataset.F,qualityRegret:dataset.G,overheadEnergy:dataset.H,computeSeconds:dataset.I});
  if(es)esMarkers.push({...es,stop:'Standard ES',energySaving:dataset.K,qualityRegret:dataset.L,overheadEnergy:dataset.M,computeSeconds:dataset.N});
  if(full)fullMarkers.push({...full,stop:'Full100'});
});
const marker=(name,data,color,symbol,size)=>({name,type:'scatter',data,itemStyle:{color},symbol,symbolSize:size});
const fmt=(value,digits,suffix='')=>Number.isFinite(Number(value))?`${Number(value).toFixed(digits)}${suffix}`:'—';
const controllerValues=(point,energySaving,qualityRegret,overheadEnergy,computeSeconds)=>({epoch:point?.epoch??'—',quality:point?fmt(point.value[1],2,' %'):'—',training:point?fmt(point.value[0],3,' Wh'):'—',saving:fmt(energySaving,2,' %'),regret:fmt(qualityRegret,2,' pp'),overhead:fmt(overheadEnergy,5,' Wh'),compute:fmt(computeSeconds,4,' s')});
const comparisonTable=dataset=>{const r=controllerValues(dataset.rapecPoint,dataset.F,dataset.G,dataset.H,dataset.I),e=controllerValues(dataset.esPoint,dataset.K,dataset.L,dataset.M,dataset.N);const row=(label,a,b)=>`<tr><td style="padding:2px 7px 2px 0;color:#aaa">${label}</td><td style="padding:2px 8px">${a}</td><td style="padding:2px 0 2px 8px">${b}</td></tr>`;return `<table style="margin-top:6px;border-collapse:collapse;font-size:11px;line-height:1.2;max-width:700px"><tr><th></th><th style="border-left:4px solid #2E8B57;padding:3px 8px;text-align:left">RAPEC-G v4</th><th style="border-left:4px solid #E69F00;padding:3px 8px;text-align:left">Standard ES</th></tr>${row('Stop epoch',r.epoch,e.epoch)}${row('Quality at stop',r.quality,e.quality)}${row('Training GPU energy',r.training,e.training)}${row('Energy saving',r.saving,e.saving)}${row('Quality regret',r.regret,e.regret)}${row('Overhead',r.overhead,e.overhead)}${row('Compute time',r.compute,e.compute)}</table>`;};
return {
  animation: false,
  tooltip: {trigger:'item',confine:true,enterable:true,extraCssText:'max-width:740px;white-space:normal;',formatter:p=>{const d=p.data,dataset=datasets.get(d.case_id),head=`<div style="font-size:11px;line-height:1.25"><b>${d.code} — ${d.fullName}</b> · ${d.stop||'Full100 trajectory'}<br/>Epoch ${d.epoch} · ${d.value[0].toFixed(3)} Wh cumulative GPU · ${d.value[1].toFixed(2)} % best quality</div>`;return dataset?head+comparisonTable(dataset):head;}},
  legend: {type:'scroll',bottom:0,data:curves.map(series=>series.name)},
  grid: {left:80,right:35,top:45,bottom:95},
  xAxis: {type:'value',name:'Cumulative Full100 GPU energy (Wh)',nameLocation:'middle',nameGap:48,min:0},
  yAxis: {type:'value',name:'Best normalized quality (%)',nameLocation:'middle',nameGap:58,min:0,max:100},
  series:[...curves,marker('RAPEC-G v4 stop',rapecMarkers,'#2E8B57','circle',13),marker('Standard ES stop',esMarkers,'#E69F00','diamond',13),marker('Full100 endpoint',fullMarkers,'#1976D2','rect',8)]
};
"""


ENERGY_PER_QUALITY_SCRIPT = FRAME_JS + r"""
const rows = new Map();
frames.forEach(frame => {
  const labels = frameLabels(frame), value = frameValue(frame), caseId = labels.case_id;
  if (!caseId || !Number.isFinite(value)) return;
  const row = rows.get(caseId) || {case_id:caseId,code:labels.case_code||caseId.slice(0,3).toUpperCase(),fullName:labels.case_name||caseId,order:Number(labels.case_order)||999,task:labels.task_type || 'unknown'};
  row[frameRef(frame)] = value; rows.set(caseId,row);
});
const data = Array.from(rows.values()).sort((a,b)=>a.order-b.order||a.code.localeCompare(b.code));
const whpp = (energy, quality, start) => {
  const gain = (quality-start)*100;
  return Number.isFinite(energy) && Number.isFinite(gain) && gain > 1e-9 ? energy/gain : null;
};
const specs = [['RAPEC-G v4','C','D','#2E8B57'],['Standard ES','E','F','#E69F00'],['Full100','A','B','#1976D2']];
const series = specs.map(([name,e,q,color]) => ({name,type:'bar',itemStyle:{color},data:data.map(row => ({value:whpp(row[e],row[q],row.G),case_id:row.case_id,code:row.code,fullName:row.fullName,energy:row[e],quality:row[q],start:row.G}))}));
return {
  animation:false,
  tooltip:{trigger:'item',formatter:p=>{const d=p.data;if(!Number.isFinite(d.value))return `${p.seriesName}<br/><b>${d.code} — ${d.fullName}</b><br/>Case ID: ${d.case_id}<br/>Undefined: no positive quality gain`;return `${p.seriesName}<br/><b>${d.code} — ${d.fullName}</b><br/>Case ID: ${d.case_id}<br/>Energy: ${d.energy.toFixed(3)} Wh<br/>Start quality: ${(d.start*100).toFixed(2)} %<br/>Stop quality: ${(d.quality*100).toFixed(2)} %<br/><b>${d.value.toFixed(3)} Wh/pp</b>`;}},
  legend:{bottom:0},grid:{left:80,right:25,top:35,bottom:125},
  xAxis:{type:'category',data:data.map(row=>row.code),axisLabel:{rotate:45}},
  yAxis:{type:'value',name:'GPU energy (Wh) per +1 quality pp',nameLocation:'middle',nameGap:62,min:0},series
};
"""


RAW_QUALITY_SCRIPT = FRAME_JS + r"""
const raw = new Map(), stops = {rapec:new Map(), es:new Map()};
frames.forEach(frame => {
  const labels=frameLabels(frame), value=frameValue(frame);
  if (!Number.isFinite(value) || !labels.case_id) return;
  const refId=frameRef(frame);
  if (refId === 'B') {stops.rapec.set(labels.case_id,value);return;}
  if (refId === 'C') {stops.es.set(labels.case_id,value);return;}
  if (refId !== 'A' || !labels.raw_quality_metric) return;
  const key=`${labels.raw_quality_metric}|||${labels.case_id}`;
  const rows=raw.get(key)||[]; rows.push({epoch:Number(labels.epoch),value,metric:labels.raw_quality_metric,case_id:labels.case_id,code:labels.case_code||labels.case_id.slice(0,3).toUpperCase(),fullName:labels.case_name||labels.case_id,order:Number(labels.case_order)||999}); raw.set(key,rows);
});
const lowerBetter = new Set(['loss','cross_entropy','perplexity','mae','mse','rmse','nrmse','error_rate']);
const groups = new Map();
for (const rows of raw.values()) {
  rows.sort((a,b)=>a.epoch-b.epoch); const sample=rows[0], minimize=lowerBetter.has(sample.metric);
  const bestUntil=stop=>{const chosen=rows.filter(r=>r.epoch<=stop).map(r=>r.value);if(!chosen.length)return null;return minimize?Math.min(...chosen):Math.max(...chosen);};
  const result={case_id:sample.case_id,code:sample.code,fullName:sample.fullName,order:sample.order,rapec:bestUntil(stops.rapec.get(sample.case_id)),es:bestUntil(stops.es.get(sample.case_id)),full:bestUntil(Infinity)};
  const list=groups.get(sample.metric)||[]; list.push(result); groups.set(sample.metric,list);
}
const metrics=Array.from(groups.keys()).sort();
if (!metrics.length) return {title:{text:'No raw-quality data for the current selection',left:'center',top:'middle'},series:[]};
const columns=Math.min(2,metrics.length), rowsCount=Math.ceil(metrics.length/columns), rowSlot=86/rowsCount;
const grid=[],xAxis=[],yAxis=[],series=[],titles=[];
metrics.forEach((metric,index)=>{const col=index%columns,row=Math.floor(index/columns),left=col===0?'7%':'56%',width=columns===1?'88%':'39%',titleTop=3+row*rowSlot,gridTop=titleTop+5,gridHeight=Math.max(7,rowSlot-11);const values=groups.get(metric).sort((a,b)=>a.order-b.order||a.code.localeCompare(b.code));grid.push({left,width,top:`${gridTop}%`,height:`${gridHeight}%`,containLabel:true});xAxis.push({type:'category',gridIndex:index,data:values.map(v=>v.code),axisLabel:{rotate:35,fontSize:8,hideOverlap:true}});yAxis.push({type:'value',gridIndex:index,scale:true,axisLabel:{fontSize:8}});titles.push({text:metric.replaceAll('_',' '),left,top:`${titleTop}%`,textStyle:{fontSize:12,fontWeight:'bold'}});[['RAPEC-G v4','rapec','#2E8B57'],['Standard ES','es','#E69F00'],['Full100','full','#1976D2']].forEach(([name,key,color])=>series.push({name,type:'bar',xAxisIndex:index,yAxisIndex:index,itemStyle:{color},data:values.map(v=>({value:v[key],code:v.code,fullName:v.fullName,case_id:v.case_id,metric}))}));});
return {animation:false,title:titles,tooltip:{trigger:'item',formatter:p=>`${p.seriesName}<br/><b>${p.data.code} — ${p.data.fullName}</b><br/>Case ID: ${p.data.case_id}<br/>Metric: ${p.data.metric}<br/>Value: ${Number(p.data.value).toFixed(4)}`},legend:{bottom:0},grid,xAxis,yAxis,series};
"""


TRADEOFF_SCRIPT = FRAME_JS + r"""
const rows=new Map();
frames.forEach(frame=>{const labels=frameLabels(frame),value=frameValue(frame),refId=frameRef(frame);if(!refId||!labels.case_id||!Number.isFinite(value))return;const key=`${refId[0]}:${labels.case_id}`,row=rows.get(key)||{case_id:labels.case_id,code:labels.case_code||labels.case_id.slice(0,3).toUpperCase(),fullName:labels.case_name||labels.case_id,task:labels.task_type,kind:refId[0]};row[refId[1]]=value;rows.set(key,row);});
const make=(kind,name,color,symbol,index,comparison)=>({name,type:'scatter',xAxisIndex:index,yAxisIndex:index,symbol,symbolSize:13,itemStyle:{color},label:{show:true,position:'top',fontSize:9,formatter:p=>p.data.code},data:Array.from(rows.values()).filter(r=>r.kind===kind&&Number.isFinite(r.X)&&Number.isFinite(r.Y)).map(r=>({value:[r.X,r.Y],case_id:r.case_id,code:r.code,fullName:r.fullName,task:r.task,comparison}))});
return {animation:false,title:[{text:'RAPEC-G v4 vs Full100',left:'10%'},{text:'Standard ES vs Full100',left:'42%'},{text:'RAPEC-G v4 vs Standard ES',left:'73%'}],tooltip:{formatter:p=>`${p.data.comparison}<br/><b>${p.data.code} — ${p.data.fullName}</b><br/>Case ID: ${p.data.case_id}<br/>Task: ${p.data.task}<br/>Energy saving: ${p.data.value[0].toFixed(2)} %<br/>Quality regret: ${p.data.value[1].toFixed(2)} pp`},legend:{bottom:0},grid:[{left:'5%',width:'27%',top:55,bottom:75},{left:'37%',width:'27%',top:55,bottom:75},{left:'69%',width:'27%',top:55,bottom:75}],xAxis:[{type:'value',gridIndex:0,name:'RAPEC-G saving vs Full100 (%)',nameLocation:'middle',nameGap:42,axisLine:{onZero:true}},{type:'value',gridIndex:1,name:'ES saving vs Full100 (%)',nameLocation:'middle',nameGap:42,axisLine:{onZero:true}},{type:'value',gridIndex:2,name:'RAPEC-G saving vs ES (%)',nameLocation:'middle',nameGap:42,axisLine:{onZero:true}}],yAxis:[{type:'value',gridIndex:0,name:'RAPEC-G regret vs Full100 (pp)',nameLocation:'middle',nameGap:52,axisLine:{onZero:true}},{type:'value',gridIndex:1,name:'ES regret vs Full100 (pp)',nameLocation:'middle',nameGap:52,axisLine:{onZero:true}},{type:'value',gridIndex:2,name:'RAPEC-G regret vs ES (pp)',nameLocation:'middle',nameGap:52,axisLine:{onZero:true}}],series:[make('R','RAPEC-G vs Full100','#2E8B57','circle',0,'RAPEC-G v4 vs Full100'),make('E','ES vs Full100','#E69F00','diamond',1,'Standard ES vs Full100'),make('C','RAPEC-G vs ES','#7A3FB0','triangle',2,'RAPEC-G v4 vs Standard ES')]};
"""


TASK_GENERALISATION_SCRIPT = FRAME_JS + r"""
const rows=new Map();
frames.forEach(frame=>{const labels=frameLabels(frame),value=frameValue(frame),refId=frameRef(frame);if(!refId||!labels.case_id||!Number.isFinite(value))return;const row=rows.get(labels.case_id)||{case_id:labels.case_id,task:labels.task_type||'unknown'};row[refId]=value;rows.set(labels.case_id,row);});
const grouped=new Map();for(const row of rows.values()){const list=grouped.get(row.task)||[];list.push(row);grouped.set(row.task,list);}
const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
const ci=a=>{if(a.length<2)return[mean(a),mean(a)];let state=2166136261>>>0;const next=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};const samples=[];for(let b=0;b<1000;b++){let sum=0;for(let i=0;i<a.length;i++)sum+=a[Math.floor(next()*a.length)];samples.push(sum/a.length);}samples.sort((x,y)=>x-y);return[samples[24],samples[974]];};
const tasks=Array.from(grouped.keys()).sort(), energy=[], regret=[];
tasks.forEach(task=>{const list=grouped.get(task),e=list.map(r=>r.A).filter(Number.isFinite),q=list.map(r=>r.B).filter(Number.isFinite),ec=ci(e),qc=ci(q);energy.push({value:mean(e),ci:ec,n:e.length});regret.push({value:mean(q),ci:qc,n:q.length});});
const errorSeries=(values,index,color)=>({type:'custom',xAxisIndex:index,yAxisIndex:index,silent:true,renderItem:(params,api)=>{const x=api.coord([api.value(0),api.value(1)]),lo=api.coord([api.value(0),api.value(2)]),hi=api.coord([api.value(0),api.value(3)]);return{type:'group',children:[{type:'line',shape:{x1:x[0],y1:lo[1],x2:x[0],y2:hi[1]},style:{stroke:color,lineWidth:2}},{type:'line',shape:{x1:x[0]-5,y1:lo[1],x2:x[0]+5,y2:lo[1]},style:{stroke:color,lineWidth:2}},{type:'line',shape:{x1:x[0]-5,y1:hi[1],x2:x[0]+5,y2:hi[1]},style:{stroke:color,lineWidth:2}}]};},data:values.map((v,i)=>[i,v.value,v.ci[0],v.ci[1]])});
return {animation:false,tooltip:{trigger:'axis',formatter:items=>{const i=items[0]?.dataIndex??0;return `${tasks[i]}<br/>Energy saving: ${energy[i].value.toFixed(2)} % (95% bootstrap CI ${energy[i].ci[0].toFixed(2)}–${energy[i].ci[1].toFixed(2)}, n=${energy[i].n})<br/>Quality regret: ${regret[i].value.toFixed(2)} pp (95% bootstrap CI ${regret[i].ci[0].toFixed(2)}–${regret[i].ci[1].toFixed(2)}, n=${regret[i].n})`;}},title:[{text:'Energy saving',left:'23%'},{text:'Quality regret',left:'70%'}],grid:[{left:'7%',right:'55%',top:45,bottom:100},{left:'55%',right:'5%',top:45,bottom:100}],xAxis:[{type:'category',gridIndex:0,data:tasks,axisLabel:{rotate:45}},{type:'category',gridIndex:1,data:tasks,axisLabel:{rotate:45}}],yAxis:[{type:'value',gridIndex:0,name:'%'},{type:'value',gridIndex:1,name:'pp'}],series:[{name:'Mean energy saving',type:'bar',xAxisIndex:0,yAxisIndex:0,data:energy.map(v=>v.value),itemStyle:{color:'#2E8B57'}},errorSeries(energy,0,'#1B5E20'),{name:'Mean quality regret',type:'bar',xAxisIndex:1,yAxisIndex:1,data:regret.map(v=>v.value),itemStyle:{color:'#7A3FB0'}},errorSeries(regret,1,'#4A148C')]};
"""


def historical_panel(source_id: int, new_id: int, y: int, description: str) -> dict:
    source = json.loads(
        (ROOT / "grafana" / "dashboards" / "60-run.json").read_text(encoding="utf-8")
    )
    panel = copy.deepcopy(next(item for item in source["panels"] if item["id"] == source_id))
    panel["id"] = new_id
    panel["description"] = description
    panel["gridPos"]["y"] = y
    return panel


def build(default_run_id: str, default_curve_case: str) -> dict:
    lf = 'benchmark_run_id="$benchmark_run_id",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases"'
    opf = 'benchmark_run_id="$benchmark_run_id",task_type=~"$task_types",benchmark_stage=~"$stages",benchmark_case_id=~"$cases"'
    rapec = f'{lf},controller_id="rapec-g-v4"'
    es = f'{lf},controller_id="standard-es-10"'
    e_r = f"live_shadow_counterfactual_total_energy_wh{{{rapec}}}"
    e_s = f"live_shadow_counterfactual_total_energy_wh{{{es}}}"
    q_r = f"live_shadow_best_quality_at_stop{{{rapec}}}"
    q_s = f"live_shadow_best_quality_at_stop{{{es}}}"
    # Both virtual controllers belong to the same benchmark case, while their
    # controller_id and stop_reason labels intentionally differ. Restrict the
    # vector match to stable case identity labels so no completed pair is lost.
    pair_join = (
        "on(benchmark_run_id,benchmark_version,stage,case_id,case_code,"
        "case_name,case_order,task_type,energy_measurement_method)"
    )
    energy_rapec_es = f"(({e_s} - {pair_join} {e_r}) / {pair_join} {e_s}) * 100"
    regret_rapec_es = f"({q_s} - {pair_join} {q_r}) * 100"
    panels: list[dict] = []

    panels += [
        row(100, "1 · Scope and Global Operational Totals", "Ausschließlich Kennzahlen der ausgewählten 48-Fall-Thesis-Live-Shadow-Kampagne.", 0),
        {
            "id": 101,
            "title": "Dashboard Scope and Interpretation",
            "description": "Dokumentiert Datenquellen, Vergleichslogik und Grenzen des zentralen Thesis-Dashboards.",
            "type": "text",
            "gridPos": {"x": 0, "y": 1, "w": 24, "h": 4},
            "options": {"mode": "markdown", "content": "**Campaign scope:** 48 real Full100 jobs. Each real job produces one independently represented virtual Standard-ES job and one virtual RAPEC-G-v4 job from the same online Live-Shadow trajectory: 48 real + 48 ES virtual + 48 RAPEC-G v4 virtual = 144 represented jobs. Physical system and phase totals count only the 48 real executions. Cost and CO₂ are derived from energy using fixed factors."},
        },
    ]
    global_stats = [
        (1, "Total Jobs", "144 repräsentierte Jobs bei vollständiger Kampagne: 48 reale Full100-, 48 virtuelle Standard-ES- und 48 virtuelle RAPEC-G-v4-Jobs.", 'thesis_live_shadow_represented_jobs_total{benchmark_run_id="$benchmark_run_id"}', "none"),
        (2, "SLURM Total GPU Energy", "Physisch gemessene GPU-Energie der 48 realen Full100-Jobs über den vollständigen Job-Lebenszyklus.", f'sum(slurm_job_gpu_energy_kwh{{{opf}}}) * 1000', "watth"),
        (3, "CodeCarbon Total GPU Energy", "Von CodeCarbon geschätzte GPU-Energie der 48 realen Full100-Jobs über den vollständigen Job-Lebenszyklus.", f'sum(slurm_job_codecarbon_job_total_gpu_energy_kwh{{{opf}}}) * 1000', "watth"),
        (4, "CodeCarbon Total Energy", "CodeCarbon-Gesamtenergie der 48 realen Full100-Jobs einschließlich CPU- und RAM-Anteilen.", f'sum(slurm_job_codecarbon_job_total_energy_kwh{{{opf}}}) * 1000', "watth"),
        (5, "SLURM Total Cost", "Aus realer Full100-Jobenergie und konstantem Strompreis abgeleitete Kosten.", f'sum(slurm_job_estimated_electricity_cost_eur{{{opf}}})', "currencyEUR"),
        (6, "CodeCarbon Total Cost", "Aus CodeCarbon-Gesamtenergie und konstantem Strompreis abgeleitete Kosten.", f'sum(slurm_job_codecarbon_estimated_electricity_cost_eur{{{opf}}})', "currencyEUR"),
        (7, "SLURM Total CO₂", "Aus realer Full100-Jobenergie und konstantem Emissionsfaktor abgeleitete CO₂-Masse.", f'sum(slurm_job_estimated_co2_kg{{{opf}}})', "kg"),
        (8, "CodeCarbon Total CO₂", "Aus CodeCarbon-Gesamtenergie und konstantem Emissionsfaktor abgeleitete CO₂-Masse.", f'sum(slurm_job_codecarbon_estimated_co2_kg{{{opf}}})', "kg"),
    ]
    for index, (pid, title, desc, expr, unit) in enumerate(global_stats):
        panels.append(stat(pid, title, desc, expr, unit, (index % 4) * 6, 5 + (index // 4) * 5, 6, 2 if unit != "none" else 0))

    panels.append(row(200, "2 · Measurement Agreement, Phases, and Epoch Behaviour", "Vergleicht die beiden Messmethoden innerhalb derselben 48 realen Full100-Jobs.", 15))
    panels.append(
        echarts(
            201,
            "Phase Duration by Job (s)",
            "Vier klar benannte Lebenszyklusgruppen je realem Full100-Job. Die Achse verwendet die Drei-Zeichen-Kürzel; vollständiger Fallname und Job-ID stehen im Tooltip.",
            [
                target("A", f'slurm_job_phase_duration_seconds{{{opf},phase="preprocessing_initialization"}}'),
                target("B", f'slurm_job_phase_duration_seconds{{{opf},phase="training"}}'),
                target("C", f'slurm_job_phase_duration_seconds{{{opf},phase="finalization_evaluation"}}'),
                target("D", f'slurm_job_phase_duration_seconds{{{opf},phase="other"}}'),
            ],
            PHASE_DURATION_SCRIPT,
            0,
            16,
            24,
            9,
        )
    )
    phase_quality = f'max by (job_id) (slurm_job_epoch_quality_score{{{opf}}}) * 100'
    phase_epochs = f'count by (job_id) (slurm_job_epoch_gpu_energy_kwh{{{opf}}})'
    panels += [
        echarts(202, "SLURM Phase: Energy, Quality, and Epoch Count", "SLURM-GPU-Energie der ausgewählten Lebenszyklusphase, beste normalisierte Full100-Qualität und Epochenzahl. Kürzel stehen auf der Achse; vollständiger Fallname und Job-ID im Tooltip.", [target("A", f'slurm_job_phase_gpu_energy_kwh{{{opf},phase="$slurm_phase"}} * 1000'), target("B", phase_quality), target("C", phase_epochs)], PHASE_SCRIPT, 0, 25, 12, 14),
        echarts(203, "CodeCarbon Phase: Energy, Quality, and Epoch Count", "CodeCarbon-GPU-Energie der ausgewählten Lebenszyklusphase, beste normalisierte Full100-Qualität und Epochenzahl. Kürzel stehen auf der Achse; vollständiger Fallname und Job-ID im Tooltip.", [target("A", f'slurm_job_phase_codecarbon_gpu_energy_kwh{{{opf},phase="$codecarbon_phase"}} * 1000'), target("B", phase_quality), target("C", phase_epochs)], PHASE_SCRIPT, 12, 25, 12, 14),
        echarts(204, "SLURM − CodeCarbon Training GPU Difference by Job", "Vorzeichenbehaftete Differenz je realem Full100-Job: SLURM minus CodeCarbon. Positive Werte werden als grüne Balken nach oben, negative als rote Balken nach unten dargestellt. Die Achse verwendet Kürzel, der Tooltip den vollständigen Fallnamen und die Job-ID.", [target("A", f'(slurm_job_phase_gpu_energy_kwh{{{opf},phase="training"}} - ignoring(__name__,measurement_status) slurm_job_phase_codecarbon_gpu_energy_kwh{{{opf},phase="training"}}) * 1000')], SIGNED_DIFF_SCRIPT, 0, 39, 24, 13),
    ]
    panels += [
        echarts(205, "Epoch GPU Utilization by Job (%)", "Mittlere GPU-Auslastung je Epoche. Jeder der 48 Fälle erhält eine stabile eigene Farbe; die Legende unten ordnet Farbe und Kürzel zu, der Tooltip zeigt den vollständigen Namen.", [target("A", f'slurm_job_epoch_gpu_util_avg_pct{{{opf}}}')], epoch_job_script("GPU utilization (%)", "%"), 0, 52, 12, 9),
        echarts(206, "SLURM Epoch GPU Energy by Job (Wh)", "Gemessene SLURM-GPU-Energie je Epoche. Jeder der 48 Fälle erhält eine stabile eigene Farbe; die Legende unten ordnet Farbe und Kürzel zu, der Tooltip zeigt den vollständigen Namen.", [target("A", f'slurm_job_epoch_gpu_energy_kwh{{{opf}}} * 1000')], epoch_job_script("GPU energy (Wh)", "Wh"), 12, 52, 12, 9),
    ]

    panels.append(row(300, "3 · RAPEC-G v4 Pairwise Thesis Results", "Makro-Mittel über die ausgewählten Datasets für drei klar gerichtete Vergleiche.", 61))
    mean_specs = [
        (301, "Mean Energy Saving: RAPEC-G vs ES", "Makro-Mittel von (E_ES − E_RAPEC-G) / E_ES × 100. Positiv bedeutet, dass RAPEC-G gegenüber Standard ES Energie spart.", f"avg({energy_rapec_es})", "percent"),
        (302, "Mean Quality Regret: RAPEC-G vs ES", "Makro-Mittel von (Q_ES − Q_RAPEC-G) × 100. Positiv bedeutet geringere normalisierte Qualität von RAPEC-G.", f"avg({regret_rapec_es})", "percentpoint"),
        (303, "Mean Energy Saving: RAPEC-G vs Full100", "Makro-Mittel der relativen RAPEC-G-Energieeinsparung gegenüber Full100.", f'avg(live_shadow_energy_saving_fraction{{{rapec}}}) * 100', "percent"),
        (304, "Mean Quality Regret: RAPEC-G vs Full100", "Makro-Mittel des normalisierten RAPEC-G-Qualitätsverlusts gegenüber Full100.", f'avg(live_shadow_quality_regret{{{rapec}}}) * 100', "percentpoint"),
        (305, "Mean Energy Saving: ES vs Full100", "Makro-Mittel der relativen Standard-ES-Energieeinsparung gegenüber Full100.", f'avg(live_shadow_energy_saving_fraction{{{es}}}) * 100', "percent"),
        (306, "Mean Quality Regret: ES vs Full100", "Makro-Mittel des normalisierten Standard-ES-Qualitätsverlusts gegenüber Full100.", f'avg(live_shadow_quality_regret{{{es}}}) * 100', "percentpoint"),
        (307, "RAPEC-G Controller Overhead", "Summe der separat attributierten RAPEC-G-v4-Overhead-Energie der ausgewählten Datasets.", f'sum(live_shadow_controller_overhead_energy_wh{{{rapec}}})', "watth"),
        (308, "Standard ES Overhead", "Summe der separat attributierten Standard-ES-Overhead-Energie der ausgewählten Datasets.", f'sum(live_shadow_controller_overhead_energy_wh{{{es}}})', "watth"),
    ]
    for index, (pid, title, desc, expr, unit) in enumerate(mean_specs):
        panels.append(stat(pid, title, desc, expr, unit, (index % 4) * 6, 62 + (index // 4) * 5, 6))

    panels.append(row(400, "4 · Dataset-Level Controller Comparison", "RAPEC-G v4, Standard ES und Full100 je Dataset; keine Vermischung der Dataset-Ebene mit globalen Summen.", 72))
    panels += [
        bars(401, "Stop Epoch by Dataset: RAPEC-G, ES, Full100", "Stop-Epochen von RAPEC-G v4 und Standard ES auf derselben Full100-Trajektorie; Full100 erscheint als Referenz bei Epoche 100.", [("A", f'live_shadow_controller_stop_epoch{{{rapec}}}', "RAPEC-G v4"), ("B", f'live_shadow_controller_stop_epoch{{{es}}}', "Standard ES"), ("C", f'live_shadow_full100_best_quality{{{rapec}}} * 0 + 100', "Full100")], "none", 0, 73, 24),
        bars(402, "Normalized Quality by Dataset: RAPEC-G, ES, Full100", "Beste taskunabhängig normalisierte Qualität bis zum jeweiligen Stopppunkt. Full100 ist die beste Qualität der vollständigen realen Trajektorie.", [("A", f'{q_r} * 100', "RAPEC-G v4"), ("B", f'{q_s} * 100', "Standard ES"), ("C", f'live_shadow_full100_best_quality{{{rapec}}} * 100', "Full100")], "percent", 0, 84, 24),
        bars(403, "Quality Regret by Dataset: Three Comparisons", "Normalisierter Quality Regret in Prozentpunkten. Positive Werte bedeuten, dass das erstgenannte Verfahren schlechter ist; negative Werte eine Verbesserung.", [("A", f'live_shadow_quality_regret{{{rapec}}} * 100', "RAPEC-G vs Full100"), ("B", regret_rapec_es, "RAPEC-G vs ES"), ("C", f'live_shadow_quality_regret{{{es}}} * 100', "ES vs Full100")], "percentpoint", 0, 95, 24),
        bars(404, "Controller and ES Overhead Energy by Dataset", "Ausschließlich separat attributierte Overhead-Energie der Stop-Entscheidungen; Trainingsenergie ist nicht enthalten.", [("A", f'live_shadow_controller_overhead_energy_wh{{{rapec}}}', "RAPEC-G v4"), ("B", f'live_shadow_controller_overhead_energy_wh{{{es}}}', "Standard ES")], "watth", 0, 106, 12),
        bars(405, "Controller and ES Compute Time by Dataset", "Ausschließlich Rechenzeit der Stop-Entscheidungen; Trainingszeit und Datenvorbereitung sind nicht enthalten.", [("A", f'live_shadow_controller_compute_seconds{{{rapec}}}', "RAPEC-G v4"), ("B", f'live_shadow_controller_compute_seconds{{{es}}}', "Standard ES")], "s", 12, 106, 12),
        bars(406, "Energy Saving by Dataset: Three Comparisons", "Relative Energieeinsparung für RAPEC-G vs Full100, RAPEC-G vs Standard ES und Standard ES vs Full100. Positive Werte sind Einsparungen; negative Werte Mehrverbrauch.", [("A", f'live_shadow_energy_saving_fraction{{{rapec}}} * 100', "RAPEC-G vs Full100"), ("B", energy_rapec_es, "RAPEC-G vs ES"), ("C", f'live_shadow_energy_saving_fraction{{{es}}} * 100', "ES vs Full100")], "percent", 0, 117, 24),
        echarts(407, "Raw Quality by Dataset and Metric", "Dynamische Rohqualitätsansicht. Pro tatsächlich vorhandener und ausgewählter Rohmetrik wird ein eigenes Unterdiagramm erzeugt; inkompatible Skalen werden nie kombiniert. Gezeigt wird der beste Wert bis zum RAPEC-G-/ES-Stopp sowie der beste Full100-Wert. Für Loss, Cross-Entropy, Perplexity, MAE, RMSE und NRMSE gilt: kleiner ist besser.", [target("A", 'thesis_live_shadow_epoch_raw_quality{benchmark_run_id="$benchmark_run_id",case_id=~"$cases",task_type=~"$task_types",stage=~"$stages",raw_quality_metric=~"$raw_quality_metrics"}'), target("B", f'live_shadow_controller_stop_epoch{{{rapec}}}'), target("C", f'live_shadow_controller_stop_epoch{{{es}}}')], RAW_QUALITY_SCRIPT, 0, 128, 24, 28),
    ]

    panels.append(row(500, "5 · Energy–Quality Frontier", "Verknüpft reale Full100-Lernkurven, kontrafaktische Stopppunkte und Energieeffizienz der Qualitätsverbesserung.", 156))
    panels += [
        echarts(
            501,
            "Full100 Quality–Energy Curve with RAPEC-G and ES Stop Points",
            "Eine reale Full100-Linie je ausgewähltem Fall. RAPEC-G-v4- und Standard-ES-Informationen zeigen im Tooltip Stop-Epoche, Trainingsenergie, Qualität, Einsparung, Regret, Overhead und Compute Time.",
            [
                target("A", 'thesis_live_shadow_epoch_gpu_energy_wh{benchmark_run_id="$benchmark_run_id",case_id=~"$cases",task_type=~"$task_types",stage=~"$stages"}'),
                target("B", 'thesis_live_shadow_epoch_quality_score{benchmark_run_id="$benchmark_run_id",case_id=~"$cases",task_type=~"$task_types",stage=~"$stages"} * 100'),
                target("C", f'live_shadow_controller_stop_epoch{{{rapec}}}'),
                target("D", f'live_shadow_controller_stop_epoch{{{es}}}'),
                target("F", f'live_shadow_energy_saving_fraction{{{rapec}}} * 100'),
                target("G", f'live_shadow_quality_regret{{{rapec}}} * 100'),
                target("H", f'live_shadow_controller_overhead_energy_wh{{{rapec}}}'),
                target("I", f'live_shadow_controller_compute_seconds{{{rapec}}}'),
                target("K", f'live_shadow_energy_saving_fraction{{{es}}} * 100'),
                target("L", f'live_shadow_quality_regret{{{es}}} * 100'),
                target("M", f'live_shadow_controller_overhead_energy_wh{{{es}}}'),
                target("N", f'live_shadow_controller_compute_seconds{{{es}}}'),
            ],
            ENERGY_QUALITY_CURVE_SCRIPT,
            0,
            157,
            24,
            16,
        ),
        echarts(502, "GPU Energy per +1 Normalized Quality Point by Dataset", "Durchschnittlicher Energieaufwand je zusätzlichem normalisierten Qualitätsprozentpunkt: Energie bis zum Stop inklusive Overhead geteilt durch (Stopqualität − Qualität in Epoche 1) × 100. Full100 verwendet die vollständige Trainingsenergie. Kleinere Werte sind effizienter; nichtpositive Qualitätsgewinne bleiben undefiniert.", [target("A", f'live_shadow_full100_training_energy_wh{{{rapec}}}'), target("B", f'live_shadow_full100_best_quality{{{rapec}}}'), target("C", e_r), target("D", q_r), target("E", e_s), target("F", q_s), target("G", 'thesis_live_shadow_epoch_quality_score{benchmark_run_id="$benchmark_run_id",epoch="1",case_id=~"$cases",task_type=~"$task_types",stage=~"$stages"}')], ENERGY_PER_QUALITY_SCRIPT, 0, 173, 24, 16),
        echarts(503, "Energy Saving vs Quality Regret by Dataset", "Drei getrennte Teilgrafiken zeigen RAPEC-G v4 vs Full100, Standard ES vs Full100 und RAPEC-G v4 vs Standard ES. Jeder Punkt ist ein Fall; X ist die gerichtete Energieeinsparung, Y der gerichtete normalisierte Quality Regret.", [target("RX", f'live_shadow_energy_saving_fraction{{{rapec}}} * 100'), target("RY", f'live_shadow_quality_regret{{{rapec}}} * 100'), target("EX", f'live_shadow_energy_saving_fraction{{{es}}} * 100'), target("EY", f'live_shadow_quality_regret{{{es}}} * 100'), target("CX", energy_rapec_es), target("CY", regret_rapec_es)], TRADEOFF_SCRIPT, 0, 189, 24, 15),
    ]

    panels.append(row(600, "6 · Generalisation and Robustness", "Task-stratifizierte Effekte, Unsicherheit und konservative Erfolgskennzahlen.", 204))
    panels.append(echarts(601, "RAPEC-G Generalisation by Task Type (Mean and 95% CI)", "Makro-Mittel der RAPEC-G-Energieeinsparung und des Quality Regret je Task-Typ. Fehlerbalken sind deterministische 95-%-Bootstrap-Konfidenzintervalle mit 1.000 Resamples auf Dataset-Ebene; n steht im Tooltip.", [target("A", f'live_shadow_energy_saving_fraction{{{rapec}}} * 100'), target("B", f'live_shadow_quality_regret{{{rapec}}} * 100')], TASK_GENERALISATION_SCRIPT, 0, 205, 24, 16))
    robustness = [
        (602, "Selected Datasets", "Anzahl ausgewählter RAPEC-G-v4-Datasets mit vollständigem Stop-Ergebnis.", f'count(live_shadow_controller_stop_epoch{{{rapec}}})', "none"),
        (603, "Positive Energy Saving", "Anteil ausgewählter Datasets, bei denen RAPEC-G gegenüber Full100 positive Energieeinsparung erzielt.", f'avg(live_shadow_energy_saving_fraction{{{rapec}}} > bool 0) * 100', "percent"),
        (604, "Quality Within 10 pp", "Anteil ausgewählter Datasets mit höchstens 10 normalisierten Prozentpunkten Quality Regret gegenüber Full100.", f'avg(live_shadow_quality_regret{{{rapec}}} <= bool 0.10) * 100', "percent"),
        (605, "Joint Success", "Anteil ausgewählter Datasets mit positiver Energieeinsparung und Quality Regret von höchstens 10 Prozentpunkten.", f'avg((live_shadow_energy_saving_fraction{{{rapec}}} > bool 0) * ignoring(__name__) (live_shadow_quality_regret{{{rapec}}} <= bool 0.10)) * 100', "percent"),
        (606, "Worst Quality Regret", "Größter normalisierter RAPEC-G-Quality-Regret unter den ausgewählten Datasets.", f'max(live_shadow_quality_regret{{{rapec}}}) * 100', "percentpoint"),
        (607, "Minimum Energy Saving", "Niedrigste RAPEC-G-Energieeinsparung unter den ausgewählten Datasets; negative Werte kennzeichnen Mehrverbrauch.", f'min(live_shadow_energy_saving_fraction{{{rapec}}}) * 100', "percent"),
    ]
    for index, (pid, title, desc, expr, unit) in enumerate(robustness):
        panels.append(stat(pid, title, desc, expr, unit, (index % 3) * 8, 221 + (index // 3) * 5, 8, 0 if unit == "none" else 2))

    variables = [
        constant_variable("benchmark_run_id", default_run_id),
        variable("task_types", "Task Types", 'label_values(live_shadow_controller_stop_epoch{benchmark_run_id="$benchmark_run_id",controller_id="rapec-g-v4"}, task_type)'),
        variable("stages", "Stages", 'label_values(live_shadow_controller_stop_epoch{benchmark_run_id="$benchmark_run_id",controller_id="rapec-g-v4",task_type=~"$task_types"}, stage)'),
        variable("cases", "Datasets", 'label_values(live_shadow_controller_stop_epoch{benchmark_run_id="$benchmark_run_id",controller_id="rapec-g-v4",task_type=~"$task_types",stage=~"$stages"}, case_id)'),
        variable("raw_quality_metrics", "Raw Quality Metrics", 'label_values(thesis_live_shadow_epoch_raw_quality{benchmark_run_id="$benchmark_run_id",case_id=~"$cases"}, raw_quality_metric)'),
        custom_variable("slurm_phase", "SLURM Phase", ["preprocessing_initialization", "training", "finalization_evaluation"], "training"),
        custom_variable("codecarbon_phase", "CodeCarbon Phase", ["preprocessing_initialization", "training", "finalization_evaluation"], "training"),
    ]
    return {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "liveNow": False,
        "panels": panels,
        "refresh": "30s",
        "schemaVersion": 39,
        "tags": ["thesis", "appendix", "rapec-g-v4", "live-shadow", "48-real-96-virtual"],
        "templating": {"list": variables},
        "time": {"from": "now-30d", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "title": "Thesis Results – Energy-Adaptive MLOps",
        "uid": "rapec-g-v4-live-shadow-cv-nv",
        "version": 1,
        "weekStart": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--default-run-id", default="20260812T143435Z-live-shadow-dev")
    parser.add_argument("--default-curve-case", default="carpk-aerial-vehicle")
    parser.add_argument("--title", default="Thesis Results – Energy-Adaptive MLOps")
    parser.add_argument("--uid", default="rapec-g-v4-live-shadow-cv-nv")
    args = parser.parse_args()
    dashboard = build(args.default_run_id, args.default_curve_case)
    dashboard["title"] = args.title
    dashboard["uid"] = args.uid
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
