from __future__ import annotations

import argparse
import json
from pathlib import Path


DS = {"type": "prometheus", "uid": "prometheus"}


def target(ref: str, expr: str, legend: str, table: bool = False) -> dict:
    return {"refId": ref, "expr": expr, "legendFormat": legend, "instant": table, "range": not table, "format": "table" if table else "time_series", "editorMode": "code", "datasource": DS}


def stat(pid: int, title: str, expr: str, unit: str, x: int, y: int = 1) -> dict:
    unit_name = "percent" if unit == "percentunit" else unit
    return {"id": pid, "title": title, "type": "stat", "gridPos": {"x": x, "y": y, "w": 3, "h": 5}, "datasource": DS, "fieldConfig": {"defaults": {"unit": unit_name, "color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}}, "overrides": []}, "options": {"colorMode": "value", "graphMode": "none", "textMode": "value_and_name", "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}, "targets": [target("A", expr, title)]}


def row(pid: int, title: str, y: int) -> dict:
    return {"id": pid, "title": title, "type": "row", "collapsed": False, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}


def bars(pid: int, title: str, metric: str, unit: str, x: int, y: int, multiplier: str = "") -> dict:
    filters = 'benchmark_run_id="$benchmark_run_id",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases"'
    return {"id": pid, "title": title, "type": "barchart", "gridPos": {"x": x, "y": y, "w": 12, "h": 9}, "datasource": DS, "fieldConfig": {"defaults": {"unit": unit, "color": {"mode": "palette-classic"}, "custom": {"fillOpacity": 85, "lineWidth": 1, "stacking": {"group": "A", "mode": "none"}}}, "overrides": []}, "options": {"xField": "case_id", "orientation": "vertical", "xTickLabelRotation": 45, "stacking": "none", "groupWidth": 0.7, "barWidth": 0.25, "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}}, "targets": [target("A", f'{metric}{{{filters}}}{multiplier}', "{{variant}}", True)], "transformations": [{"id": "sortBy", "options": {"fields": {}, "sort": [{"field": "case_id", "desc": False}]}}, {"id": "groupingToMatrix", "options": {"rowField": "case_id", "columnField": "variant", "valueField": "Value", "emptyValue": "null"}}]}


def single_bars(pid: int, title: str, metric: str, unit: str, x: int, y: int, multiplier: str = "") -> dict:
    panel = bars(pid, title, metric, unit, x, y, multiplier)
    panel["targets"][0]["legendFormat"] = title
    panel["transformations"] = panel["transformations"][:1]
    return panel


def global_stat(pid: int, title: str, expr: str, unit: str, x: int, y: int) -> dict:
    panel = stat(pid, title, expr, unit, x, y)
    panel["gridPos"]["w"] = 3
    panel["gridPos"]["h"] = 4
    return panel


def task_lines(pid: int, title: str, targets: list[tuple[str, str, str]], x: int, y: int, task_type: str | None = None, width: int = 12, unit: str = "percentunit") -> dict:
    if task_type:
        filters = f'benchmark_run_id="$benchmark_run_id",task_type="{task_type}"'
    else:
        filters = 'benchmark_run_id="$benchmark_run_id",task_type=~"$task_types",benchmark_stage=~"$stages",benchmark_case_id=~"$cases"'
    return {
        "id": pid, "title": title, "type": "timeseries", "gridPos": {"x": x, "y": y, "w": width, "h": 9},
        "datasource": DS,
        "fieldConfig": {"defaults": {"unit": unit, "min": 0 if unit == "percentunit" else None, "max": 1 if unit == "percentunit" else None, "custom": {"drawStyle": "line", "lineWidth": 2, "showPoints": "always"}}, "overrides": []},
        "options": {"legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}},
        "targets": [target(ref, f'{metric}{{{filters}}}', legend) for ref, metric, legend in targets],
    }


ECHARTS_FRAME_HELPERS = r"""
const frames = (context && context.panel && context.panel.data && context.panel.data.series) || [];
const values = (field) => {
  if (!field || field.values == null) return [];
  if (typeof field.values.toArray === 'function') return field.values.toArray();
  if (field.values.buffer) return Array.from(field.values.buffer);
  return Array.isArray(field.values) ? field.values : [];
};
const numericField = (frame) => (frame.fields || []).find((field) => field.type === 'number' || field.name === 'Value');
const pointValue = (frame) => {
  const field = numericField(frame);
  const data = values(field);
  return data.length ? Number(data[data.length - 1]) : null;
};
const pointLabels = (frame) => (numericField(frame) && numericField(frame).labels) || {};
"""


def echarts_target(ref: str, expr: str) -> dict:
    item = target(ref, expr, "{{variant}} / Job {{job_id}}")
    item.update({"instant": True, "range": False, "format": "time_series"})
    return item


def energy_quality_scatter(
    pid: int, title: str, x: int, y: int, task_type: str | None = None, width: int = 12
) -> dict:
    filters = 'benchmark_run_id="$benchmark_run_id",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases"'
    if task_type:
        filters += f',task_type="{task_type}"'
    script = ECHARTS_FRAME_HELPERS + r"""
const points = new Map();
frames.forEach((frame) => {
  const labels = pointLabels(frame);
  const value = pointValue(frame);
  if (!Number.isFinite(value) || !labels.job_id || !labels.variant) return;
  const key = `${labels.variant}|${labels.job_id}`;
  const point = points.get(key) || { job: labels.job_id, variant: labels.variant, task: labels.task_type || 'unknown' };
  if (frame.refId === 'A') point.energy = value;
  if (frame.refId === 'B') point.quality = value;
  points.set(key, point);
});
const colors = { object_detection: '#4C78A8', image_classification: '#F58518', semantic_segmentation: '#54A24B' };
const symbols = { Full100: 'circle', Candidate: 'diamond' };
const grouped = new Map();
Array.from(points.values()).filter((p) => Number.isFinite(p.energy) && Number.isFinite(p.quality)).forEach((p) => {
  const name = `${p.task} / ${p.variant}`;
  if (!grouped.has(name)) grouped.set(name, []);
  grouped.get(name).push(p);
});
return {
  animation: false,
  tooltip: {
    trigger: 'item',
    formatter: (p) => `${p.data.job}<br/>Task: ${p.data.task}<br/>Variant: ${p.data.variant}<br/>Energy: ${p.data.value[0].toFixed(3)} Wh<br/>Quality: ${p.data.value[1].toFixed(2)} %`
  },
  legend: { type: 'scroll', bottom: 0 },
  grid: { left: 75, right: 30, top: 35, bottom: 90 },
  xAxis: { type: 'value', name: 'SLURM GPU Energy (Wh)', nameLocation: 'middle', nameGap: 48, min: 0 },
  yAxis: { type: 'value', name: 'Quality (%)', nameLocation: 'middle', nameGap: 55, min: 0, max: 100 },
  series: Array.from(grouped.entries()).map(([name, data]) => ({
    name,
    type: 'scatter',
    symbol: symbols[data[0].variant] || 'circle',
    symbolSize: data[0].variant === 'Full100' ? 16 : 18,
    itemStyle: { color: colors[data[0].task] || '#999' },
    label: { show: true, position: 'top', formatter: (p) => p.data.job },
    data: data.map((p) => ({ value: [p.energy, p.quality], job: `Job ${p.job}`, task: p.task, variant: p.variant }))
  }))
};
"""
    return {
        "id": pid,
        "title": title,
        "type": "volkovlabs-echarts-panel",
        "gridPos": {"x": x, "y": y, "w": width, "h": 11},
        "datasource": DS,
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {"renderer": "svg", "followTheme": True, "getOption": script},
        "targets": [
            echarts_target("A", f"controller_benchmark_energy_wh{{{filters}}}"),
            echarts_target("B", f"controller_benchmark_best_quality{{{filters}}} * 100"),
        ],
    }


def mirrored_energy_chart(pid: int, x: int, y: int, width: int = 24) -> dict:
    filters = 'benchmark_run_id="$benchmark_run_id",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases"'
    script = ECHARTS_FRAME_HELPERS + r"""
const jobs = new Map();
frames.forEach((frame) => {
  const labels = pointLabels(frame);
  const value = pointValue(frame);
  if (!Number.isFinite(value) || !labels.job_id || !labels.variant) return;
  const key = `${labels.variant}|${labels.job_id}`;
  const row = jobs.get(key) || { job: labels.job_id, variant: labels.variant, task: labels.task_type || 'unknown' };
  if (frame.refId === 'A') row.slurm = value;
  if (frame.refId === 'B') row.codecarbon = value;
  jobs.set(key, row);
});
const numeric = (value) => {
  const match = String(value).match(/\d+/);
  return match ? Number(match[0]) : Number.MAX_SAFE_INTEGER;
};
const rows = Array.from(jobs.values())
  .filter((row) => Number.isFinite(row.slurm) || Number.isFinite(row.codecarbon))
  .sort((a, b) => numeric(a.job) - numeric(b.job) || a.variant.localeCompare(b.variant));
const categories = rows.map((row) => `Job ${row.job}\n${row.variant}`);
return {
  animation: false,
  legend: { bottom: 0 },
  tooltip: {
    trigger: 'axis',
    axisPointer: { type: 'shadow' },
    formatter: (items) => {
      const row = rows[items[0].dataIndex];
      return `Job ${row.job} (${row.variant})<br/>Task: ${row.task}<br/>SLURM GPU: ${Math.abs(row.slurm || 0).toFixed(3)} Wh<br/>CodeCarbon GPU: ${Math.abs(row.codecarbon || 0).toFixed(3)} Wh`;
    }
  },
  grid: { left: 75, right: 30, top: 35, bottom: 105 },
  xAxis: { type: 'category', data: categories, axisLabel: { interval: 0, rotate: 45, fontSize: 10 } },
  yAxis: {
    type: 'value',
    name: 'GPU Energy (Wh)',
    axisLabel: { formatter: (value) => Math.abs(value) },
    splitLine: { lineStyle: { color: '#666', opacity: 0.25 } }
  },
  series: [
    {
      name: 'SLURM GPU Energy (above)',
      type: 'bar',
      barMaxWidth: 22,
      itemStyle: { color: '#4C78A8' },
      label: { show: true, position: 'top', formatter: (p) => Math.abs(p.value).toFixed(2) },
      data: rows.map((row) => Number.isFinite(row.slurm) ? Math.abs(row.slurm) : null)
    },
    {
      name: 'CodeCarbon GPU Energy (below)',
      type: 'bar',
      barMaxWidth: 22,
      itemStyle: { color: '#E45756' },
      label: { show: true, position: 'bottom', formatter: (p) => Math.abs(p.value).toFixed(2) },
      data: rows.map((row) => Number.isFinite(row.codecarbon) ? -Math.abs(row.codecarbon) : null)
    }
  ]
};
"""
    return {
        "id": pid,
        "title": "SLURM vs CodeCarbon Training GPU Energy by Job (Mirrored Wh)",
        "description": "Both methods use the same training-phase boundary. CodeCarbon is plotted below zero only for visual comparison; both measurements are positive GPU energy values.",
        "type": "volkovlabs-echarts-panel",
        "gridPos": {"x": x, "y": y, "w": width, "h": 12},
        "datasource": DS,
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {"renderer": "svg", "followTheme": True, "getOption": script},
        "targets": [
            echarts_target("A", f"controller_benchmark_comparable_slurm_gpu_energy_wh{{{filters}}}"),
            echarts_target("B", f"controller_benchmark_comparable_codecarbon_gpu_energy_wh{{{filters}}}"),
        ],
    }


def phase_energy_quality_chart(pid: int, x: int, y: int, width: int = 24) -> dict:
    filters = 'benchmark_run_id="$benchmark_run_id",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases",phase=~"$phases"'
    base_filters = 'benchmark_run_id="$benchmark_run_id",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases"'
    script = ECHARTS_FRAME_HELPERS + r"""
const rows = new Map();
const phaseSet = new Set();
const keyOf = (labels) => `${labels.case_id || 'unknown'}|${labels.variant || 'unknown'}`;
frames.forEach((frame) => {
  const labels = pointLabels(frame);
  const value = pointValue(frame);
  if (!Number.isFinite(value) || !labels.case_id || !labels.variant) return;
  const key = keyOf(labels);
  const row = rows.get(key) || {
    case_id: labels.case_id,
    variant: labels.variant,
    task: labels.task_type || 'unknown',
    job: labels.job_id || '',
    phase: {},
  };
  if (labels.phase) {
    phaseSet.add(labels.phase);
    row.phase[labels.phase] = row.phase[labels.phase] || {};
    if (frame.refId === 'A') row.phase[labels.phase].energy = value;
    if (frame.refId === 'B') row.phase[labels.phase].cost = value;
    if (frame.refId === 'C') row.phase[labels.phase].co2 = value;
  } else {
    if (frame.refId === 'D') row.quality = value * 100;
    if (frame.refId === 'E') row.epochs = value;
  }
  rows.set(key, row);
});
const orderValue = (row) => `${row.case_id}|${row.variant === 'Full100' ? '0' : '1'}`;
const dataRows = Array.from(rows.values()).sort((a, b) => orderValue(a).localeCompare(orderValue(b)));
const phases = Array.from(phaseSet).sort((a, b) => {
  const order = { preprocessing_labels: 1, preprocessing_split: 2, preprocessing_crop: 3, training: 4, other: 5 };
  return (order[a] || 99) - (order[b] || 99) || a.localeCompare(b);
});
const phaseLabel = (phase) => ({
  preprocessing_labels: 'Preprocessing Labels',
  preprocessing_split: 'Preprocessing Split',
  preprocessing_crop: 'Preprocessing Crop',
  training: 'Training',
  other: 'Other',
})[phase] || phase;
const categories = dataRows.map((row) => `${row.case_id}\n${row.variant}`);
const colors = ['#2E8B57', '#6BBF8E', '#B3E0C2', '#1F6B44', '#8CC9A5'];
const costColors = ['#C47F00', '#D99A22', '#E8B85E', '#A86600', '#F0CF91'];
const co2Colors = ['#B24C63', '#C9677B', '#DA8D9C', '#8B3448', '#E8B3BE'];
const series = [];
phases.forEach((phase, idx) => {
  series.push({
    name: `Wh / ${phaseLabel(phase)}`,
    type: 'bar',
    stack: 'energy',
    yAxisIndex: 0,
    barMaxWidth: 18,
    itemStyle: { color: colors[idx % colors.length] },
    data: dataRows.map((row) => row.phase[phase]?.energy ?? 0),
  });
  series.push({
    name: `EUR / ${phaseLabel(phase)}`,
    type: 'bar',
    stack: 'cost',
    yAxisIndex: 1,
    barMaxWidth: 18,
    itemStyle: { color: costColors[idx % costColors.length] },
    data: dataRows.map((row) => row.phase[phase]?.cost ?? 0),
  });
  series.push({
    name: `CO2 / ${phaseLabel(phase)}`,
    type: 'bar',
    stack: 'co2',
    yAxisIndex: 2,
    barMaxWidth: 18,
    itemStyle: { color: co2Colors[idx % co2Colors.length] },
    data: dataRows.map((row) => row.phase[phase]?.co2 ?? 0),
  });
});
series.push({
  name: 'Quality (%)',
  type: 'line',
  yAxisIndex: 3,
  symbol: 'circle',
  symbolSize: 7,
  lineStyle: { width: 3, color: '#4C78A8' },
  itemStyle: { color: '#4C78A8' },
  data: dataRows.map((row) => Number.isFinite(row.quality) ? row.quality : null),
});
series.push({
  name: 'Epoch Count',
  type: 'line',
  yAxisIndex: 4,
  symbol: 'diamond',
  symbolSize: 8,
  lineStyle: { width: 2, type: 'dashed', color: '#666666' },
  itemStyle: { color: '#666666' },
  data: dataRows.map((row) => Number.isFinite(row.epochs) ? row.epochs : null),
});
return {
  animation: false,
  legend: { type: 'scroll', bottom: 0 },
  tooltip: {
    trigger: 'axis',
    axisPointer: { type: 'shadow' },
    formatter: (items) => {
      const row = dataRows[items[0].dataIndex];
      const lines = [`${row.case_id} (${row.variant})`, `Job: ${row.job || '-'}`, `Task: ${row.task}`];
      items.forEach((item) => {
        if (item.value == null) return;
        const unit = item.seriesName.startsWith('Wh') ? ' Wh' : item.seriesName.startsWith('EUR') ? ' EUR' : item.seriesName.startsWith('CO2') ? ' kg' : '';
        lines.push(`${item.marker}${item.seriesName}: ${Number(item.value).toFixed(3)}${unit}`);
      });
      return lines.join('<br/>');
    },
  },
  grid: { left: 75, right: 245, top: 35, bottom: 155 },
  xAxis: { type: 'category', data: categories, axisLabel: { interval: 0, rotate: 45, fontSize: 10 } },
  yAxis: [
    { type: 'value', name: 'Wh', position: 'left' },
    { type: 'value', name: 'EUR', position: 'right' },
    { type: 'value', name: 'kg', position: 'right', offset: 55 },
    { type: 'value', name: 'Quality %', position: 'right', offset: 110, min: 0, max: 100 },
    { type: 'value', name: 'epochs', position: 'right', offset: 180, minInterval: 1 },
  ],
  series,
};
"""
    return {
        "id": pid,
        "title": "SLURM Phase: Energy, Cost, CO2, Quality, and Epoch Count by Job/Case",
        "description": "Case names are shown directly on the X-axis. Energy, cost, and CO2 are stacked by phase; quality and epoch count are shown as lines.",
        "type": "volkovlabs-echarts-panel",
        "gridPos": {"x": x, "y": y, "w": width, "h": 15},
        "datasource": DS,
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {"renderer": "svg", "followTheme": True, "getOption": script},
        "targets": [
            echarts_target("A", f"controller_benchmark_phase_gpu_energy_wh{{{filters}}}"),
            echarts_target("B", f"controller_benchmark_phase_cost_eur{{{filters}}}"),
            echarts_target("C", f"controller_benchmark_phase_co2_kg{{{filters}}}"),
            echarts_target("D", f"controller_benchmark_best_quality{{{base_filters}}}"),
            echarts_target("E", f"controller_benchmark_epochs{{{base_filters}}}"),
        ],
    }


def phase_duration_chart(pid: int, x: int, y: int, width: int = 24) -> dict:
    filters = 'benchmark_run_id="$benchmark_run_id",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases",phase=~"$phases"'
    script = ECHARTS_FRAME_HELPERS + r"""
const rows = new Map();
const phaseSet = new Set();
frames.forEach((frame) => {
  const labels = pointLabels(frame);
  const value = pointValue(frame);
  if (!Number.isFinite(value) || !labels.case_id || !labels.variant || !labels.phase) return;
  const key = `${labels.case_id}|${labels.variant}`;
  const row = rows.get(key) || { case_id: labels.case_id, variant: labels.variant, task: labels.task_type || 'unknown', job: labels.job_id || '', phase: {} };
  phaseSet.add(labels.phase);
  row.phase[labels.phase] = value;
  rows.set(key, row);
});
const phaseLabel = (phase) => ({
  preprocessing_labels: 'Preprocessing Labels',
  preprocessing_split: 'Preprocessing Split',
  preprocessing_crop: 'Preprocessing Crop',
  training: 'Training',
  other: 'Other',
})[phase] || phase;
const phases = Array.from(phaseSet).sort((a, b) => {
  const order = { preprocessing_labels: 1, preprocessing_split: 2, preprocessing_crop: 3, training: 4, other: 5 };
  return (order[a] || 99) - (order[b] || 99) || a.localeCompare(b);
});
const rowsList = Array.from(rows.values()).sort((a, b) => `${a.case_id}|${a.variant === 'Full100' ? '0' : '1'}`.localeCompare(`${b.case_id}|${b.variant === 'Full100' ? '0' : '1'}`));
const categories = rowsList.map((row) => `${row.case_id}\n${row.variant}`);
const colors = ['#7A5195', '#BC5090', '#EF5675', '#FF764A', '#FFA600'];
return {
  animation: false,
  legend: { type: 'scroll', bottom: 0 },
  tooltip: {
    trigger: 'axis',
    axisPointer: { type: 'shadow' },
    formatter: (items) => {
      const row = rowsList[items[0].dataIndex];
      const lines = [`${row.case_id} (${row.variant})`, `Job: ${row.job || '-'}`, `Task: ${row.task}`];
      items.forEach((item) => lines.push(`${item.marker}${item.seriesName}: ${Number(item.value || 0).toFixed(2)} s`));
      return lines.join('<br/>');
    }
  },
  grid: { left: 75, right: 35, top: 35, bottom: 140 },
  xAxis: { type: 'category', data: categories, axisLabel: { interval: 0, rotate: 45, fontSize: 10 } },
  yAxis: { type: 'value', name: 'seconds' },
  series: phases.map((phase, idx) => ({
    name: phaseLabel(phase),
    type: 'bar',
    stack: 'duration',
    barMaxWidth: 24,
    itemStyle: { color: colors[idx % colors.length] },
    data: rowsList.map((row) => row.phase[phase] ?? 0),
  })),
};
"""
    return {
        "id": pid,
        "title": "Phase Duration by Job/Case (s)",
        "type": "volkovlabs-echarts-panel",
        "gridPos": {"x": x, "y": y, "w": width, "h": 11},
        "datasource": DS,
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {"renderer": "svg", "followTheme": True, "getOption": script},
        "targets": [echarts_target("A", f"controller_benchmark_phase_duration_seconds{{{filters}}}")],
    }


def phase_tabs(pid: int, x: int, y: int, width: int = 24) -> dict:
    content = """
<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;padding:12px 0;min-height:64px;">
  <a href="?var-phases=preprocessing_labels" style="padding:11px 16px;border-radius:999px;border:1px solid #c47f00;background:#fff4df;color:#7a4f00;text-decoration:none;font-weight:700;">Preprocessing Labels</a>
  <a href="?var-phases=preprocessing_split" style="padding:11px 16px;border-radius:999px;border:1px solid #b24c63;background:#fdecef;color:#7e2d40;text-decoration:none;font-weight:700;">Preprocessing Split</a>
  <a href="?var-phases=training" style="padding:11px 16px;border-radius:999px;border:1px solid #2E8B57;background:#eef7f1;color:#184d33;text-decoration:none;font-weight:700;">Training</a>
  <a href="?var-phases=other" style="padding:11px 16px;border-radius:999px;border:1px solid #5b7c99;background:#eef3f8;color:#28465f;text-decoration:none;font-weight:700;">Other</a>
  <a href="?var-phases=All" style="padding:11px 16px;border-radius:999px;border:1px solid #777;background:#f1f1f1;color:#333;text-decoration:none;font-weight:700;">All Phases</a>
</div>
"""
    return {
        "id": pid,
        "title": "SLURM Phase Tabs",
        "type": "text",
        "gridPos": {"x": x, "y": y, "w": width, "h": 4},
        "transparent": True,
        "options": {"mode": "html", "content": content},
    }


def grouped_case_bars(pid: int, title: str, metric: str, unit: str, x: int, y: int, multiplier: str = "") -> dict:
    filters = 'benchmark_run_id="$benchmark_run_id",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases"'
    return {
        "id": pid,
        "title": title,
        "type": "barchart",
        "gridPos": {"x": x, "y": y, "w": 12, "h": 9},
        "datasource": DS,
        "fieldConfig": {"defaults": {"unit": unit, "color": {"mode": "palette-classic"}, "custom": {"fillOpacity": 85, "lineWidth": 1, "stacking": {"group": "A", "mode": "none"}}}, "overrides": []},
        "options": {"xField": "case_id", "orientation": "vertical", "xTickLabelRotation": 45, "stacking": "none", "groupWidth": 0.7, "barWidth": 0.22, "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}},
        "targets": [target("A", f'{metric}{{{filters}}}{multiplier}', "{{variant}}", True)],
        "transformations": [
            {"id": "sortBy", "options": {"fields": {}, "sort": [{"field": "case_id", "desc": False}]}},
            {"id": "groupingToMatrix", "options": {"rowField": "case_id", "columnField": "variant", "valueField": "Value", "emptyValue": "null"}},
        ],
    }


def variable(name: str, label: str, query: str, multi: bool = True) -> dict:
    return {"name": name, "label": label, "type": "query", "datasource": DS, "definition": query, "query": {"query": query, "refId": "StandardVariableQuery"}, "refresh": 1, "sort": 1, "multi": multi, "includeAll": multi, "allValue": ".*" if multi else None, "current": {"selected": multi, "text": ["All"] if multi else "", "value": ["$__all"] if multi else ""}, "options": []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = 'benchmark_run_id="$benchmark_run_id",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases"'
    panels = [
        row(100, "Executive Summary", 0),
        stat(1, "Selected Pairs", f"count(controller_benchmark_case_info{{{selected}}})", "none", 0),
        stat(2, "Energy Success", f"avg(controller_benchmark_energy_target_met{{{selected}}}) * 100", "percent", 3),
        stat(3, "Quality Success", f"avg(controller_benchmark_quality_target_met{{{selected}}}) * 100", "percent", 6),
        stat(4, "Joint Success", f"avg(controller_benchmark_energy_target_met{{{selected}}} * controller_benchmark_quality_target_met{{{selected}}}) * 100", "percent", 9),
        stat(5, "Mean Energy Saving", f"avg(controller_benchmark_energy_saving_fraction{{{selected}}}) * 100", "percent", 12),
        stat(6, "Mean Quality Regret", f"avg(controller_benchmark_quality_regret{{{selected}}}) * 100", "percentpoint", 15),
        stat(7, "Overall Benchmark Passed", 'controller_benchmark_passed{benchmark_run_id="$benchmark_run_id"}', "bool", 18),
        stat(8, "Overall Generalisation Claim", 'controller_benchmark_full_generalisation_claim_available{benchmark_run_id="$benchmark_run_id"}', "bool", 21),
        row(105, "Global Operational Totals", 6),
        global_stat(33, "Total Jobs", f"count(controller_benchmark_energy_wh{{{selected}}})", "none", 0, 7),
        global_stat(34, "SLURM Global Total GPU Energy (Wh)", f"sum(controller_benchmark_job_energy_wh{{{selected}}}) or sum(controller_benchmark_energy_wh{{{selected}}})", "watth", 3, 7),
        global_stat(35, "CodeCarbon Global Total GPU Energy (Wh)", f"sum(controller_benchmark_codecarbon_gpu_energy_wh{{{selected}}})", "watth", 6, 7),
        global_stat(36, "CodeCarbon Global Total Energy (Wh)", f"sum(controller_benchmark_codecarbon_total_energy_wh{{{selected}}})", "watth", 9, 7),
        global_stat(37, "SLURM Global Total Cost (EUR)", f"sum(controller_benchmark_cost_eur{{{selected}}})", "currencyEUR", 12, 7),
        global_stat(38, "CodeCarbon Global Total Cost (EUR)", f"sum(controller_benchmark_codecarbon_cost_eur{{{selected}}})", "currencyEUR", 15, 7),
        global_stat(39, "SLURM Global Total CO2 (kg)", f"sum(controller_benchmark_co2_kg{{{selected}}})", "kg", 18, 7),
        global_stat(40, "CodeCarbon Global Total CO2 (kg)", f"sum(controller_benchmark_codecarbon_co2_kg{{{selected}}})", "kg", 21, 7),
        row(102, "Energy-Quality Trade-off and Measurement Comparison", 12),
        energy_quality_scatter(20, "Task-Independent Energy vs Normalized Quality by Job", 0, 13, width=24),
        mirrored_energy_chart(24, 0, 24),
        energy_quality_scatter(21, "Object Detection: GPU Energy vs mAP50-95 by Job", 0, 36, "object_detection"),
        energy_quality_scatter(22, "Image Classification: GPU Energy vs Macro-F1 by Job", 12, 36, "image_classification"),
        energy_quality_scatter(23, "Semantic Segmentation: GPU Energy vs mIoU by Job", 0, 47, "semantic_segmentation", width=24),
        row(106, "Pairwise Controller Effects", 58),
        bars(10, "Full100 vs Candidate GPU Energy", "controller_benchmark_energy_wh", "watth", 0, 59),
        bars(12, "Full100 vs Candidate Normalized Quality", "controller_benchmark_best_quality", "percent", 12, 59, " * 100"),
        single_bars(11, "Energy Saving by Case", "controller_benchmark_energy_saving_fraction", "percent", 0, 68, " * 100"),
        single_bars(13, "Normalized Quality Regret by Case", "controller_benchmark_quality_regret", "percentpoint", 12, 68, " * 100"),
        bars(14, "Completed Epochs", "controller_benchmark_epochs", "none", 0, 77),
        grouped_case_bars(29, "Full100 vs Candidate Duration by Case (s)", "controller_benchmark_duration_seconds", "s", 12, 77),
        single_bars(31, "Epochs Saved by Case", "controller_benchmark_epochs_saved", "none", 0, 86),
        single_bars(32, "Duration Saving by Case (%)", "controller_benchmark_duration_saving_fraction", "percent", 12, 86, " * 100"),
        grouped_case_bars(30, "CodeCarbon GPU Energy by Case (Wh)", "controller_benchmark_codecarbon_gpu_energy_wh", "watth", 0, 95),
        single_bars(15, "Energy Target Met", "controller_benchmark_energy_target_met", "bool", 12, 95),
        row(101, "Learning Curves and Task-Specific Quality", 104),
        task_lines(16, "Task-Independent Quality Q_t", [("A", "slurm_job_epoch_quality_score", "{{task_type}} / {{benchmark_case_id}}")], 0, 105, width=24),
        task_lines(17, "Object Detection: mAP50, mAP50-95, Precision, and Recall", [("A", "slurm_job_epoch_map50", "mAP50 / {{benchmark_case_id}}"), ("B", "slurm_job_epoch_map50_95", "mAP50-95 / {{benchmark_case_id}}"), ("C", "slurm_job_epoch_precision", "Precision / {{benchmark_case_id}}"), ("D", "slurm_job_epoch_recall", "Recall / {{benchmark_case_id}}")], 0, 114, "object_detection"),
        task_lines(18, "Image Classification: Accuracy and Macro-F1", [("A", "slurm_job_epoch_accuracy", "Accuracy / {{benchmark_case_id}}"), ("B", "slurm_job_epoch_macro_f1", "Macro-F1 / {{benchmark_case_id}}")], 12, 114, "image_classification"),
        task_lines(19, "Semantic Segmentation: mIoU and Dice", [("A", "slurm_job_epoch_miou", "mIoU / {{benchmark_case_id}}"), ("B", "slurm_job_epoch_dice", "Dice / {{benchmark_case_id}}")], 0, 123, "semantic_segmentation", width=24),
        row(103, "Phase-Level Operational View", 132),
        phase_tabs(42, 0, 133),
        phase_energy_quality_chart(28, 0, 137),
        phase_duration_chart(41, 0, 152),
        row(104, "Controller Probabilistic Diagnostics", 164),
        task_lines(
            25,
            "PEP Expected Gain and Probability by Epoch",
            [
                ("A", "slurm_job_epoch_controller_plugin_pep_expected_quality_gain_next_horizon", "Expected gain / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_pep_prob_gain_gt_threshold", "P(gain > threshold) / {{benchmark_case_id}}"),
            ],
            0,
            165,
            width=12,
        ),
        task_lines(
            26,
            "PEP Expected Energy and Utility by Epoch",
            [
                ("A", "slurm_job_epoch_controller_plugin_pep_expected_energy_next_horizon_wh", "Expected energy Wh / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_pep_utility_quality_per_wh", "Utility Q/Wh / {{benchmark_case_id}}"),
            ],
            12,
            165,
            width=12,
            unit="none",
        ),
        task_lines(
            27,
            "PEP Feature Coverage and Stop Evidence",
            [
                ("A", "slurm_job_epoch_controller_plugin_pep_feature_coverage_fraction", "Feature coverage / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_pep_low_value_streak", "Low-value streak / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_pep_candidate_stop", "Candidate stop / {{benchmark_case_id}}"),
            ],
            0,
            174,
            width=24,
            unit="none",
        ),
        task_lines(
            43,
            "RAPEC-v3 Expected Gain and Dynamic Threshold by Epoch",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec_expected_quality_gain_next_horizon", "Expected gain / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec_dynamic_meaningful_gain", "Dynamic meaningful gain / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec_prob_gain_gt_dynamic_threshold", "P(gain > dynamic threshold) / {{benchmark_case_id}}"),
            ],
            0,
            183,
            width=24,
            unit="none",
        ),
        task_lines(
            44,
            "RAPEC-v3 Energy Utility and Adaptive Threshold by Epoch",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec_expected_energy_next_horizon_wh", "Expected energy Wh / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec_utility_quality_per_wh", "Utility Q/Wh / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec_dynamic_utility_threshold", "Dynamic utility threshold / {{benchmark_case_id}}"),
            ],
            0,
            192,
            width=24,
            unit="none",
        ),
        task_lines(
            45,
            "RAPEC-v3 Learning State and Stop Evidence by Epoch",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec_learning_state_code", "Learning state code / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec_low_value_streak", "Low-value streak / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec_candidate_stop", "Candidate stop / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec_uncertainty_ok", "Uncertainty actionable / {{benchmark_case_id}}"),
            ],
            0,
            201,
            width=24,
            unit="none",
        ),
        task_lines(
            46,
            "RAPEC-v4 Task-Aware Adapter Settings by Epoch",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec4_adapter_horizon_epochs", "Horizon / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec4_adapter_patience", "Patience / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec4_adapter_probability_threshold", "Probability threshold / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec4_adapter_task_family_code", "Task family / {{benchmark_case_id}}"),
            ],
            0,
            210,
            width=24,
            unit="none",
        ),
        task_lines(
            47,
            "RAPEC-v4 Dynamic Risk and Utility Thresholds by Epoch",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec_dynamic_meaningful_gain", "Dynamic gain / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec_dynamic_utility_threshold", "Dynamic utility threshold / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec4_adapter_uncertainty_to_gain_ratio", "Uncertainty ratio / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec4_adapter_minimum_quality_floor", "Quality floor / {{benchmark_case_id}}"),
            ],
            0,
            219,
            width=24,
            unit="none",
        ),
        task_lines(
            48,
            "RAPEC-v5 Self-Calibration Settings by Epoch",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec5_horizon_epochs", "Horizon / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec5_patience", "Patience / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec5_probability_threshold", "Probability threshold / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec5_utility_quantile", "Utility quantile / {{benchmark_case_id}}"),
            ],
            0,
            228,
            width=24,
            unit="none",
        ),
        task_lines(
            49,
            "RAPEC-v5 Curve Risk, Stability, and Energy Variability by Epoch",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec5_delayed_learning_risk", "Delayed-learning risk / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec5_curve_stability", "Curve stability / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec5_saturation_score", "Saturation score / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec5_energy_variability", "Energy variability / {{benchmark_case_id}}"),
            ],
            0,
            237,
            width=24,
            unit="none",
        ),
        row(105, "RAPEC-v6 Online Multi-Horizon Pareto Diagnostics", 246),
        task_lines(
            50,
            "RAPEC-v6 Multi-Horizon Expected Quality Gain by Epoch",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec6_h1_expected_gain", "1 epoch / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec6_h3_expected_gain", "3 epochs / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec6_h5_expected_gain", "5 epochs / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec6_h10_expected_gain", "10 epochs / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec6_h20_expected_gain", "20 epochs / {{benchmark_case_id}}"),
                ("F", "slurm_job_epoch_controller_plugin_rapec6_dynamic_epsilon", "Dynamic epsilon / {{benchmark_case_id}}"),
            ],
            0,
            247,
            width=24,
            unit="none",
        ),
        task_lines(
            51,
            "RAPEC-v6 Risk, Recovery, and Pareto Stop Evidence",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec6_risk_probability", "Future-gain risk / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec6_dynamic_risk_alpha", "Risk limit / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec6_recovery_probability", "Recovery probability / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec6_recovery_limit", "Recovery limit / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec6_valuable_horizon_count", "Valuable horizons / {{benchmark_case_id}}"),
                ("F", "slurm_job_epoch_controller_plugin_rapec6_candidate_stop", "Candidate stop / {{benchmark_case_id}}"),
            ],
            0,
            256,
            width=24,
            unit="none",
        ),
        task_lines(
            52,
            "RAPEC-v6 Predicted Training Energy by Horizon (Wh)",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec6_h1_expected_energy_wh", "1 epoch / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec6_h3_expected_energy_wh", "3 epochs / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec6_h5_expected_energy_wh", "5 epochs / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec6_h10_expected_energy_wh", "10 epochs / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec6_h20_expected_energy_wh", "20 epochs / {{benchmark_case_id}}"),
            ],
            0,
            265,
            width=12,
            unit="Wh",
        ),
        task_lines(
            53,
            "RAPEC-v6 Online Uncertainty Calibration",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec6_h5_calibration_radius", "5-epoch radius / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec6_h10_calibration_radius", "10-epoch radius / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec6_h20_calibration_radius", "20-epoch radius / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec6_feature_coverage_fraction", "Feature coverage / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec6_curve_stability", "Curve stability / {{benchmark_case_id}}"),
            ],
            12,
            265,
            width=12,
            unit="none",
        ),
        row(106, "RAPEC-v7 Online Bayesian Diagnostics", 274),
        task_lines(
            54,
            "RAPEC-v7 Posterior Quality Gain by Horizon",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec7_h1_expected_gain", "1 epoch / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec7_h3_expected_gain", "3 epochs / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec7_h5_expected_gain", "5 epochs / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec7_h10_expected_gain", "10 epochs / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec7_h20_expected_gain", "20 epochs / {{benchmark_case_id}}"),
                ("F", "slurm_job_epoch_controller_plugin_rapec7_dynamic_epsilon", "Dynamic detectable gain / {{benchmark_case_id}}"),
            ],
            0,
            275,
            width=24,
            unit="none",
        ),
        task_lines(
            55,
            "RAPEC-v7 Posterior Probability and Dynamic Stop Evidence",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec7_h5_prob_relevant_gain", "P(relevant gain, 5 epochs) / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec7_h10_prob_relevant_gain", "P(relevant gain, 10 epochs) / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec7_h20_prob_relevant_gain", "P(relevant gain, 20 epochs) / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec7_dynamic_risk_limit", "Dynamic probability limit / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec7_posterior_recovery_probability", "Posterior recovery probability / {{benchmark_case_id}}"),
                ("F", "slurm_job_epoch_controller_plugin_rapec7_candidate_stop", "Candidate stop / {{benchmark_case_id}}"),
            ],
            0,
            284,
            width=24,
            unit="none",
        ),
        task_lines(
            56,
            "RAPEC-v7 Posterior Training Energy by Horizon (Wh)",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec7_h1_expected_energy_wh", "1 epoch / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec7_h3_expected_energy_wh", "3 epochs / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec7_h5_expected_energy_wh", "5 epochs / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec7_h10_expected_energy_wh", "10 epochs / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec7_h20_expected_energy_wh", "20 epochs / {{benchmark_case_id}}"),
            ],
            0,
            293,
            width=12,
            unit="Wh",
        ),
        task_lines(
            57,
            "RAPEC-v7 Posterior Training Duration by Horizon (s)",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec7_h1_expected_duration_seconds", "1 epoch / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec7_h3_expected_duration_seconds", "3 epochs / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec7_h5_expected_duration_seconds", "5 epochs / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec7_h10_expected_duration_seconds", "10 epochs / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec7_h20_expected_duration_seconds", "20 epochs / {{benchmark_case_id}}"),
            ],
            12,
            293,
            width=12,
            unit="s",
        ),
        task_lines(
            58,
            "RAPEC-v7 Bayesian Model Weights and Uncertainty",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec7_quality_model_weight_linear", "Linear weight / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec7_quality_model_weight_logarithmic", "Logarithmic weight / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec7_quality_model_weight_saturation", "Saturation weight / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec7_quality_model_weight_exponential", "Exponential weight / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec7_quality_model_weight_change_point", "Change-point weight / {{benchmark_case_id}}"),
                ("G", "slurm_job_epoch_controller_plugin_rapec7_quality_model_weight_local_trend", "Local-trend weight / {{benchmark_case_id}}"),
                ("F", "slurm_job_epoch_controller_plugin_rapec7_model_disagreement", "Model disagreement / {{benchmark_case_id}}"),
            ],
            0,
            302,
            width=24,
            unit="none",
        ),
        row(107, "RAPEC-v8 Bayesian-v3 Diagnostics", 311),
        task_lines(
            59,
            "RAPEC-v8 v3 and Bayesian Stop Agreement",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec8_base_v3_stop", "RAPEC-v3 stop / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec8_bayesian_low_gain", "Bayesian low-gain evidence / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec8_bayesian_veto", "Bayesian veto / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec8_candidate_stop", "Final candidate / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec8_confirmation_streak", "Confirmation streak / {{benchmark_case_id}}"),
            ],
            0,
            312,
            width=24,
            unit="none",
        ),
        task_lines(
            60,
            "RAPEC-v8 Posterior Gain and Dynamic Threshold",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec8_posterior_expected_gain", "Posterior expected gain / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec8_posterior_gain_lower", "10% credible bound / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec8_posterior_gain_upper", "90% credible bound / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec8_dynamic_relevant_gain", "RAPEC-v3 dynamic gain / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec8_posterior_probability_relevant_gain", "P(relevant gain) / {{benchmark_case_id}}"),
                ("F", "slurm_job_epoch_controller_plugin_rapec8_probability_limit", "Probability limit / {{benchmark_case_id}}"),
            ],
            0,
            321,
            width=24,
            unit="none",
        ),
        task_lines(
            61,
            "RAPEC-v8 Posterior Energy Utility",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec8_expected_energy_wh", "Expected horizon energy (Wh) / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec8_expected_utility_per_wh", "Posterior utility Q/Wh / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec8_posterior_mean_epoch_delta", "Posterior mean epoch delta / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec8_posterior_epoch_delta_stddev", "Posterior epoch-delta stddev / {{benchmark_case_id}}"),
            ],
            0,
            330,
            width=24,
            unit="none",
        ),
        row(108, "RAPEC-v9 Risk-Constrained Bayesian Diagnostics", 339),
        task_lines(
            62,
            "RAPEC-v9 Probability of a Relevant Future Quality Gain",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec9_h1_prob_relevant_gain", "1 epoch / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec9_h5_prob_relevant_gain", "5 epochs / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec9_h10_prob_relevant_gain", "10 epochs / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec9_h20_prob_relevant_gain", "20 epochs / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec9_endpoint_prob_relevant_gain", "Remaining budget / {{benchmark_case_id}}"),
                ("F", "slurm_job_epoch_controller_plugin_rapec9_quality_risk_alpha", "Risk limit / {{benchmark_case_id}}"),
            ],
            0,
            340,
            width=24,
            unit="none",
        ),
        task_lines(
            63,
            "RAPEC-v9 Dynamic Quality Equivalence and Predicted Gain",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec9_dynamic_quality_epsilon", "Dynamic equivalence / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec9_instantaneous_quality_epsilon", "Instantaneous noise margin / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec9_h5_expected_gain", "Expected gain, 5 epochs / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec9_h10_expected_gain", "Expected gain, 10 epochs / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec9_h20_expected_gain", "Expected gain, 20 epochs / {{benchmark_case_id}}"),
                ("F", "slurm_job_epoch_controller_plugin_rapec9_endpoint_gain_upper", "Upper gain, remaining budget / {{benchmark_case_id}}"),
            ],
            0,
            349,
            width=24,
            unit="none",
        ),
        task_lines(
            64,
            "RAPEC-v9 Energy Utility and Sequential Stop Evidence",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec9_h5_expected_excess_utility_per_wh", "Excess utility, 5 epochs / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec9_h10_expected_excess_utility_per_wh", "Excess utility, 10 epochs / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec9_h20_expected_excess_utility_per_wh", "Excess utility, 20 epochs / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec9_dynamic_utility_floor", "Dynamic utility floor / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec9_stop_evidence_probability", "Stop evidence / {{benchmark_case_id}}"),
                ("F", "slurm_job_epoch_controller_plugin_rapec9_evidence_stop_probability", "Stop threshold / {{benchmark_case_id}}"),
            ],
            0,
            358,
            width=24,
            unit="none",
        ),
        task_lines(
            65,
            "RAPEC-v9 Readiness, Constraints, and Regime Guards",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec9_model_ready", "Model ready / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec9_quality_constraint_met", "Quality constraint / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec9_energy_low_value", "Low energy utility / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec9_learning_rate_transition", "LR transition guard / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec9_regime_recovery", "Recovery guard / {{benchmark_case_id}}"),
                ("F", "slurm_job_epoch_controller_plugin_rapec9_record_waiting_recovery", "Record-waiting guard / {{benchmark_case_id}}"),
                ("G", "slurm_job_epoch_controller_plugin_rapec9_telemetry_quality_model_available", "Telemetry quality model / {{benchmark_case_id}}"),
                ("H", "slurm_job_epoch_controller_plugin_rapec9_candidate_stop", "Candidate stop / {{benchmark_case_id}}"),
                ("I", "slurm_job_epoch_controller_plugin_rapec9_dynamic_minimum_observations", "Dynamic minimum observations / {{benchmark_case_id}}"),
            ],
            0,
            367,
            width=24,
            unit="none",
        ),
        task_lines(
            66,
            "RAPEC-v9 Predicted Training Energy by Horizon (Wh)",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec9_h1_expected_energy_wh", "1 epoch / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec9_h5_expected_energy_wh", "5 epochs / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec9_h10_expected_energy_wh", "10 epochs / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec9_h20_expected_energy_wh", "20 epochs / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec9_endpoint_expected_energy_wh", "Remaining budget / {{benchmark_case_id}}"),
            ],
            0,
            376,
            width=24,
            unit="watth",
        ),
        row(109, "RAPEC-v10 Task-Independent Dynamic Warm-up", 385),
        task_lines(
            67,
            "RAPEC-v10 Online Signal and Noise Calibration",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec10_recent_gain", "Recent quality gain / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec10_detrended_validation_noise", "Detrended validation noise / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec10_signal_to_noise_ratio", "Signal-to-noise ratio / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec10_noise_stability_ratio", "Noise stability ratio / {{benchmark_case_id}}"),
            ],
            0,
            386,
            width=24,
            unit="none",
        ),
        task_lines(
            68,
            "RAPEC-v10 Dynamic Minimum Epoch and Release State",
            [
                ("A", "slurm_job_epoch_controller_plugin_rapec10_effective_min_epochs", "Effective minimum epoch / {{benchmark_case_id}}"),
                ("B", "slurm_job_epoch_controller_plugin_rapec10_dynamic_min_epoch_fraction", "Dynamic minimum fraction / {{benchmark_case_id}}"),
                ("C", "slurm_job_epoch_controller_plugin_rapec10_stable_warmup_windows", "Stable windows / {{benchmark_case_id}}"),
                ("D", "slurm_job_epoch_controller_plugin_rapec10_warmup_released", "Warm-up released / {{benchmark_case_id}}"),
                ("E", "slurm_job_epoch_controller_plugin_rapec10_learning_is_active", "Learning active / {{benchmark_case_id}}"),
                ("F", "slurm_job_epoch_controller_plugin_rapec10_noise_is_stable", "Noise stable / {{benchmark_case_id}}"),
            ],
            0,
            395,
            width=24,
            unit="none",
        ),
    ]
    dashboard = {"annotations": {"list": []}, "editable": True, "graphTooltip": 1, "id": None, "panels": panels, "refresh": "30s", "schemaVersion": 39, "tags": ["controller-benchmark", "slurm", "energy"], "templating": {"list": [
        variable("benchmark_run_id", "Benchmark Run", "label_values(controller_benchmark_case_info, benchmark_run_id)", False),
        variable("task_types", "Task Type", 'label_values(controller_benchmark_case_info{benchmark_run_id="$benchmark_run_id"}, task_type)'),
        variable("stages", "Stages", 'label_values(controller_benchmark_case_info{benchmark_run_id="$benchmark_run_id",task_type=~"$task_types"}, stage)'),
        variable("cases", "Cases", 'label_values(controller_benchmark_case_info{benchmark_run_id="$benchmark_run_id",task_type=~"$task_types",stage=~"$stages"}, case_id)'),
        variable("phases", "Phases", 'label_values(controller_benchmark_phase_gpu_energy_wh{benchmark_run_id="$benchmark_run_id",task_type=~"$task_types",stage=~"$stages",case_id=~"$cases"}, phase)'),
    ]}, "time": {"from": "now-30d", "to": "now"}, "timezone": "browser", "title": "Controller Benchmark - Stability and Generalisation", "uid": "controller-benchmark", "version": 1}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dashboard, separators=(",", ":")), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
