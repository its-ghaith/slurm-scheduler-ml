# setup_poc.ps1
param([string]$Action = "all")

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   ML Energy Tracking - Proof of Concept" -ForegroundColor Cyan
Write-Host "   Prometheus + Grafana + MLflow Integration" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

function Start-Services {
    Write-Host "[1/6] Starte Monitoring Services..." -ForegroundColor Yellow
    docker-compose up -d
    Write-Host "Warte auf Services..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    Write-Host "Services gestartet:" -ForegroundColor Green
    Write-Host "  - Prometheus: http://localhost:9090" -ForegroundColor Cyan
    Write-Host "  - Grafana: http://localhost:3000 (admin/admin)" -ForegroundColor Cyan
    Write-Host "  - MLflow: http://localhost:5000" -ForegroundColor Cyan
    Write-Host "  - cAdvisor: http://localhost:8080" -ForegroundColor Cyan
}

function Build-Image {
    Write-Host ""
    Write-Host "[2/6] Baue PyTorch Image..." -ForegroundColor Yellow
    docker build -t ml-energy-tracker .
    Write-Host ""
    Write-Host "[3/6] Baue MLflow Image..." -ForegroundColor Yellow
    docker build -t mlflow:latest -f Dockerfile.mlflow .
}

function Run-Experiments {
    Write-Host ""
    Write-Host "[4/6] Führe Experimente durch..." -ForegroundColor Yellow
    $batchSizes = @(16, 32, 64, 128)
    foreach ($bs in $batchSizes) {
        Write-Host "`n>>> Experiment mit Batch Size $bs" -ForegroundColor Magenta
        docker run --rm `
            --name "experiment_batch$bs" `
            --cpus="2.0" `
            --memory="4g" `
            --gpus all `
            -e MLFLOW_TRACKING_URI=http://mlflow:5000 `
            -e BATCH_SIZE=$bs `
            -v ${PWD}:/app `
            -v /sys/class/powercap:/sys/class/powercap:ro `
            --network monitoring `
            ml-energy-tracker `
            python train_with_energy_tracking_mlflow.py
    }
}

function Show-Dashboards {
    Write-Host ""
    Write-Host "[5/6] Öffne Dashboards..." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "                    WICHTIGE LINKS" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  MLflow Experiment Tracking:  http://localhost:5000" -ForegroundColor Cyan
    Write-Host "  Grafana Dashboard:           http://localhost:3000" -ForegroundColor Cyan
    Write-Host "    Login: admin / admin" -ForegroundColor Cyan
    Write-Host "  Prometheus Targets:          http://localhost:9090/targets" -ForegroundColor Cyan
    Write-Host "  cAdvisor Container-Metriken: http://localhost:8080" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    Start-Process "http://localhost:5000"
    Start-Process "http://localhost:3000"
    Start-Process "http://localhost:9090"
}

function Show-Guide {
    Write-Host ""
    Write-Host "GRAFANA DASHBOARD SETUP:" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green
    Write-Host "1. http://localhost:3000 öffnen (admin/admin)"
    Write-Host "2. Configuration -> Data Sources -> Add data source -> Prometheus"
    Write-Host "   URL: http://prometheus:9090 -> Save & Test"
    Write-Host "3. + -> Import -> Dashboard ID 14282 (cAdvisor) oder 1860 (Node Exporter)"
    Write-Host ""
    Write-Host "MLFLOW UI:" -ForegroundColor Green
    Write-Host "  - http://localhost:5000 zeigt alle Experimente"
    Write-Host "  - Vergleichen Sie verschiedene Batch Sizes"
    Write-Host ""
}

switch ($Action) {
    "all" {
        Start-Services
        Build-Image
        Run-Experiments
        Show-Dashboards
        Show-Guide
    }
    "services" { Start-Services }
    "build" { Build-Image }
    "run" { Run-Experiments }
    "dashboard" { Show-Dashboards; Show-Guide }
    default {
        Write-Host "Usage: .\setup_poc.ps1 [all|services|build|run|dashboard]"
    }
}
Write-Host "`n✅ Proof-of-Concept abgeschlossen!" -ForegroundColor Green