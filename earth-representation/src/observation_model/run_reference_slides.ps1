$ErrorActionPreference = 'Stop'
$obsProjectRoot = Split-Path -Parent $PSScriptRoot
$obsOutput = Join-Path $PSScriptRoot 'outputs'
$obsPython = Join-Path (Split-Path -Parent $obsProjectRoot) 'landslide-seg/.venv-use-independent/Scripts/python.exe'
$obsSource = Join-Path $obsOutput 'releases/probe_v1/earth-representation'
$env:CUDA_VISIBLE_DEVICES = '1'
$env:OBSERVATION_PROJECT_ROOT = $obsProjectRoot
$env:OBSERVATION_RELEASE = Join-Path $obsOutput 'phase0/release_local_v1'
$env:PYTHONUTF8 = '1'
$obsResults = @()
Push-Location $obsSource
try {
    foreach ($obsKind in @('random', 'legacy', 'dofa', 'anysat')) {
        $obsName = if ($obsKind -eq 'legacy') { 'legacy_cnn_dino' } else { $obsKind }
        $obsFeatures = Join-Path $obsOutput "frozen_features/landslide/$obsName"
        $obsProbe = Join-Path $obsOutput "probes/landslide/$obsName"
        $obsArgs = @('-u', '-X', 'utf8', '-m', 'observation_model.frozen_features', '--dataset', 'landslide', '--kind', $obsKind, '--out', $obsFeatures)
        if ($obsKind -eq 'legacy') { $obsArgs += @('--checkpoint', (Join-Path $obsProjectRoot 'outputs/verification/runs/verified_s2_standard.best_unlabeled.pt')) }
        $obsCurrent = @{ status = 'extracting'; kind = $obsKind; gpu = 1; pid = $PID; source_release = $obsSource; completed = $obsResults }
        $obsCurrent | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $obsOutput 'reference_slides_queue_status.json')
        & $obsPython @obsArgs *> (Join-Path $obsOutput "logs/features_landslide_$obsName.console.log")
        $obsFeatureExit = $LASTEXITCODE
        $obsProbeExit = $null
        if ($obsFeatureExit -eq 0) {
            $obsCurrent.status = 'probing'
            $obsCurrent | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $obsOutput 'reference_slides_queue_status.json')
            & $obsPython -u -X utf8 -m observation_model.frozen_probe --features $obsFeatures --out $obsProbe --task segmentation --max-updates 2000 *> (Join-Path $obsOutput "logs/probe_landslide_$obsName.console.log")
            $obsProbeExit = $LASTEXITCODE
        }
        $obsResults += @{ kind = $obsKind; feature_exit = $obsFeatureExit; probe_exit = $obsProbeExit }
    }
    @{ status = 'terminal_pending_acceptance'; pid = $PID; completed = $obsResults } | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $obsOutput 'reference_slides_queue_status.json')
} finally { Pop-Location }
