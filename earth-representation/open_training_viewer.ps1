$ErrorActionPreference = 'Stop'
$viewerRoot = 'http://127.0.0.1:8766/'
$viewerPage = $viewerRoot + '?scheduled=1#/runs/encoder_v2_G5_scheduled_live'

function Test-ViewerReady {
    try { return (Invoke-WebRequest -Uri $viewerRoot -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 }
    catch { return $false }
}

try {
    if (-not (Test-ViewerReady)) {
        Start-ScheduledTask -TaskName 'EarthRepresentation_Viewer_8766'
        $viewerReady = $false
        for ($attempt = 0; $attempt -lt 20; $attempt++) {
            if (Test-ViewerReady) { $viewerReady = $true; break }
            Start-Sleep -Seconds 1
        }
        if (-not $viewerReady) { throw 'Training Viewer did not start on port 8766.' }
    }
    Start-Process $viewerPage
}
catch {
    $popup = New-Object -ComObject WScript.Shell
    $null = $popup.Popup($_.Exception.Message, 0, 'Training Viewer', 16)
    exit 1
}
