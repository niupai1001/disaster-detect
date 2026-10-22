param([switch]$Apply, [switch]$Verify, [switch]$Finalize)
$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$ArchiveRoot = Join-Path $ProjectRoot 'archive/experiment_history'
$CatalogPath = Join-Path $ArchiveRoot 'catalog.json'

function Assert-WorkspacePath([string]$Path) {
    $Full = [IO.Path]::GetFullPath($Path)
    $Prefix = $ProjectRoot.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
    if (-not $Full.StartsWith($Prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target outside intended project: $Full"
    }
    return $Full
}

function Verify-Catalog {
    $Catalog = Get-Content -LiteralPath $CatalogPath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($Record in $Catalog.files) {
        $Target = Assert-WorkspacePath (Join-Path $ProjectRoot $Record.archived_path)
        if (-not (Test-Path -LiteralPath $Target -PathType Leaf)) { throw "Missing archive: $Target" }
        $Actual = (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -ne $Record.sha256) { throw "Archive hash mismatch: $Target" }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'reports') -PathType Container)) { throw 'Live reports compatibility path missing' }
    Write-Output "PASS: $($Catalog.files.Count) archived files match recorded snapshot SHA256; live reports path retained; $($Catalog.pending_live_files.Count) pending live files"
}

if ($Verify) { Verify-Catalog; exit 0 }
if ($Finalize) {
    if (Test-Path -LiteralPath $CatalogPath) { Verify-Catalog; exit 0 }
    $Planned = Get-Content -LiteralPath (Join-Path $ArchiveRoot 'planned_catalog.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $FinalFiles = [Collections.Generic.List[object]]::new()
    $FinalPending = [Collections.Generic.List[object]]::new()
    foreach ($Record in $Planned.files) { $FinalFiles.Add($Record) }
    foreach ($File in Get-ChildItem -LiteralPath (Join-Path $ArchiveRoot 'reports') -Recurse -File) {
        $OriginalRelative = [IO.Path]::GetRelativePath($ArchiveRoot, $File.FullName).Replace('\','/')
        $Hash = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        try {
            $SourceHash = (Get-FileHash -LiteralPath (Join-Path $ProjectRoot $OriginalRelative) -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($SourceHash -ne $Hash) { $FinalPending.Add([pscustomobject]@{original_path=$OriginalRelative;reason='Original runtime file changed after archived snapshot'}) }
        } catch { $FinalPending.Add([pscustomobject]@{original_path=$OriginalRelative;reason='Original runtime file currently occupied'}) }
        $FinalFiles.Add([pscustomobject]@{original_path=$OriginalRelative;archived_path=('archive/experiment_history/'+$OriginalRelative);bytes=$File.Length;sha256=$Hash;kind='runtime_snapshot'})
    }
    [ordered]@{date='2026-10-21';state='archived';reports_compatibility='Original reports runtime directory retained';
        links='No directory aliases are required. query_research.py resolves historical links from their original paths.';
        files=@($FinalFiles.ToArray());pending_live_files=@($FinalPending.ToArray())} |
        ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $CatalogPath -Encoding UTF8
    Verify-Catalog
    exit 0
}
if (Test-Path -LiteralPath $CatalogPath) { throw 'Archive already exists. Use -Verify; no overwrite permitted.' }

$Keep = @('RESEARCH_INDEX.md','DATA_OFFICIAL.md','DATA_EVENT_PROTOCOL.md','MODEL_EVOLUTION.md','reporting_style.md')
$DocsRoot = Join-Path $ProjectRoot 'docs'
$Entries = @(Get-ChildItem -LiteralPath $DocsRoot -Force | Where-Object {
    if ($_.PSIsContainer) { $_.Name -ne 'assets' } else { $_.Name -notin $Keep }
})
$Moves = @()
$Records = [Collections.Generic.List[object]]::new()
$Copies = [Collections.Generic.List[object]]::new()
$Pending = [Collections.Generic.List[object]]::new()

foreach ($Entry in $Entries) {
    $Source = Assert-WorkspacePath $Entry.FullName
    if (($Entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Refusing to move existing link: $Source" }
    $Relative = [IO.Path]::GetRelativePath($ProjectRoot, $Source)
    $Destination = Assert-WorkspacePath (Join-Path $ArchiveRoot $Relative)
    if (Test-Path -LiteralPath $Destination) { throw "Would overwrite: $Destination" }
    $Moves += [pscustomobject]@{ Source=$Source; Destination=$Destination }
    $Files = if ($Entry.PSIsContainer) { @(Get-ChildItem -LiteralPath $Source -Recurse -File) } else { @($Entry) }
    foreach ($File in $Files) {
        $OriginalRelative = [IO.Path]::GetRelativePath($ProjectRoot, $File.FullName).Replace('\','/')
        $Records.Add([pscustomobject]@{
            original_path=$OriginalRelative
            archived_path=('archive/experiment_history/' + $OriginalRelative)
            bytes=$File.Length
            sha256=(Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            kind='moved_historical_document'
        })
    }
}

# Training may still write to reports. Snapshot stable files, never move its runtime directory.
foreach ($File in Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'reports') -Recurse -File) {
    $Source = Assert-WorkspacePath $File.FullName
    $OriginalRelative = [IO.Path]::GetRelativePath($ProjectRoot, $Source).Replace('\','/')
    $Destination = Assert-WorkspacePath (Join-Path $ArchiveRoot $OriginalRelative)
    if (Test-Path -LiteralPath $Destination) { throw "Would overwrite snapshot: $Destination" }
    try { $Hash = (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash.ToLowerInvariant() }
    catch {
        $Pending.Add([pscustomobject]@{original_path=$OriginalRelative;reason='File currently occupied or not stable; runtime path retained'})
        continue
    }
    $Copies.Add([pscustomobject]@{Source=$Source;Destination=$Destination;Record=[pscustomobject]@{
        original_path=$OriginalRelative;archived_path=('archive/experiment_history/'+$OriginalRelative)
        bytes=$File.Length;sha256=$Hash;kind='stable_runtime_snapshot'
    }})
}

Write-Output "Plan verified within $ProjectRoot : $($Moves.Count) document moves; $($Copies.Count) report snapshots; $($Pending.Count) pending live files"
if (-not $Apply) { Write-Output 'Preview only. Use -Apply for authorized archival.'; exit 0 }

New-Item -ItemType Directory -Path $ArchiveRoot -Force | Out-Null
# Write the complete planned mapping first, so a partial interruption is recoverable.
$Planned = [ordered]@{ date='2026-10-21'; state='planned'; files=@($Records.ToArray()) }
$Planned | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $ArchiveRoot 'planned_catalog.json') -Encoding UTF8
foreach ($Move in $Moves) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $Move.Destination) -Force | Out-Null
    Move-Item -LiteralPath $Move.Source -Destination $Move.Destination
}

foreach ($Copy in $Copies) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $Copy.Destination) -Force | Out-Null
    try {
        Copy-Item -LiteralPath $Copy.Source -Destination $Copy.Destination
        $SourceHash = (Get-FileHash -LiteralPath $Copy.Source -Algorithm SHA256).Hash.ToLowerInvariant()
        $CopyHash = (Get-FileHash -LiteralPath $Copy.Destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($SourceHash -ne $Copy.Record.sha256 -or $CopyHash -ne $Copy.Record.sha256) {
            $Pending.Add([pscustomobject]@{original_path=$Copy.Record.original_path;reason='Changed during snapshot; not accepted as stable archive'})
            continue
        }
        $Records.Add($Copy.Record)
    } catch {
        $Pending.Add([pscustomobject]@{original_path=$Copy.Record.original_path;reason='Became occupied during snapshot; not accepted as stable archive'})
    }
}

# Historical bytes stay unchanged. The query helper resolves their links using original paths.

$Catalog = [ordered]@{
    date='2026-10-21'; state='archived'
    policy='Original history bytes preserved. Default agents read only the maintained research index; history is on demand.'
    reports_compatibility='reports stays at its original runtime path; stable snapshots reside in archive/experiment_history/reports'
    live_artifacts='Training outputs/configs/data/checkpoints remain at their original paths; archive aliases are links, not copies.'
    files=@($Records.ToArray())
    pending_live_files=@($Pending.ToArray())
}
$Catalog | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $CatalogPath -Encoding UTF8
Verify-Catalog
