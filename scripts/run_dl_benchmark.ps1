#requires -Version 5.1
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][string]$DatasetPath,
    [ValidateSet('dlinear','lstm','mamba','patchtst','ahat_fin')][string[]]$Models = @('dlinear','lstm','mamba','patchtst'),
    [ValidateSet('gpu','cpu')][string]$Device = 'gpu',
    [int[]]$Seeds = @(42,52,62,72,82),
    [int[]]$Folds = @(0),
    [int]$Epochs = 100,
    [string]$RunId = '',
    [string]$OutputRoot = 'outputs',
    [ValidateSet('auto','restart','fail')][string]$ResumeMode = 'auto',
    [switch]$ForceFresh,
    [switch]$PackageCheckpoints,
    [int]$MaxRetries = 2,
    [switch]$EstimateOnly
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root
if (-not $RunId) { $RunId = 'run-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') }
if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$' -or $RunId.Contains('..')) { throw 'RunId must be a safe folder name.' }
if ($Epochs -lt 1 -or $MaxRetries -lt 0 -or $Seeds.Count -eq 0 -or $Folds.Count -eq 0) { throw 'Epochs and seed/fold lists must be positive; retries cannot be negative.' }
if ($Folds | Where-Object { $_ -lt 0 }) { throw 'Fold IDs must be nonnegative.' }
if ($ForceFresh) { $ResumeMode = 'restart' }
$data = (Resolve-Path -LiteralPath $DatasetPath -ErrorAction Stop).Path
foreach ($name in @('daily_market_data.csv','monthly_macro_data.csv','codes_and_sources.md')) {
    $file = Join-Path $data $name
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing $file. Obtain and verify the local Kaggle dataset before running; this script never downloads data." }
    if ((Get-Item -LiteralPath $file).Length -eq 0) { throw "Empty dataset input: $file" }
}
$expected = (Join-Path $root 'data\kaggle_global_indian_markets\global-india-markets-macro')
if ([IO.Path]::GetFullPath($data) -ne [IO.Path]::GetFullPath($expected)) { throw "The mounted notebook audit reads $expected. Place the verified files there, then pass that directory as DatasetPath." }
$output = [IO.Path]::GetFullPath((Join-Path $root $OutputRoot))
if (-not $output.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'OutputRoot must be inside the repository.' }
if ($OutputRoot -ne 'outputs') { throw 'The existing Compose mount supports OutputRoot=outputs only.' }
$run = Join-Path $output $RunId
if ((Test-Path -LiteralPath $run) -and $ResumeMode -eq 'fail') { throw 'Run exists and ResumeMode=fail.' }
if ((Test-Path -LiteralPath $run) -and $ResumeMode -eq 'restart') { Write-Host 'Existing run will be archived by the container runner.' }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker Compose is required. Install Docker Desktop and enable NVIDIA Container Toolkit for GPU runs.' }
if ($Device -eq 'gpu' -and -not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw 'GPU mode requires NVIDIA driver and Container Toolkit. Use supported WSL2/Docker CUDA, or choose Device=cpu.' }
if (-not $env:JUPYTER_TOKEN) { throw 'Set a nonempty JUPYTER_TOKEN in your PowerShell session before running.' }
$plan = @{
    run_id=$RunId; model_names=$Models; seeds=$Seeds; folds=$Folds; device=$Device
    epochs=$Epochs; resume_mode=$ResumeMode; dataset=$data; output=$run
    estimate_only=[bool]$EstimateOnly; checkpoint_archive=[bool]$PackageCheckpoints
}
Write-Host ('Planned run: ' + ($plan | ConvertTo-Json -Compress -Depth 5))
Write-Host 'Commands: uv venv --seed .venv-benchmark; docker compose build jupyter; docker compose run --rm -e ... jupyter ./run_all_models.sh'
Write-Host 'Estimate: prior telemetry is required for a numeric forecast; ESTIMATE_ONLY=1 runs one measured epoch per selected model.'
if ($WhatIfPreference) { return }
if (-not $PSCmdlet.ShouldProcess($run, 'Create local environment and run benchmark inside Docker')) { return }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv for a project-local Python environment, then retry.' }
if (-not (Test-Path '.venv-benchmark')) { & uv venv --python 3.11 --seed '.venv-benchmark'; if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' } }
if ($Device -eq 'cpu') {
    & '.\.venv-benchmark\Scripts\pip.exe' install -r requirements.cpu.txt
    if ($LASTEXITCODE -ne 0) { throw 'Pinned CPU development package installation failed' }
}
New-Item -ItemType Directory -Path $output -Force | Out-Null
$preflight = Join-Path $output ($RunId + '.preflight.json')
$files = @('daily_market_data.csv','monthly_macro_data.csv','codes_and_sources.md') | ForEach-Object {
    $p = Join-Path $data $_
    @{name=$_; bytes=(Get-Item -LiteralPath $p).Length; sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
}
@{schema_version='1.0'; timestamp_utc=[DateTime]::UtcNow.ToString('o'); run_id=$RunId; data_files=$files; plan=$plan} |
    ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $preflight -Encoding UTF8
$env:GIT_REVISION = (& git rev-parse HEAD).Trim()
$composeArgs = @('compose','run','--rm', '-e',"GIT_REVISION=$env:GIT_REVISION",'-e',"RUN_ID=$RunId",'-e',"MODELS=$($Models -join ',')",'-e',"SEEDS=$($Seeds -join ',')",'-e',"FOLDS=$($Folds -join ',')",'-e',"DEVICE=$Device",'-e',"RESUME_MODE=$ResumeMode",'-e',"EPOCHS=$Epochs",'-e',"MAX_RETRIES=$MaxRetries",'-e',"ESTIMATE_ONLY=$([int][bool]$EstimateOnly)",'jupyter','./run_all_models.sh')
try {
    & docker compose build jupyter
    if ($LASTEXITCODE -ne 0) { throw 'Docker build failed' }
    & docker @composeArgs
    if ($LASTEXITCODE -ne 0) { throw "Container runner failed with exit code $LASTEXITCODE; inspect $run" }
    $archive = Join-Path $root ("artifacts\$RunId.zip")
    if (-not (Test-Path -LiteralPath $archive)) { throw 'Runner returned success without archive' }
    if (-not $PackageCheckpoints) {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $tmp = $archive + '.filtered'
        $source = [IO.Compression.ZipFile]::OpenRead($archive)
        $dest = [IO.Compression.ZipFile]::Open($tmp, [IO.Compression.ZipArchiveMode]::Create)
        try {
            foreach ($entry in $source.Entries) {
                if ($entry.FullName -match '/checkpoints/' -or $entry.FullName -match '/\.env') { continue }
                $copy = $dest.CreateEntry($entry.FullName, [IO.Compression.CompressionLevel]::Optimal)
                if ($entry.Length -eq 0) { continue }
                $inStream = $entry.Open(); $outStream = $copy.Open()
                try { $inStream.CopyTo($outStream) } finally { $inStream.Dispose(); $outStream.Dispose() }
            }
        } finally { $source.Dispose(); $dest.Dispose() }
        Move-Item -LiteralPath $tmp -Destination $archive -Force
    }
    Write-Host "Completed archive: $archive"
    Write-Host "Manifest: $(Join-Path $run 'bundle_manifest.json')"
    Write-Host "Data hashes: $preflight"
} catch {
    Write-Error $_
    throw
}
