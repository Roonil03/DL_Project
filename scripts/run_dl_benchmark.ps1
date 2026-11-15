#requires -Version 5.1
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$DatasetPath = '',
    [ValidateSet('global_indian','india_indices','fred_sectors')][string]$DatasetMode = 'india_indices',
    [ValidateSet('dlinear','lstm','mamba','patchtst','ahat_fin')][string[]]$Models = @('dlinear','lstm','mamba','patchtst'),
    [ValidateSet('gpu')][string]$Device = 'gpu',
    [int[]]$Seeds = @(42,52,62,72,82),
    [int[]]$Folds = @(),
    [int]$Epochs = 100,
    [string]$RunId = '',
    [string]$OutputRoot = 'outputs',
    [ValidateSet('auto','restart','fail')][string]$ResumeMode = 'auto',
    [switch]$ForceFresh,
    [switch]$PackageCheckpoints,
    [switch]$ExcludeCheckpoints,
    [int]$MaxRetries = 2,
    [switch]$EstimateOnly
)
$ErrorActionPreference = 'Stop'
function Get-RunnerResult {
    param([string]$Run, [string]$Archive, [bool]$Pilot)
    if ($Pilot) {
        $timing = Join-Path $Run 'telemetry\timing.json'
        if (-not (Test-Path -LiteralPath $timing -PathType Leaf)) { throw 'Runner returned success without pilot timing.' }
        return $timing
    }
    if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) { throw 'Runner returned success without archive' }
    return $Archive
}
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root
if (-not $RunId) { $RunId = 'run-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') }
if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$' -or $RunId.Contains('..')) { throw 'RunId must be a safe folder name.' }
if ($Folds.Count -eq 0 -and $DatasetMode -eq 'global_indian') { $Folds = @(0) }
if ($Epochs -lt 1 -or $MaxRetries -lt 0 -or $Seeds.Count -eq 0) { throw 'Epochs and seed list must be positive; retries cannot be negative.' }
if ($Folds | Where-Object { $_ -lt 0 }) { throw 'Fold IDs must be nonnegative.' }
if ($ForceFresh) { $ResumeMode = 'restart' }
if ($PackageCheckpoints -and $ExcludeCheckpoints) { throw 'PackageCheckpoints and ExcludeCheckpoints cannot be combined.' }
if (-not $DatasetPath) {
    if ($DatasetMode -eq 'india_indices') { $DatasetPath = 'data\india_indices' }
    elseif ($DatasetMode -eq 'fred_sectors') { $DatasetPath = 'data\fred_nasdaq_sectors' }
    else { $DatasetPath = 'data\kaggle_global_indian_markets\global-india-markets-macro' }
}
$dataRoot = [IO.Path]::GetFullPath((Join-Path $root 'data'))
$data = if ([IO.Path]::IsPathRooted($DatasetPath)) { [IO.Path]::GetFullPath($DatasetPath) } else { [IO.Path]::GetFullPath((Join-Path $root $DatasetPath)) }
if (-not $data.StartsWith($dataRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'DatasetPath must be under the repository data directory mounted into Docker.' }
$requiredFiles = if ($DatasetMode -eq 'india_indices') { @('NIFTY_BANK.parquet','NIFTY_IT.parquet','NIFTY_PHARMA.parquet','index_list.csv') } elseif ($DatasetMode -eq 'fred_sectors') { @('sector_indices.csv') } else { @('daily_market_data.csv','monthly_macro_data.csv','codes_and_sources.md') }
if ($DatasetMode -eq 'india_indices') {
    $hashes = @{
        'NIFTY_BANK.parquet' = 'cb19fc783ac88fd0f57be97dab2b615a5be5bb86659925315170bb47deb7795c'
        'NIFTY_IT.parquet' = '281106003c13169a69f91c9cb0709169bea5663cf04b2c733e6abafceb751aab'
        'NIFTY_PHARMA.parquet' = '71e517166f82e12a0a2740222d34d47d10d4a2b104c940636762bd09a6bd3ded'
        'index_list.csv' = 'c164b2e1ce9cef2df67f98ee695b25684cc6be360a0999465f0a283260ba5292'
    }
    $baseUrl = 'https://raw.githubusercontent.com/aswinv90/indian-market-historical-data/1f4aff7c3e53a9f2bfb6a167b0d80f4d0c8f8bf9/data/INDICES/'
    foreach ($name in $requiredFiles) {
        $file = Join-Path $data $name
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
            if ($WhatIfPreference) { Write-Host "Would download pinned Indian data to $data"; return }
            New-Item -ItemType Directory -Path $data -Force | Out-Null
            $temporary = "$file.download"
            try {
                Invoke-WebRequest -Uri ($baseUrl + $name) -OutFile $temporary -TimeoutSec 60
                if ((Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hashes[$name]) { throw "Checksum mismatch: $name" }
                Move-Item -LiteralPath $temporary -Destination $file
            } finally { if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary } }
        }
        if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hashes[$name]) { throw "Existing snapshot checksum mismatch: $name" }
    }
}
$sectorFile = Join-Path $data 'sector_indices.csv'
if ($DatasetMode -eq 'fred_sectors' -and -not (Test-Path -LiteralPath $sectorFile -PathType Leaf)) {
    if ($WhatIfPreference) { Write-Host "Would download FRED CSV to $sectorFile"; return }
    New-Item -ItemType Directory -Path $data -Force | Out-Null
    $temporary = "$sectorFile.download"
    $url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQNQUSB10LMT,NASDAQNQUSB20LMT,NASDAQNQUSB30LMT,NASDAQNQUSB50LMT,NASDAQNQUSB60LMT'
    try {
        Invoke-WebRequest -Uri $url -OutFile $temporary -TimeoutSec 60
        $expected = 'observation_date,NASDAQNQUSB10LMT,NASDAQNQUSB20LMT,NASDAQNQUSB30LMT,NASDAQNQUSB50LMT,NASDAQNQUSB60LMT'
        if ((Get-Content -LiteralPath $temporary -TotalCount 1) -ne $expected) { throw 'Downloaded file has unexpected FRED CSV header.' }
        Move-Item -LiteralPath $temporary -Destination $sectorFile
        $env:FIN_DATA_RETRIEVED_AT = [DateTime]::UtcNow.ToString('o')
    } finally { if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary } }
}
foreach ($name in $requiredFiles) {
    $file = Join-Path $data $name
    if (-not (Test-Path -LiteralPath $file -PathType Leaf) -or (Get-Item -LiteralPath $file).Length -eq 0) { throw "Missing or empty dataset input: $file" }
}
$data = (Resolve-Path -LiteralPath $data -ErrorAction Stop).Path
if (-not $data.StartsWith($dataRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Resolved DatasetPath must be under the repository data directory mounted into Docker.' }
$containerData = '/home/jovyan/work/data/' + $data.Substring($dataRoot.Length + 1).Replace('\','/')
$output = [IO.Path]::GetFullPath((Join-Path $root $OutputRoot))
if (-not $output.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'OutputRoot must be inside the repository.' }
if ($OutputRoot -ne 'outputs') { throw 'The existing Compose mount supports OutputRoot=outputs only.' }
$run = Join-Path $output $RunId
if ((Test-Path -LiteralPath $run) -and $ResumeMode -eq 'fail') { throw 'Run exists and ResumeMode=fail.' }
if ((Test-Path -LiteralPath $run) -and $ResumeMode -eq 'restart') { Write-Host 'Existing run will be archived by the container runner.' }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker Compose is required. Install Docker Desktop and enable NVIDIA Container Toolkit for GPU runs.' }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw 'GPU mode requires an NVIDIA driver and NVIDIA Container Toolkit. On Windows, use WSL2 with Docker GPU support.' }
if (-not $env:JUPYTER_TOKEN) { throw 'Set a nonempty JUPYTER_TOKEN in your PowerShell session before running.' }
$plan = @{
    run_id=$RunId; model_names=$Models; seeds=$Seeds; folds=$Folds; device=$Device
    epochs=$Epochs; resume_mode=$ResumeMode; dataset=$data; output=$run
    fold_selection=if ($Folds.Count -eq 0) { 'auto from dataset configuration' } else { 'explicit' }
    estimate_only=[bool]$EstimateOnly; checkpoint_archive=(-not [bool]$ExcludeCheckpoints)
}
Write-Host ('Planned run: ' + ($plan | ConvertTo-Json -Compress -Depth 5))
Write-Host 'Commands: reuse local image or build if absent; start Compose service; GPU preflight; execute runner.'
Write-Host 'Estimate: prior telemetry is required for a numeric forecast; ESTIMATE_ONLY=1 runs one measured epoch per selected model.'
if ($WhatIfPreference) { return }
if (-not $PSCmdlet.ShouldProcess($run, 'Build CUDA container, verify GPU access, and run benchmark in Docker')) { return }
New-Item -ItemType Directory -Path $output -Force | Out-Null
$preflight = Join-Path $output ($RunId + '.preflight.json')
$files = $requiredFiles | ForEach-Object {
    $p = Join-Path $data $_
    @{name=$_; bytes=(Get-Item -LiteralPath $p).Length; sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
}
@{schema_version='1.0'; timestamp_utc=[DateTime]::UtcNow.ToString('o'); run_id=$RunId; data_files=$files; plan=$plan} |
    ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $preflight -Encoding UTF8
$env:GIT_REVISION = (& git rev-parse HEAD).Trim()
try {
    # Check the daemon before deciding whether an image needs to be built.
    & docker info *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Docker daemon unavailable. Start Docker Desktop and retry.' }
    # Reuse an existing CUDA image; build only when it is absent.
    & docker image inspect fin-dl-benchmark:cuda12.4-india1 *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'CUDA image is not present locally; building it once.'
        & docker compose build --pull jupyter
        if ($LASTEXITCODE -ne 0) { throw 'Docker build failed' }
    } else {
        Write-Host 'Reusing local image fin-dl-benchmark:cuda12.4-india1.'
    }
    & docker compose up -d jupyter
    if ($LASTEXITCODE -ne 0) { throw 'Could not start the rebuilt Jupyter container.' }
    $gpuCheck = "import torch; assert torch.cuda.is_available(), 'CUDA unavailable in container'; x=torch.ones(1,device='cuda'); print({'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'tensor':x.item()})"
    & docker compose exec -T jupyter /opt/venv/bin/python -c $gpuCheck
    if ($LASTEXITCODE -ne 0) { throw 'CUDA preflight failed inside Docker. Check NVIDIA Container Toolkit, Docker GPU support, and the CUDA image.' }
    $execArgs = @('compose','exec','-T','-e',"GIT_REVISION=$env:GIT_REVISION",'-e',"RUN_ID=$RunId",'-e',"MODELS=$($Models -join ',')",'-e',"SEEDS=$($Seeds -join ',')",'-e',"FOLDS=$($Folds -join ',')",'-e',"DEVICE=gpu",'-e',"RESUME_MODE=$ResumeMode",'-e',"EPOCHS=$Epochs",'-e',"MAX_RETRIES=$MaxRetries",'-e',"ESTIMATE_ONLY=$([int][bool]$EstimateOnly)",'-e',"DATASET_MODE=$DatasetMode",'-e',"DATA_DIR=$containerData",'-e',"FIN_ASSETS=$env:FIN_ASSETS",'-e',"FIN_DATA_RETRIEVED_AT=$env:FIN_DATA_RETRIEVED_AT",'-e',"PACKAGE_CHECKPOINTS=$([int](-not [bool]$ExcludeCheckpoints))",'jupyter','bash','./run_all_models.sh')
    & docker @execArgs
    if ($LASTEXITCODE -ne 0) { throw "Container runner failed with exit code $LASTEXITCODE; inspect $run" }
    $archive = Join-Path $root ("artifacts\$RunId.zip")
    $result = Get-RunnerResult -Run $run -Archive $archive -Pilot ([bool]$EstimateOnly)
    if ($EstimateOnly) {
        Write-Host "Measured pilot timing: $result"
        return
    }
    Write-Host "Completed archive: $result"
    Write-Host "Manifest: $(Join-Path $run 'bundle_manifest.json')"
    Write-Host "Data hashes: $preflight"
} catch {
    Write-Error $_
    throw
}
