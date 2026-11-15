#requires -Version 5.1
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$DatasetPath = '',
    [ValidateSet('global_indian','french_industry')][string]$DatasetMode = 'global_indian',
    [ValidateSet('dlinear','lstm','mamba','patchtst','ahat_fin')][string[]]$Models = @('dlinear','lstm','mamba','patchtst'),
    [ValidateSet('gpu')][string]$Device = 'gpu',
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
if (-not $DatasetPath) {
    if ($DatasetMode -eq 'french_industry') { $DatasetPath = 'data\french_industry' }
    else { $DatasetPath = 'data\kaggle_global_indian_markets\global-india-markets-macro' }
}
$data = (Resolve-Path -LiteralPath $DatasetPath -ErrorAction Stop).Path
$requiredFiles = if ($DatasetMode -eq 'french_industry') { @('10_Industry_Portfolios_daily_CSV.zip') } else { @('daily_market_data.csv','monthly_macro_data.csv','codes_and_sources.md') }
foreach ($name in $requiredFiles) {
    $file = Join-Path $data $name
    if (-not (Test-Path -LiteralPath $file -PathType Leaf) -or (Get-Item -LiteralPath $file).Length -eq 0) { throw "Missing or empty dataset input: $file" }
}
$dataRoot = [IO.Path]::GetFullPath((Join-Path $root 'data'))
if (-not $data.StartsWith($dataRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'DatasetPath must be under the repository data directory mounted into Docker.' }
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
    estimate_only=[bool]$EstimateOnly; checkpoint_archive=[bool]$PackageCheckpoints
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
    # Stop stale containers and reuse an existing CUDA image. Build only when
    # the image is absent; preserve volumes and user data.
    & docker image inspect fin-dl-benchmark:cuda12.4 *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'CUDA image is not present locally; building it once.'
        & docker compose build --pull jupyter
        if ($LASTEXITCODE -ne 0) { throw 'Docker build failed' }
    } else {
        Write-Host 'Reusing local image fin-dl-benchmark:cuda12.4.'
    }
    & docker compose up -d jupyter
    if ($LASTEXITCODE -ne 0) { throw 'Could not start the rebuilt Jupyter container.' }
    $gpuCheck = "import torch; assert torch.cuda.is_available(), 'CUDA unavailable in container'; x=torch.ones(1,device='cuda'); print({'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'tensor':x.item()})"
    & docker compose exec -T jupyter /opt/venv/bin/python -c $gpuCheck
    if ($LASTEXITCODE -ne 0) { throw 'CUDA preflight failed inside Docker. Check NVIDIA Container Toolkit, Docker GPU support, and the CUDA image.' }
    $execArgs = @('compose','exec','-T','-e',"GIT_REVISION=$env:GIT_REVISION",'-e',"RUN_ID=$RunId",'-e',"MODELS=$($Models -join ',')",'-e',"SEEDS=$($Seeds -join ',')",'-e',"FOLDS=$($Folds -join ',')",'-e',"DEVICE=gpu",'-e',"RESUME_MODE=$ResumeMode",'-e',"EPOCHS=$Epochs",'-e',"MAX_RETRIES=$MaxRetries",'-e',"ESTIMATE_ONLY=$([int][bool]$EstimateOnly)",'-e',"DATASET_MODE=$DatasetMode",'-e',"DATA_DIR=$containerData",'-e',"PACKAGE_CHECKPOINTS=$([int][bool]$PackageCheckpoints)",'jupyter','bash','./run_all_models.sh')
    & docker @execArgs
    if ($LASTEXITCODE -ne 0) { throw "Container runner failed with exit code $LASTEXITCODE; inspect $run" }
    $archive = Join-Path $root ("artifacts\$RunId.zip")
    if (-not (Test-Path -LiteralPath $archive)) { throw 'Runner returned success without archive' }
    Write-Host "Completed archive: $archive"
    Write-Host "Manifest: $(Join-Path $run 'bundle_manifest.json')"
    Write-Host "Data hashes: $preflight"
} catch {
    Write-Error $_
    throw
}
