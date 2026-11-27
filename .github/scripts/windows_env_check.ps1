# Check that the job env looks like a plain pip/uv user's: no conda, no Library\bin and no
# torch runtime DLLs on PATH, since any of them hides missing-DLL bugs (pytorch/pytorch#205808).
# -Fix strips them from the env of the following steps via GITHUB_ENV instead of failing.
param([switch]$Fix)

$dlls = 'sycl*.dll', 'libmmd.dll', 'mkl_core*.dll', 'uv.dll'

function Get-Reason([string]$dir) {
    if ($dir -match '(?i)miniforge|miniconda|anaconda|\\condabin') { return 'conda dir' }
    if ($dir -match '(?i)\\Library\\bin\\?$') { return 'Library\bin dir' }
    # System dirs are what a real user has too.
    if ($dir.StartsWith($env:SystemRoot, [StringComparison]::OrdinalIgnoreCase)) { return $null }
    if (-not (Test-Path -LiteralPath $dir -PathType Container)) { return $null }
    $hits = @(Get-ChildItem -Path (Join-Path $dir '*') -Include $dlls -File -ErrorAction SilentlyContinue)
    if ($hits) { return "dir with torch runtime DLLs ($($hits.Name -join ', '))" }
    return $null
}

$paths = @($env:PATH -split ';' | Where-Object { $_ })
Write-Host 'PATH:'
$paths | ForEach-Object { Write-Host "  $_" }
Write-Host 'python on PATH:'
Get-Command python -All -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $($_.Source)" }

$keep = @()
$problems = @()
foreach ($p in $paths) {
    $reason = Get-Reason $p
    if ($reason) { $problems += "$reason on PATH: $p" } else { $keep += $p }
}
$condaVars = @(Get-ChildItem env: | Where-Object { ($_.Name -like 'CONDA*' -or $_.Name -like '_CONDA*') -and $_.Value })
$condaVars | ForEach-Object { $problems += "conda variable: $($_.Name)=$($_.Value)" }

if (-not $problems) {
    Write-Host 'Job environment is clean'
    exit 0
}
if ($Fix) {
    $problems | ForEach-Object { Write-Host "::warning::Removed from the job env: $_" }
    $lines = @("PATH=$($keep -join ';')") + @($condaVars | ForEach-Object { "$($_.Name)=" })
    $lines | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    exit 0
}
$problems | ForEach-Object { Write-Host "::error::$_" }
Write-Host '::error::Start the runner from a plain shell or as a service, with no conda/oneAPI dirs on the system PATH'
exit 1
