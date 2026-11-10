param(
    [Parameter(Mandatory=$false)]
    [string]$Message = "chore: auto checkpoint sync"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# 1. Ensure Git from Anaconda PyTorch environment is on PATH
$CondaGitBin = "E:\anaconda\envs\pytorch\Library\bin"
if (Test-Path "$CondaGitBin\git.exe") {
    $env:PATH = "$CondaGitBin;$env:PATH"
}

# 2. Initialize Git repo if not already initialized
if (-not (Test-Path "$RepoRoot\.git")) {
    Write-Host "[GitSync] Initializing Git repository on branch 'main'..." -ForegroundColor Cyan
    git init -b main
    git config user.name "LiuQiuhan-3RRR"
    git config user.email "27798@users.noreply.github.com"
}

# 3. Co-locate & Backup WS63 Firmware files from D:\ws63
$FwDest = Join-Path $RepoRoot "firmware_ws63"
if (-not (Test-Path $FwDest)) {
    New-Item -ItemType Directory -Path $FwDest -Force | Out-Null
}
$FwSourceDir = "D:\ws63\src\application\samples\custom"
if (Test-Path $FwSourceDir) {
    Copy-Item "$FwSourceDir\*.c" $FwDest -Force
    Copy-Item "$FwSourceDir\CMakeLists.txt" $FwDest -Force -ErrorAction SilentlyContinue
}

# 4. Timestamped archive of calib_params.json if present
$CalibFile = Join-Path $RepoRoot "calib_params.json"
if (Test-Path $CalibFile) {
    $BackupDir = Join-Path $RepoRoot "backups"
    if (-not (Test-Path $BackupDir)) {
        New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
    }
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LatestBackup = Get-ChildItem $BackupDir -Filter "calib_params_*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $NeedCopy = $true
    if ($LatestBackup) {
        $HashCurr = (Get-FileHash $CalibFile).Hash
        $HashPrev = (Get-FileHash $LatestBackup.FullName).Hash
        if ($HashCurr -eq $HashPrev) {
            $NeedCopy = $false
        }
    }
    if ($NeedCopy) {
        Copy-Item $CalibFile (Join-Path $BackupDir "calib_params_$Timestamp.json") -Force
        Write-Host "[GitSync] Archived calib_params.json -> backups/calib_params_$Timestamp.json" -ForegroundColor Green
    }
}

# 5. Stage and Commit changes
git add -A
$Status = git status --porcelain
if ($Status) {
    git commit -m $Message
    Write-Host "[GitSync] Committed: $Message" -ForegroundColor Green
} else {
    Write-Host "[GitSync] Working tree clean, nothing new to commit." -ForegroundColor Yellow
}

# 6. Push to GitHub if remote 'origin' is configured
$Remotes = git remote
if ($Remotes -contains "origin") {
    Write-Host "[GitSync] Pushing to GitHub (origin/main)..." -ForegroundColor Cyan
    git push -u origin main
    Write-Host "[GitSync] Successfully pushed to GitHub!" -ForegroundColor Green
} else {
    Write-Host "[GitSync] Local commit saved. Note: GitHub remote 'origin' not yet bound. Run: git remote add origin <YOUR_GITHUB_REPO_URL>" -ForegroundColor Magenta
}
