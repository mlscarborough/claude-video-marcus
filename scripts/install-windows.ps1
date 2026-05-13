# install-windows.ps1 — One-shot setup for /watch skill on Windows.
# Run from the skill root directory: .\scripts\install-windows.ps1
#
# Prerequisites: Python 3.10+, ffmpeg, git, winget
# Run as standard user (not Admin) — will prompt for Admin if needed.

param(
    [switch]$SkipTesseract,
    [switch]$SkipVenv,
    [switch]$SkipWatcher
)

$ErrorActionPreference = "Stop"
$SkillRoot = Split-Path -Parent $PSScriptRoot

Write-Host ""
Write-Host "=== /watch skill installer ===" -ForegroundColor Cyan
Write-Host "Skill root: $SkillRoot"
Write-Host ""

# ── 1. Python version check ──────────────────────────────────────────────────
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pyVer = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python not found. Install Python 3.10+ from https://python.org"
    exit 1
}
Write-Host "  $pyVer — OK" -ForegroundColor Green

# ── 2. ffmpeg check ──────────────────────────────────────────────────────────
Write-Host "Checking ffmpeg..." -ForegroundColor Yellow
try {
    $null = ffmpeg -version 2>&1
    Write-Host "  ffmpeg — OK" -ForegroundColor Green
} catch {
    Write-Warning "ffmpeg not found. Installing via winget..."
    winget install Gyan.FFmpeg --silent
    Write-Host "  ffmpeg installed" -ForegroundColor Green
}

# ── 3. Tesseract OCR ─────────────────────────────────────────────────────────
if (-not $SkipTesseract) {
    Write-Host "Checking Tesseract OCR..." -ForegroundColor Yellow
    $tessPath = "C:\Program Files\Tesseract-OCR\tesseract.exe"
    if (Test-Path $tessPath) {
        $tessVer = & $tessPath --version 2>&1 | Select-Object -First 1
        Write-Host "  Tesseract $tessVer — OK" -ForegroundColor Green
    } else {
        Write-Host "  Tesseract not found. Installing via winget (requires Admin)..." -ForegroundColor Yellow
        Start-Process powershell -ArgumentList "winget install UB-Mannheim.TesseractOCR --silent" -Verb RunAs -Wait
        if (Test-Path $tessPath) {
            Write-Host "  Tesseract installed — OK" -ForegroundColor Green
        } else {
            Write-Warning "Tesseract install may have failed. Check manually: $tessPath"
        }
    }
}

# ── 4. Python venv + pip install ─────────────────────────────────────────────
if (-not $SkipVenv) {
    Write-Host "Setting up Python virtual environment..." -ForegroundColor Yellow
    $venvPath = Join-Path $SkillRoot ".venv"
    if (-not (Test-Path $venvPath)) {
        python -m venv $venvPath
        Write-Host "  venv created at $venvPath" -ForegroundColor Green
    } else {
        Write-Host "  venv already exists" -ForegroundColor Green
    }

    $pip = Join-Path $venvPath "Scripts\pip.exe"
    $reqFile = Join-Path $SkillRoot "requirements.txt"
    Write-Host "  Installing dependencies from requirements.txt..."
    & $pip install -r $reqFile --quiet
    Write-Host "  Dependencies installed — OK" -ForegroundColor Green
}

# ── 5. .env file check ───────────────────────────────────────────────────────
Write-Host "Checking .env file..." -ForegroundColor Yellow
$envFile = "$env:USERPROFILE\.config\watch\.env"
if (Test-Path $envFile) {
    Write-Host "  .env exists at $envFile — OK" -ForegroundColor Green
} else {
    Write-Host "  Creating .env template..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path (Split-Path $envFile) | Out-Null
    @"
# Vision and audio APIs
GEMINI_API_KEY=<paste from Google AI Studio>
GROQ_API_KEY=<paste from console.groq.com>

# Supabase
SUPABASE_URL=<paste from Supabase dashboard>
SUPABASE_SERVICE_KEY=<paste service_role key>

# Archive
WATCH_ARCHIVE_LABEL=BackupSSD
WATCH_ARCHIVE_SUBPATH=WatchArchive
WATCH_PENDING_DIR=$env:USERPROFILE\.watch-pending

# Obsidian vault
OBSIDIAN_VAULT_PATH=$env:USERPROFILE\Documents\AI-Agents-Wiki

# Cleanup
WORK_DIR_RETENTION_HOURS=48

# Multi-laptop role
WATCH_IS_WRITER_LAPTOP=true

# Audit endpoint (set after deploying audit-stage2 Edge Function)
WATCH_AUDIT_ENDPOINT=<https://<ref>.supabase.co/functions/v1/audit-stage2>

SETUP_COMPLETE=false
"@ | Out-File -FilePath $envFile -Encoding utf8

    # Restrict permissions to current user only
    icacls $envFile /inheritance:r /grant:r "$($env:USERNAME):F" | Out-Null
    Write-Host "  .env template created — EDIT IT before running /watch!" -ForegroundColor Yellow
}

# ── 6. Obsidian vault directory ───────────────────────────────────────────────
Write-Host "Checking Obsidian vault..." -ForegroundColor Yellow
$vaultPath = "$env:USERPROFILE\Documents\AI-Agents-Wiki"
if (Test-Path $vaultPath) {
    Write-Host "  Vault exists at $vaultPath — OK" -ForegroundColor Green
} else {
    New-Item -ItemType Directory -Force -Path $vaultPath | Out-Null
    New-Item -ItemType Directory -Force -Path "$vaultPath\videos" | Out-Null
    Write-Host "  Vault created at $vaultPath" -ForegroundColor Green
}

# ── 7. Register sync-watcher scheduled task ───────────────────────────────────
if (-not $SkipWatcher) {
    Write-Host "Registering sync-watcher scheduled task..." -ForegroundColor Yellow
    $taskScript = Join-Path $PSScriptRoot "register-watcher-task.ps1"
    if (Test-Path $taskScript) {
        try {
            Start-Process powershell -ArgumentList "-File `"$taskScript`"" -Verb RunAs -Wait
            Write-Host "  Watcher task registered — OK" -ForegroundColor Green
        } catch {
            Write-Warning "Could not register watcher task automatically. Run manually as Admin: .\scripts\register-watcher-task.ps1"
        }
    } else {
        Write-Warning "register-watcher-task.ps1 not found at $taskScript"
    }
}

# ── 8. Run --doctor ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Running /watch --doctor to verify setup..." -ForegroundColor Yellow
$python = Join-Path $SkillRoot ".venv\Scripts\python.exe"
$watchScript = Join-Path $SkillRoot "scripts\watch.py"
& $python $watchScript --doctor

Write-Host ""
Write-Host "=== Installation complete ===" -ForegroundColor Cyan
Write-Host "Next steps:"
Write-Host "  1. Edit $envFile with your real API keys"
Write-Host "  2. Run: & '$python' '$watchScript' --doctor"
Write-Host "  3. Try: & '$python' '$watchScript' 'https://www.youtube.com/watch?v=<id>'"
Write-Host ""
