param(
    [string]$Provider = $env:LLM_PROVIDER,
    [string]$DataDir = $env:RAG_DATA_DIR,
    [string]$ApiKey,
    [string]$Model,
    [double]$Temperature
)

if (-not $Provider) { $Provider = "openai" }
if (-not $DataDir) { $DataDir = "data/sample" }

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Cricket Analytics RAG - Runner" -ForegroundColor Cyan
Write-Host "Provider : $Provider" -ForegroundColor Cyan
Write-Host "DataDir  : $DataDir" -ForegroundColor Cyan
if ($Model)      { Write-Host "Model    : $Model" -ForegroundColor Cyan }
if ($Temperature -ne $null) { Write-Host "Temp     : $Temperature" -ForegroundColor Cyan }
Write-Host "=============================================" -ForegroundColor Cyan

$env:LLM_PROVIDER = $Provider
$env:RAG_DATA_DIR = $DataDir
$env:PYTHONPATH = (Get-Location).Path

# Load .env if present (defaults)
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^[A-Za-z_][A-Za-z0-9_]*=') {
            $name, $value = $_.Split('=',2)
            [Environment]::SetEnvironmentVariable($name, $value)
        }
    }
}

# Override with CLI-provided values
if ($Model) { $env:LLM_MODEL_NAME = $Model }
if ($Temperature -ne $null) { $env:LLM_TEMPERATURE = [string]$Temperature }
if ($ApiKey) {
    switch ($Provider.ToLower()) {
        'openai' { $env:OPENAI_API_KEY = $ApiKey }
        'groq'   { $env:GROQ_API_KEY = $ApiKey }
        'gemini' { $env:GEMINI_API_KEY = $ApiKey; $env:GOOGLE_API_KEY = $ApiKey }
        default  { Write-Warning "Unknown provider '$Provider'. Ignoring ApiKey override." }
    }
}

# Preflight: check API key availability for selected provider
function Get-ProviderKeyNames([string]$p) {
    switch ($p.ToLower()) {
        'openai' { return @('OPENAI_API_KEY') }
        'groq'   { return @('GROQ_API_KEY') }
        'gemini' { return @('GEMINI_API_KEY','GOOGLE_API_KEY') }
        default  { return @() }
    }
}

$keyNames = Get-ProviderKeyNames -p $Provider
$hasKey = $false
foreach ($k in $keyNames) { if ($env:$k) { $hasKey = $true } }
if (-not $hasKey -and $keyNames.Count -gt 0) {
    Write-Warning "No API key detected for provider '$Provider'."
    Write-Host "- You can pass -ApiKey '<KEY>' to this script, or" -ForegroundColor Yellow
    Write-Host "- Paste a key per request in the Web UI at /ui (field: API Key)." -ForegroundColor Yellow
    Write-Host "Proceeding without LLM will enable retrieval-only (sources) responses." -ForegroundColor Yellow
}

# Verify venv tools exist
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Error "Missing .venv. Create it and install requirements: python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path ".\.venv\Scripts\uvicorn.exe")) {
    Write-Error "Uvicorn not found in venv. Run: .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt"
    exit 1
}

# Initialize vectorstore using venv Python (fail fast with clear message)
try {
    .\.venv\Scripts\python.exe .\scripts\init_vectorstore.py
} catch {
    Write-Error "Vectorstore initialization failed. Check RAG_DATA_DIR ('$DataDir') and dependencies. $_"
    exit 1
}

# Run API
try {
    .\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000
} catch {
    Write-Error "Failed to start API: $_"
    exit 1
}
