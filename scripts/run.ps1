param(
    [string]$Provider = $env:LLM_PROVIDER,
    [string]$DataDir = $env:RAG_DATA_DIR
)

if (-not $Provider) { $Provider = "openai" }
if (-not $DataDir) { $DataDir = "data/sample" }

Write-Host "Using provider: $Provider; DataDir: $DataDir"

$env:LLM_PROVIDER = $Provider
$env:RAG_DATA_DIR = $DataDir
$env:PYTHONPATH = (Get-Location).Path

# Load .env if present
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^[A-Za-z_][A-Za-z0-9_]*=') {
            $name, $value = $_.Split('=',2)
            [Environment]::SetEnvironmentVariable($name, $value)
        }
    }
}

# Initialize vectorstore using venv Python
 .\.venv\Scripts\python.exe .\scripts\init_vectorstore.py

# Run API
.\.venv\Scripts\uvicorn.exe src.api:app --host 0.0.0.0 --port 8000
