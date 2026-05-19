# Merge Alpha Vantage MCP into Cursor config (remote server — no uv required).
# Usage (from repo root):
#   $env:ALPHA_VANTAGE_API_KEY = "your-key"
#   .\scripts\configure_cursor_alphavantage_mcp.ps1
# Or reads ALPHA_VANTAGE_API_KEY from .env if present.

param(
    [string] $McpPath = "$env:USERPROFILE\.cursor\mcp.json",
    [string] $ApiKey = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $repoRoot ".env"

if (-not $ApiKey) {
    $ApiKey = $env:ALPHA_VANTAGE_API_KEY
}
if (-not $ApiKey -and (Test-Path $envFile)) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*ALPHA_VANTAGE_API_KEY\s*=\s*(.+)\s*$') {
            $ApiKey = $Matches[1].Trim().Trim('"').Trim("'")
            break
        }
    }
}
if (-not $ApiKey) {
    Write-Error "Set ALPHA_VANTAGE_API_KEY or pass -ApiKey"
}

if (-not (Test-Path $McpPath)) {
    $dir = Split-Path $McpPath -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    @{ mcpServers = @{} } | ConvertTo-Json -Depth 10 | Set-Content $McpPath -Encoding utf8
}

$json = Get-Content $McpPath -Raw | ConvertFrom-Json
if (-not $json.PSObject.Properties['mcpServers']) {
    $json | Add-Member -NotePropertyName mcpServers -NotePropertyValue ([pscustomobject]@{})
}
$url = "https://mcp.alphavantage.co/mcp?apikey=$ApiKey"
$json.mcpServers | Add-Member -NotePropertyName alphavantage -NotePropertyValue ([pscustomobject]@{ url = $url }) -Force
$json | ConvertTo-Json -Depth 10 | Set-Content $McpPath -Encoding utf8
Write-Host "Updated $McpPath — alphavantage MCP (remote) configured."
