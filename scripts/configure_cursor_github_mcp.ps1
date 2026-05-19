# Configure GitHub MCP in Cursor using a Personal Access Token (PAT).
# The built-in GitHub plugin OAuth often fails with "Authorization header is badly formatted".
# Official remote server requires a PAT, not a gho_ OAuth token from git credential manager.
#
# Usage:
#   .\scripts\configure_cursor_github_mcp.ps1 -Pat ghp_xxxxxxxx
#   # or
#   $env:GITHUB_PAT = "ghp_xxxxxxxx"; .\scripts\configure_cursor_github_mcp.ps1
#
# Create PAT: https://github.com/settings/personal-access-tokens/new
# Scopes: repo, read:org, read:user (adjust as needed)

param(
    [string] $Pat = "",
    [string] $McpPath = "$env:USERPROFILE\.cursor\mcp.json"
)

$ErrorActionPreference = "Stop"
if (-not $Pat) { $Pat = $env:GITHUB_PAT }
if (-not $Pat) { $Pat = $env:GITHUB_TOKEN }
$Pat = "$Pat".Trim()
if (-not $Pat) {
    Write-Error "Pass -Pat or set GITHUB_PAT / GITHUB_TOKEN (must be a PAT: ghp_... or github_pat_..., not gho_ OAuth)."
}
if ($Pat -match '^gho_') {
    Write-Error "This looks like an OAuth token (gho_). Create a PAT at https://github.com/settings/tokens instead."
}

# Verify PAT against GitHub API
$headers = @{ Authorization = "Bearer $Pat"; Accept = "application/vnd.github+json" }
try {
    $null = Invoke-RestMethod -Uri "https://api.github.com/user" -Headers $headers -TimeoutSec 20
} catch {
    Write-Error "PAT rejected by GitHub API: $($_.Exception.Message)"
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
$github = [pscustomobject]@{
    url     = "https://api.githubcopilot.com/mcp"
    headers = [pscustomobject]@{ Authorization = "Bearer $Pat" }
}
$json.mcpServers | Add-Member -NotePropertyName github -NotePropertyValue $github -Force
$json | ConvertTo-Json -Depth 10 | Set-Content $McpPath -Encoding utf8
Write-Host "Updated $McpPath - server name github (remote PAT). Restart Cursor."
Write-Host "Disable the built-in GitHub MCP plugin in Cursor Settings if two GitHub servers appear."
