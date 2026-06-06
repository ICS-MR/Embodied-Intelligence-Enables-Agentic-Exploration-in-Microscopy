$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$wheelFilename = "mmcv-2.1.0-cp310-cp310-win_amd64.whl"
$downloadDir = Join-Path $repoRoot ".runtime/downloads"
$downloadedWheel = Join-Path $downloadDir $wheelFilename
$remoteIndex = "https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html"
$releaseTag = if ($env:EIMS_MMCV_RELEASE_TAG) { $env:EIMS_MMCV_RELEASE_TAG } else { "mmcv-fallback" }
$releaseUrl = if ($env:EIMS_MMCV_FALLBACK_URL) {
    $env:EIMS_MMCV_FALLBACK_URL
}
else {
    "https://github.com/ICS-MR/Embodied-Intelligence-Enables-Agentic-Exploration-in-Microscopy/releases/download/$releaseTag/$wheelFilename"
}

function Install-WheelFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$WheelPath
    )

    & uv pip install --no-deps --force-reinstall $WheelPath
    if ($LASTEXITCODE -ne 0) {
        throw "Wheel installation failed: $WheelPath"
    }
}

Push-Location $repoRoot
try {
    Write-Host "Installing all dependencies except mmcv..."
    & uv sync --frozen --no-install-package mmcv
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed before mmcv installation."
    }

    Write-Host "Trying remote mmcv wheel from OpenMMLab..."
    & uv pip install --no-deps --force-reinstall "mmcv==2.1.0" -f $remoteIndex
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Installed mmcv from remote source."
    }
    else {
        Write-Warning "Remote mmcv install failed. Trying GitHub Release fallback."

        New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
        try {
            Invoke-WebRequest -Uri $releaseUrl -OutFile $downloadedWheel
            Install-WheelFile -WheelPath $downloadedWheel
            Write-Host "Installed mmcv from GitHub Release fallback."
        }
        catch {
            $releaseError = $_.Exception.Message
            throw (
                "GitHub Release fallback failed.`n" +
                "Release URL: $releaseUrl`n" +
                "Inner error: $releaseError"
            )
        }
    }
}
finally {
    Pop-Location
}
