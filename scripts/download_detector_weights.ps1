param(
    [string]$Repo = $(if ($env:EIMS_DETECTOR_WEIGHTS_REPO) { $env:EIMS_DETECTOR_WEIGHTS_REPO } else { "ICS-MR/Embodied-Intelligence-Enables-Agentic-Exploration-in-Microscopy" }),
    [string]$ReleaseTag = $(if ($env:EIMS_DETECTOR_WEIGHTS_RELEASE_TAG) { $env:EIMS_DETECTOR_WEIGHTS_RELEASE_TAG } else { "detector-weights" }),
    [string]$AssetBaseUrl = $env:EIMS_DETECTOR_WEIGHTS_BASE_URL,
    [string]$TargetRoot,
    [string]$AssetDir,
    [ValidateSet("2Dcell", "organoid", "mitosis", "2Dcell_brightfield", "organoid_fluorescence")]
    [string[]]$Models = @("2Dcell", "organoid", "mitosis", "2Dcell_brightfield", "organoid_fluorescence"),
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $TargetRoot) {
    $TargetRoot = Join-Path $repoRoot "detector_models"
}

$downloadDir = if ($AssetDir) { $AssetDir } else { Join-Path $repoRoot ".runtime/downloads/detector-weights" }

$modelAssets = @{
    "2Dcell" = @{
        AssetName = "2Dcell.pth"
        RelativePath = "cell2d\\weights.pth"
    }
    "organoid" = @{
        AssetName = "organoid.pth"
        RelativePath = "organoid\\weights.pth"
    }
    "mitosis" = @{
        AssetName = "mitosis_best.pth"
        RelativePath = "mitosis\\weights.pth"
    }
    "2Dcell_brightfield" = @{
        AssetName = "2Dcell_brightfield.pth"
        RelativePath = "cell2d_brightfield\\weights.pth"
    }
    "organoid_fluorescence" = @{
        AssetName = "organoid_fluorescence.pth"
        RelativePath = "organoid_fluorescence\\weights.pth"
    }
}

function Get-AssetUrl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AssetName
    )

    if ($AssetBaseUrl) {
        return ($AssetBaseUrl.TrimEnd("/") + "/" + $AssetName)
    }

    return "https://github.com/$Repo/releases/download/$ReleaseTag/$AssetName"
}

function Invoke-AssetDownload {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Uri,

        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

  $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
  if ($curl) {
        $curlArguments = @(
            "--location",
            "--fail",
            "--retry", "5",
            "--retry-delay", "2",
            "--retry-all-errors"
        )

        if (Test-Path -LiteralPath $Destination) {
            $existingLength = (Get-Item -LiteralPath $Destination).Length
            if ($existingLength -gt 0) {
                Write-Host "Resuming partial download ($existingLength bytes already received) ..."
                $curlArguments += @("--continue-at", "-")
            }
        }

        $curlArguments += @("--output", $Destination, $Uri)
        & $curl.Source @curlArguments
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed to download '$Uri' (exit code $LASTEXITCODE). Re-run the script to resume the partial download."
        }
        return
  }

    Write-Warning "curl.exe is unavailable; falling back to a non-resumable PowerShell download."
    $maximumAttempts = 5
    for ($attempt = 1; $attempt -le $maximumAttempts; $attempt++) {
        try {
            Invoke-WebRequest -Uri $Uri -OutFile $Destination
            return
        }
        catch {
            if ($attempt -eq $maximumAttempts) {
                throw
            }

            Write-Warning "Download attempt $attempt failed: $($_.Exception.Message)"
            Start-Sleep -Seconds 2
        }
    }
}

New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null

foreach ($modelName in $Models) {
    if (-not $modelAssets.ContainsKey($modelName)) {
        throw "Unsupported detector weight group: $modelName"
    }

    $asset = $modelAssets[$modelName]
    $assetName = $asset.AssetName
    $targetPath = Join-Path $TargetRoot $asset.RelativePath
    $localDownloadPath = Join-Path $downloadDir $assetName

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $targetPath) | Out-Null

    if ((Test-Path $targetPath) -and (-not $Force)) {
        Write-Host "Using existing detector weight: $targetPath"
        continue
    }

    $assetUrl = Get-AssetUrl -AssetName $assetName
    Write-Host "Downloading $assetName from release '$ReleaseTag' ..."
    Invoke-AssetDownload -Uri $assetUrl -Destination $localDownloadPath

    Copy-Item -LiteralPath $localDownloadPath -Destination $targetPath -Force
    Write-Host "Installed detector weight to $targetPath"
}
