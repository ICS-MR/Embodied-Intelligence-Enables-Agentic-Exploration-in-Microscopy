param(
    [string]$Repo = $(if ($env:EIMS_VLA_ACT_REPO) { $env:EIMS_VLA_ACT_REPO } else { "404lzh/ACT_for_microscopy" }),
    [string]$Revision = $(if ($env:EIMS_VLA_ACT_REVISION) { $env:EIMS_VLA_ACT_REVISION } else { "main" }),
    [string]$AssetBaseUrl = $env:EIMS_VLA_ACT_BASE_URL,
    [string]$TargetRoot,
    [string]$AssetDir,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $TargetRoot) {
    $TargetRoot = Join-Path $repoRoot "docs/VLA"
}

$targetDir = Join-Path $TargetRoot "ACT_for_microscopy"
$downloadDir = if ($AssetDir) { $AssetDir } else { Join-Path $repoRoot ".runtime/downloads/vla-act-assets" }

$assetNames = @(
    "ACT_for_microscopy_root.zip",
    "ACT_for_microscopy_2D_move_none.zip",
    "ACT_for_microscopy_2D_set_brightness_none.zip",
    "ACT_for_microscopy_2D_set_z_funa.zip",
    "ACT_for_microscopy_2D_set_z_none.zip",
    "ACT_for_microscopy_Cell_move_none.zip",
    "ACT_for_microscopy_Cell_set_brightness_none.zip",
    "ACT_for_microscopy_Cell_set_z_funa.zip",
    "ACT_for_microscopy_Cell_set_z_none.zip",
    "ACT_for_microscopy_Push_to_target.zip",
    "ACT_for_microscopy_Slice_move_none.zip",
    "ACT_for_microscopy_Slice_set_brightness_none.zip",
    "ACT_for_microscopy_Slice_set_z_fbna.zip",
    "ACT_for_microscopy_Slice_set_z_none.zip"
)

function Get-AssetUrl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    if ($AssetBaseUrl) {
        return ($AssetBaseUrl.TrimEnd('/') + "/" + $Name)
    }

    return "https://huggingface.co/$Repo/resolve/$Revision/$Name"
}

if (Test-Path $targetDir) {
    if (-not $Force) {
        throw "Target directory already exists: $targetDir`nRe-run with -Force to replace it."
    }

    Remove-Item -LiteralPath $targetDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null

foreach ($assetName in $assetNames) {
    $localZip = Join-Path $downloadDir $assetName

    if (-not (Test-Path $localZip)) {
        $assetUrl = Get-AssetUrl -Name $assetName
        Write-Host "Downloading $assetName ..."
        Invoke-WebRequest -Uri $assetUrl -OutFile $localZip
    }
    else {
        Write-Host "Using existing asset: $localZip"
    }

    Write-Host "Extracting $assetName ..."
    Expand-Archive -LiteralPath $localZip -DestinationPath $TargetRoot -Force
}

if (-not (Test-Path $targetDir)) {
    throw "Restore finished but target directory was not created: $targetDir"
}

Write-Host "Restored VLA ACT assets to $targetDir"
