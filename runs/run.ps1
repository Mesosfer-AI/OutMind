# OutMind Unified One-Command Runner (PowerShell)
# Usage:
#   .\runs\run.ps1 [nano|small|medium|large] [-mode quick|full]
param (
    [string]$Preset = "nano",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "   OutMind 1-Command Pipeline Launcher" -ForegroundColor Cyan
Write-Host "   Preset: $Preset" -ForegroundColor Yellow
Write-Host "=================================================" -ForegroundColor Cyan

switch ($Preset.ToLower()) {
    "nano" {
        & python runs/run_nano.py $RemainingArgs
    }
    "small" {
        & python runs/run_small.py $RemainingArgs
    }
    "medium" {
        & python runs/run_medium.py $RemainingArgs
    }
    "large" {
        & python runs/run_large.py $RemainingArgs
    }
    default {
        & python runs/pipeline.py --preset $Preset $RemainingArgs
    }
}
