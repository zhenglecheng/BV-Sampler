$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$runId = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$seeds = 43, 44, 45

# The diagnostic runner loads ../data and writes checkpoints to ./tmp.
# Give each condition a fresh directory directly under the repository root.
$commonArgs = @(
    '--dataset', 'ogbn-arxiv',
    '--norm_feat', 'false',
    '--fast', 'true',
    '--depths', '2',
    '--batch_norm', 'true',
    '--hidden_dim', '256',
    '--init_batch', '1024',
    '--sample_size', '10240',
    '--scale_factor', '5',
    '--wd', '1e-4',
    '--drop', '0.5',
    '--lr', '0.001',
    '--epochs', '1000',
    '--early_stop', '300',
    '--label_rate', '0.025',
    '--labeled_ratio', '0.9',
    '--alpha_mode', 'optimal',
    '--sampling_replacement', 'false',
    '--compute_bvm', 'true',
    '--bvm_compare_alpha', 'false',
    '--bvm_replacement', 'true',
    '--bvm_samples', '100',
    '--bvm_nodes', '500',
    '--bvm_seed', '2026',
    '--bvm_split', 'validation'
)

$conditions = @(
    @{ Name = 'no-contrastive'; Weight = '0' },
    @{ Name = 'with-contrastive'; Weight = '0.5' }
)

foreach ($seed in $seeds) {
    foreach ($condition in $conditions) {
        $runDir = Join-Path $repoRoot "bvm-ogbn-arxiv-$($condition.Name)-seed-$seed-$runId"
        New-Item -ItemType Directory -Path $runDir | Out-Null
        Copy-Item -Path (Join-Path $PSScriptRoot '*.py') -Destination $runDir
        Push-Location -LiteralPath $runDir
        try {
            Write-Host "Running $($condition.Name) with seed $seed in $runDir"
            conda run --no-capture-output -n graphany python ./main_alpha_1.py @commonArgs `
                --seed $seed --con_weight $condition.Weight --bvm_output bvm.csv
            if ($LASTEXITCODE -ne 0) {
                throw "Run failed for $($condition.Name), seed $seed (exit code $LASTEXITCODE)."
            }
            Write-Host "Saved $(Join-Path $runDir 'bvm.csv')"
        } finally {
            Pop-Location
        }
    }
}
