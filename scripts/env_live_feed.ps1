# scripts\env_live_feed.ps1
$dir = "C:\Users\julia\AppData\Roaming\MetaQuotes\Terminal\Common\Files\anchor_reversion_fx\prod\v8_policy_r1\live_feed"
$env:MT5  = Join-Path $dir "eurusd_m5_latest.csv"
$env:FEAT = Join-Path $dir "eurusd_m5_features_latest.csv"