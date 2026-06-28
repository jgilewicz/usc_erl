for d in ./wandb_logs/wandb_logs/wandb/offline-run-*/; do
    .venv/bin/wandb sync "$d" || echo "FAILED: $d"
done
