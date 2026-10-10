# Troubleshooting

> One entry per error from the error matrix. Will be populated as modules are built.

## Camera Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Cannot open camera" | Camera in use or wrong index | Close other apps; try a different camera index |
| "Permission denied" | OS camera privacy settings | Windows: Settings → Privacy → Camera; macOS: System Settings → Privacy → Camera |
| Black screen | Lens cap or driver issue | Check lens cap; update drivers |

## Model Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Model file not found" | `.task` file missing | Run `python -m posesense download-model` |
| "Schema version mismatch" | Model trained with different features | Retrain the model |
