# CSCI5527 Final

## Run training with SLURM

If you are in the project root:

`/users/7/yu001011/csci5527/CSCI5527-final`

submit the job with:

```bash
sbatch baseline_models_scripts/train_ttd.slurm
```

The script already resolves paths from its own location and runs:

- baseline_models_scripts/train_ttd.py
- project root: /users/7/yu001011/csci5527/CSCI5527-final
- venv: /users/7/yu001011/csci5527/.venv

## Optional: run with original command

If you want to run `sbatch train_ttd.slurm`, first move into the script folder:

```bash
cd baseline_models_scripts
sbatch train_ttd.slurm
```

## Useful checks

Create log folder if needed:

```bash
mkdir -p /users/7/yu001011/csci5527/logs
```

Check queue:

```bash
squeue -u $USER
```

See the current log

```bash
cat /users/7/yu001011/csci5527/logs/ttd-train-*.out
```

