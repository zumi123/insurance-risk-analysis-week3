# Task 2 – Reproducible Data Pipeline with DVC

This task sets up a reproducible and auditable data pipeline using **Data Version Control (DVC)**, essential for traceability and compliance in regulated industries such as finance and insurance.

---

## Objectives

- Version control large datasets just like code
- Track data changes across experiments and tasks
- Enable reproducibility of results
- Set up a local DVC remote for secure storage

---

## Steps Completed

### 1. Initialized DVC

```bash
pip install dvc
dvc init
```

### 2. Set Up Local Remote

```bash
mkdir -p /home/zumi/dvc-storage
dvc remote add -d localstorage /home/zumi/dvc-storage
```

### 3. Tracked Dataset with DVC

```bash
dvc add MachineLearningRating_v3.txt
```

### 4. Committed DVC Metadata to Git

```bash
git add MachineLearningRating_v3.txt.dvc .gitignore .dvc .dvcignore
git commit -m " Add dataset to DVC and setup reproducible pipeline"
```

### 5. Pushed Data to Local Remote

```bash
dvc push
```

---

## Files Added

- `MachineLearningRating_v3.txt.dvc`: Data tracking metadata
- `.dvcignore`: Optional, to exclude files from DVC tracking
- `.dvc/`: DVC system folder
- `.dvc/config`: Stores remote config info

---

## Notes

- The `.dvc` file is tracked by Git and acts like a pointer to the actual data
- The real data is stored in `.dvc/cache` and/or the configured remote (`/home/zumi/dvc-storage`)
- This setup ensures reproducibility and allows restoring any past state using `dvc pull`

