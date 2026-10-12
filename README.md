# Churn MLOps: an end-to-end ML pipeline with CI/CD

A customer-churn model taken all the way from training to production, with the
automation an MLOps team would expect: experiment tracking, a **model quality gate**,
containerised serving, vulnerability scanning, automated deployment, and monitoring.

```
 git push ─► Lint + tests ─► Train + quality gate ─► Docker build ─► Trivy scan ─► Docker Hub ─► EC2
                                   │                                                          │
                                   └─► MLflow (params, metrics, model)        Prometheus ◄── /metrics ──► Grafana
```

## What's in the box

| Concern | Tooling |
|---|---|
| Model | scikit-learn `Pipeline` (preprocessing + GradientBoosting), ships as one artifact |
| Experiment tracking | MLflow (params, metrics, model, gate verdict per run) |
| Quality gate | Training exits non-zero if ROC AUC < floor or regresses vs. production |
| Serving | FastAPI, validated inputs (Pydantic), `/health`, `/predict`, `/metrics` |
| Packaging | Multi-stage Dockerfile, non-root user, healthcheck |
| CI/CD | GitHub Actions: lint, test, train, build, scan, push, deploy |
| Security | Trivy image scan fails the build on HIGH/CRITICAL fixable CVEs |
| Monitoring | Prometheus + Grafana (latency, request rate, prediction distribution) |

The dataset is synthetic and seeded (`src/churn_mlops/data.py`), so CI needs no downloads and
you can **simulate data drift** on demand.

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
make install
make train          # trains, logs to MLflow (./mlruns), applies the gate, saves models/model.joblib
make test
make run            # API at http://localhost:8000/docs
```

Full stack with monitoring:

```bash
make up             # api :8000 | mlflow :5000 | prometheus :9090 | grafana :3000 (admin/admin)
make traffic        # normal traffic
make drift-traffic  # drifted traffic: watch churn_probability shift in Prometheus/Grafana
```

Example request:

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "tenure_months": 3, "monthly_charges": 95, "total_charges": 285, "support_tickets": 4,
  "avg_monthly_usage_gb": 80, "contract_type": "month-to-month",
  "payment_method": "e-wallet", "has_internet_backup": "no"}'
```

## The quality gate

`src/churn_mlops/train.py` evaluates on a held-out split and refuses to promote a model when:

1. `roc_auc < MIN_ROC_AUC` (default 0.70), or
2. it regresses against `models/metrics.json` (the current production model) by more than
   `MIN_IMPROVEMENT` (default -0.005).

A failed gate returns exit code 1, which fails the `train` job and stops the pipeline before anything is built or deployed. Tune it with environment variables.

## CI/CD setup

Add these in **GitHub, Settings, Secrets and variables, Actions**:

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` | Docker Hub user and an access token |
| `EC2_HOST` | Public IP/DNS of the EC2 instance |
| `EC2_USER` | e.g. `ubuntu` or `ec2-user` |
| `EC2_SSH_KEY` | Private key (PEM contents) for that instance |

EC2 prerequisites: Docker installed, your user in the `docker` group, port 80 open in the security group.

Pull requests run lint, tests, training, build and scan. Only pushes to `main` push the image and deploy.
The deploy step ends with a health check and fails the run if the new container is unhealthy.

## Project layout

```
src/churn_mlops/   config, data generator, model pipeline, training + gate, FastAPI app
tests/             data, gate logic, training promotion/blocking, API contract and metrics
monitoring/        Prometheus + Grafana provisioning
scripts/           traffic / drift simulator
.github/workflows/ ci-cd.yml
```

## Roadmap (good next steps)

- Compare against the *last deployed* model in CI (store `metrics.json` in S3 or an MLflow registry stage) instead of only the absolute AUC floor.
- Scheduled drift check (Evidently) that triggers a retraining workflow.
- Terraform for the EC2 instance, security group and key pair.
- Blue/green or canary deploys; Grafana dashboard JSON committed to the repo.
