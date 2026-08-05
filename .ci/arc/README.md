# ARC Runner Implementation

This directory contains the implementation files for running GitHub Actions jobs for `intel/torch-xpu-ops` on a local Kubernetes node through Actions Runner Controller (ARC).

The runner label exposed to workflows is:

```yaml
runs-on: xpu-ubuntu-24.04
```

The same scale set also advertises the compatibility label `ubuntu-latest`.

## Target Setup

- Repository: `https://github.com/intel/torch-xpu-ops`
- Runner scale set name: `xpu-ubuntu-24.04`
- Scale set labels: `xpu-ubuntu-24.04`, `ubuntu-24.04`, `ubuntu-latest`
- Helm release: `xpu-ubuntu-24-04`
- Controller namespace: `arc-systems`
- Runner namespace: `arc-runners`
- Runner image: `arc-xpu-runner:ubuntu-24.04-tools`
- Target node: `skx3725`
- Minimum runners: `0`
- Maximum runners: `12`
- Per-runner resources: `4` CPU and `16Gi` memory
- Cache mounts: `/home/runner/.cache` and `/opt/hostedtoolcache`
- The runner container detects the host cache UID/GID at startup and runs the GitHub runner with that same identity.
- Cache ownership is restored from the runner entrypoint and the Kubernetes `preStop` hook before the container exits.

## Files

- `Dockerfile.arc-runner-ubuntu-24.04`: builds the Ubuntu 24.04 runner image with common CI tools.
- `arc-xpu-ubuntu-24.04-values.yaml`: configures the ARC runner scale set.
- `arc-runner-entrypoint.sh`: detects host cache ownership, rewrites the runner user, starts the runner, and triggers cleanup on exit.
- `arc-runner-cleanup.sh`: restores cache ownership to the detected host UID/GID before pod deletion.
- `github-token.secret.example.yaml`: documents the required Kubernetes secret shape without storing credentials.

## 1. Prepare Access

Create a GitHub token with access to `intel/torch-xpu-ops` and permissions required by ARC to register repository runners.

Do not print, log, commit, or paste the token into files. Create the Kubernetes secret directly from the shell environment:

```bash
export KUBECONFIG=/path/to/k8s/k8s.yaml
export GITHUB_TOKEN=...

kubectl create namespace arc-runners --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic github-token \
  --namespace arc-runners \
  --from-literal=github_token="${GITHUB_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -
unset GITHUB_TOKEN
```

The committed `github-token.secret.example.yaml` file is only a schema reference.

## 2. Install Or Update ARC Controller

Install the ARC controller into `arc-systems` if it is not already present:

```bash
export KUBECONFIG=/path/to/k8s/k8s.yaml

helm upgrade --install arc \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller \
  --namespace arc-systems \
  --create-namespace \
  --timeout 10m
```

Confirm the controller service account exists:

```bash
kubectl get serviceaccount arc-gha-rs-controller -n arc-systems
```

## 3. Build The Runner Image

Build the image on the Kubernetes node that will run the ARC pods:

```bash
docker build \
  -t arc-xpu-runner:ubuntu-24.04-tools \
  -f .ci/arc/Dockerfile.arc-runner-ubuntu-24.04 \
  .ci/arc
```

Import the image into the node container runtime used by Kubernetes:

```bash
docker save arc-xpu-runner:ubuntu-24.04-tools -o /tmp/arc-xpu-runner-ubuntu-24.04-tools.tar
sudo -n ctr images import /tmp/arc-xpu-runner-ubuntu-24.04-tools.tar
rm -f /tmp/arc-xpu-runner-ubuntu-24.04-tools.tar
```

If the cluster uses a registry instead of local image import, push the image to that registry and update `template.spec.containers[0].image` in `arc-xpu-ubuntu-24.04-values.yaml`.

## 4. Prepare Host Cache Directories

Create the host cache directories on `skx3725` before deploying the scale set:

```bash
host_uid=$(id -u)
host_gid=$(id -g)

sudo -n mkdir -p \
  /var/cache/arc/xpu-ubuntu-24.04/home-cache \
  /var/cache/arc/xpu-ubuntu-24.04/tool-cache
sudo -n chown -R "${host_uid}:${host_gid}" /var/cache/arc/xpu-ubuntu-24.04
```

The runner pod checks those cache directory owners at startup. If they are not `0:0`, it updates the in-container `runner` user to the same UID/GID and runs `/home/runner/run.sh` as that user. The `ARC_RUNNER_UID`/`ARC_RUNNER_GID` values are only fallbacks for newly created root-owned host paths.

Before the pod exits, both the entrypoint trap and the Kubernetes `preStop` hook run `arc-runner-cleanup.sh` to `chown` the mounted caches back to the detected host UID/GID.

## 5. Deploy The Runner Scale Set

Deploy or update the scale set:

```bash
export KUBECONFIG=/path/to/k8s/k8s.yaml

helm upgrade --install xpu-ubuntu-24-04 \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --namespace arc-runners \
  --create-namespace \
  -f .ci/arc/arc-xpu-ubuntu-24.04-values.yaml \
  --set controllerServiceAccount.name=arc-gha-rs-controller \
  --set controllerServiceAccount.namespace=arc-systems \
  --timeout 10m
```

## 6. Verify The Idle State

An idle but healthy deployment should have a running controller, a running listener, and no runner pods until matching jobs are queued:

```bash
kubectl get pods -n arc-systems
kubectl get pods -n arc-runners
kubectl get autoscalingrunnerset -n arc-runners
```

Expected properties:

- `AutoscalingRunnerSet/xpu-ubuntu-24.04` exists.
- Minimum runners is `0` and maximum runners is `12`.
- Listener pod is running in `arc-runners`.
- Runner pods appear only while jobs with `runs-on: xpu-ubuntu-24.04` are queued or running.

## 7. Use The Runner In Workflows

Set jobs that should run on this ARC scale set to:

```yaml
runs-on: xpu-ubuntu-24.04
```

Jobs that still use `ubuntu-latest` can also be picked up by this scale set because `ubuntu-latest` is included in `scaleSetLabels`.

The runner pod lifecycle is:

1. GitHub Actions queues a job with the `xpu-ubuntu-24.04` label.
2. The ARC listener detects demand for this scale set.
3. ARC creates an ephemeral runner pod in `arc-runners`.
4. The runner registers to `intel/torch-xpu-ops` and executes one job.
5. The runner pod exits and is removed after the job finishes.
6. The scale set returns to zero runner pods when no matching jobs are queued.

## 8. Update The Deployment

After changing `Dockerfile.arc-runner-ubuntu-24.04`, rebuild and re-import the image, then restart the scale set pods:

```bash
docker build \
  -t arc-xpu-runner:ubuntu-24.04-tools \
  -f .ci/arc/Dockerfile.arc-runner-ubuntu-24.04 \
  .ci/arc
docker save arc-xpu-runner:ubuntu-24.04-tools -o /tmp/arc-xpu-runner-ubuntu-24.04-tools.tar
sudo -n ctr images import /tmp/arc-xpu-runner-ubuntu-24.04-tools.tar
rm -f /tmp/arc-xpu-runner-ubuntu-24.04-tools.tar

helm upgrade --install xpu-ubuntu-24-04 \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --namespace arc-runners \
  --create-namespace \
  -f .ci/arc/arc-xpu-ubuntu-24.04-values.yaml \
  --set controllerServiceAccount.name=arc-gha-rs-controller \
  --set controllerServiceAccount.namespace=arc-systems \
  --timeout 10m
```

## 9. Remove The Scale Set

Remove only the runner scale set:

```bash
helm uninstall xpu-ubuntu-24-04 -n arc-runners
```

Remove the token secret if this node should no longer host repository runners:

```bash
kubectl delete secret github-token -n arc-runners
```