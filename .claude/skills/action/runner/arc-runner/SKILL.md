---
name: arc-runner-solution
description: Use for the ARC + K8s GitHub Actions runner solution for intel/torch-xpu-ops, including why it is used, what it provides, how to implement it, and how it works.
---

# ARC Runner Solution

## 1. Why To Use

Use ARC + K8s to provide stable, elastic GitHub Actions runners for `intel/torch-xpu-ops` on the local node.

This solution is preferred because it:

- Runs jobs in clean ephemeral runner pods.
- Scales from zero idle runners to the required job capacity while leaving node headroom for ARC and K8s control components.
- Avoids managing many long-running runner services by hand.
- Uses Kubernetes scheduling, isolation, lifecycle management, and resource limits.
- Bakes common tools into the runner image to avoid installing them during every job.
- Reuses selected cache directories across ephemeral pods to reduce setup time.
- Keeps `ubuntu-24.04` GitHub-hosted runners separate from the local self-hosted label.

## 2. What Can Do

The active setup provides:

- Repository: `https://github.com/intel/torch-xpu-ops`
- Workflow label: `xpu-ubuntu-24.04`
- Helm release: `xpu-ubuntu-24-04`
- Runner namespace: `arc-runners`
- Controller namespace: `arc-systems`
- Runner image: `arc-xpu-runner:ubuntu-24.04-tools`
- Preinstalled tools: `git`, `gh`, `curl`, `wget`, `rsync`, `ca-certificates`
- Default timezone: UTC
- Runner OS: Ubuntu 24.04
- Minimum runners: `0`
- Maximum runners: `12`
- Per-runner resources: `4` CPU and `16Gi` memory
- Total runner CPU budget: `48` whole CPUs, below the node physical-core count so controller and system processes keep reserved headroom
- Target node: `skx3725`
- Runner pods use host-aligned UID/GID `1000:1000`, are non-privileged, use RuntimeDefault seccomp, drop Linux capabilities, and do not mount a K8s service account token.
- Cache mounts: `/home/runner/.cache` and `/opt/hostedtoolcache`

Workflows that should use this node must set:

```yaml
runs-on: xpu-ubuntu-24.04
```

## 3. How To Implement

Store the scale set configuration in `arc-xpu-ubuntu-24.04-values.yaml`:

```yaml
githubConfigUrl: https://github.com/intel/torch-xpu-ops
githubConfigSecret: github-token

runnerScaleSetName: xpu-ubuntu-24.04
scaleSetLabels:
  - ubuntu-24.04

minRunners: 0
maxRunners: 12

template:
  spec:
    automountServiceAccountToken: false
    nodeSelector:
      kubernetes.io/hostname: skx3725
    securityContext:
      seccompProfile:
        type: RuntimeDefault
    initContainers:
      - name: prepare-runner-cache
        image: busybox:1.36
        command:
          - sh
          - -c
          - chown -R 1000:1000 /home-cache /tool-cache
        securityContext:
          privileged: false
          allowPrivilegeEscalation: false
          runAsUser: 0
          runAsGroup: 0
          capabilities:
            drop:
              - ALL
            add:
              - CHOWN
        volumeMounts:
          - name: runner-home-cache
            mountPath: /home-cache
          - name: runner-tool-cache
            mountPath: /tool-cache
    containers:
      - name: runner
        image: arc-xpu-runner:ubuntu-24.04-tools
        imagePullPolicy: IfNotPresent
        command:
          - /home/runner/run.sh
        env:
          - name: ACTIONS_RUNNER_TOOL_CACHE
            value: /opt/hostedtoolcache
        securityContext:
          privileged: false
          allowPrivilegeEscalation: false
          runAsNonRoot: true
          runAsUser: 1000
          runAsGroup: 1000
          capabilities:
            drop:
              - ALL
        resources:
          requests:
            cpu: "4"
            memory: 16Gi
          limits:
            cpu: "4"
            memory: 16Gi
        volumeMounts:
          - name: runner-home-cache
            mountPath: /home/runner/.cache
          - name: runner-tool-cache
            mountPath: /opt/hostedtoolcache
    volumes:
      - name: runner-home-cache
        hostPath:
          path: /var/cache/arc/xpu-ubuntu-24.04/home-cache
          type: DirectoryOrCreate
      - name: runner-tool-cache
        hostPath:
          path: /var/cache/arc/xpu-ubuntu-24.04/tool-cache
          type: DirectoryOrCreate
```

Build and import the custom image on the node before deploying the scale set:

```bash
docker build -t arc-xpu-runner:ubuntu-24.04-tools -f Dockerfile.arc-runner-ubuntu-24.04 .
docker save arc-xpu-runner:ubuntu-24.04-tools -o /tmp/arc-xpu-runner-ubuntu-24.04-tools.tar
sudo -n ctr images import /tmp/arc-xpu-runner-ubuntu-24.04-tools.tar
rm -f /tmp/arc-xpu-runner-ubuntu-24.04-tools.tar
```

Deploy or update the scale set:

```bash
export KUBECONFIG=/path/to/k8s/k8s.yaml
helm upgrade --install xpu-ubuntu-24-04 \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --namespace arc-runners \
  --create-namespace \
  -f arc-xpu-ubuntu-24.04-values.yaml \
  --set controllerServiceAccount.name=arc-gha-rs-controller \
  --set controllerServiceAccount.namespace=arc-systems \
  --timeout 10m
```

Credential material must never be printed, logged, pasted, diffed, or committed. Keep the ARC credential object in K8s while the scale set is installed.

## 4. How To Work

ARC creates runners only when GitHub Actions has matching queued jobs.

Normal flow:

1. A workflow job requests `runs-on: xpu-ubuntu-24.04`.
2. The ARC listener detects queued demand for the scale set.
3. The ARC controller creates an ephemeral runner pod in `arc-runners`.
4. The runner registers to `intel/torch-xpu-ops` and executes one job.
5. After the job finishes, the runner pod exits and is removed.
6. When no jobs are queued, the scale set returns to zero runner pods.

Healthy idle state:

- Controller pod is running in `arc-systems`.
- Listener pod is running in `arc-runners`.
- `AutoscalingRunnerSet/xpu-ubuntu-24.04` has minimum `0` and maximum `12`.
- No runner pods are running when no matching jobs are queued.
