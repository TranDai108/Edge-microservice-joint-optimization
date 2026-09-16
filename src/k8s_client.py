"""
k8s_client.py — Shared Kubernetes API helper
==============================================================
Replaces all subprocess + kubectl shell calls with the official
`kubernetes` Python client (pip install kubernetes).

Authentication:
  - Inside a K8s pod : config.load_incluster_config()
  - Local / dev env  : config.load_kube_config()  (~/.kube/config)

Usage:
    from k8s_client import (
        get_all_worker_nodes,
        get_node_internal_ip,
        get_node_capacity_cpu,
        get_node_allocatable_memory,
        get_pod_node,
        get_pod_env,
        patch_deployment_node_selector,
        set_deployment_image,
        set_deployment_env,
        rollout_restart_deployment,
        wait_rollout_complete,
    )
"""

import logging
import os
import time
from datetime import datetime, timezone
from functools import lru_cache

from kubernetes import client, config
from kubernetes.client.rest import ApiException

log = logging.getLogger("k8s-client")

NAMESPACE = "default"
MODELED_CPU_ANNOTATION = "kltn.io/modeled-cpu"
MODELED_MEMORY_ANNOTATION = "kltn.io/modeled-memory"
# Read-side calls are used by the dashboard and controller reconciliation
# loops.  They must fail promptly when an API server or an edge node is down;
# otherwise a transient outage can stall an entire Streamlit page render.
K8S_REQUEST_TIMEOUT_S = float(os.getenv("K8S_REQUEST_TIMEOUT_S", "3"))


# ─── Client initialisation (cached singleton) ─────────────────────────────────

@lru_cache(maxsize=1)
def _init_clients() -> tuple[client.CoreV1Api, client.AppsV1Api]:
    """
    Initialise Kubernetes API clients once and cache them.
    Tries in-cluster config first (running inside a pod),
    falls back to local kubeconfig for development.
    """
    try:
        cfg = client.Configuration()
        config.load_incluster_config(client_configuration=cfg)
        # Keep the in-cluster auth config exactly as provided by the
        # kubernetes client to avoid header-key mismatches across versions.
        client.Configuration.set_default(cfg)
        log.debug("Kubernetes: loaded in-cluster config")
    except config.ConfigException:
        config.load_kube_config()
        log.debug("Kubernetes: loaded kubeconfig (~/.kube/config)")

    # Do not use urllib3's retry policy for control-plane reads.  Callers
    # already refresh on a short interval and need a fast degraded response.
    cfg.retries = 0
    api_client = client.ApiClient(configuration=cfg)
    return client.CoreV1Api(api_client), client.AppsV1Api(api_client)


def _core() -> client.CoreV1Api:
    """Return cached CoreV1Api client."""
    core, _ = _init_clients()
    return core


def _apps() -> client.AppsV1Api:
    """Return cached AppsV1Api client."""
    _, apps = _init_clients()
    return apps


# ─── Node resource queries ─────────────────────────────────────────────────────

def _is_node_ready(node) -> bool:
    """
    Return True only if the node's Ready condition is 'True'.

    spec.unschedulable covers manually cordoned nodes, but a node that
    crashes or loses network connectivity is marked Ready=False in its
    status conditions without being cordoned.  Both checks are required
    to reliably exclude unavailable nodes from the schedulable set.
    """
    for condition in (node.status.conditions or []):
        if condition.type == "Ready":
            return condition.status == "True"
    return False


def get_all_worker_nodes() -> list[str]:
    """
    Return schedulable worker node hostnames, sorted.

    Excludes nodes marked unschedulable, nodes carrying
    control-plane/master role labels, and nodes whose Ready
    condition is not True (e.g. crashed or network-partitioned nodes).
    """
    try:
        nodes = _core().list_node(_request_timeout=K8S_REQUEST_TIMEOUT_S).items
        workers: list[str] = []
        for node in nodes:
            name = (node.metadata.name or "").strip()
            if not name:
                continue

            labels = node.metadata.labels or {}
            is_control_plane = (
                "node-role.kubernetes.io/control-plane" in labels
                or "node-role.kubernetes.io/master" in labels
            )
            if is_control_plane:
                continue

            if labels.get("type") == "kwok" or "node-role.kubernetes.io/edge-twin" in labels:
                log.debug(f"  [k8s API] Excluding {name}: KWOK virtual twin node")
                continue

            if node.spec.unschedulable:
                continue

            if not _is_node_ready(node):
                log.warning(
                    f"  [k8s API] Excluding {name}: node condition Ready != True"
                )
                continue

            workers.append(name)

        workers.sort()
        return workers
    except Exception as e:
        log.warning(f"  [k8s API] Could not list worker nodes: {e}")
        return []


def get_all_worker_nodes_any_status() -> list[str]:
    """Return all edge worker node hostnames regardless of Ready/unschedulable state.

    Used by the node-health watchdog to determine the full expected node set so
    that nodes which are already NotReady at pod startup are still tracked.
    """
    try:
        nodes = _core().list_node(_request_timeout=K8S_REQUEST_TIMEOUT_S).items
        workers: list[str] = []
        for node in nodes:
            name = (node.metadata.name or "").strip()
            if not name:
                continue
            labels = node.metadata.labels or {}
            is_control_plane = (
                "node-role.kubernetes.io/control-plane" in labels
                or "node-role.kubernetes.io/master" in labels
            )
            if is_control_plane:
                continue
            if labels.get("type") == "kwok" or "node-role.kubernetes.io/edge-twin" in labels:
                continue
            workers.append(name)
        workers.sort()
        return workers
    except Exception as e:
        log.warning(f"  [k8s API] Could not list all worker nodes: {e}")
        return []


def get_node_internal_ip(hostname: str) -> str | None:
    """
    Return the InternalIP for a node hostname.
    """
    try:
        node = _core().read_node(hostname, _request_timeout=K8S_REQUEST_TIMEOUT_S)
        for addr in node.status.addresses or []:
            if addr.type == "InternalIP" and addr.address:
                return addr.address
        log.warning(f"  [k8s API] No InternalIP found for node {hostname}")
        return None
    except Exception as e:
        log.warning(f"  [k8s API] Could not read node IP for {hostname}: {e}")
        return None

def get_node_capacity_cpu(hostname: str) -> float:
    """
    Read allocatable CPU cores for a node via the Kubernetes API.

    Kubernetes reports CPU as:
      - An integer string ("2") for whole cores
      - A millicore string ("1500m") for fractional cores

    Equivalent to:
        kubectl get node {hostname} -o jsonpath='{.status.allocatable.cpu}'

    Returns 2.0 as a safe fallback if the node is unreachable.
    """
    try:
        node = _core().read_node(hostname, _request_timeout=K8S_REQUEST_TIMEOUT_S)
        raw = node.status.allocatable.get("cpu", "")
        if raw.endswith("m"):
            cap = float(raw[:-1]) / 1000.0
        else:
            cap = float(raw)
        log.debug(f"  [k8s API] {hostname} allocatable CPU: {cap} cores")
        return cap
    except Exception as e:
        log.warning(f"  [k8s API] Could not read CPU capacity for {hostname}: {e}")
        return 2.0


def get_node_allocatable_memory(hostname: str) -> float:
    """
    Read allocatable RAM (GB) for a node via the Kubernetes API.

    Kubernetes reports memory in Ki, Mi, Gi, or raw bytes.

    Equivalent to:
        kubectl get node {hostname} -o jsonpath='{.status.allocatable.memory}'

    Returns 2.0 GB as a safe fallback.
    """
    try:
        node = _core().read_node(hostname, _request_timeout=K8S_REQUEST_TIMEOUT_S)
        raw = node.status.allocatable.get("memory", "")
        if raw.endswith("Ki"):
            mem_gb = int(raw[:-2]) / 1024 / 1024
        elif raw.endswith("Mi"):
            mem_gb = int(raw[:-2]) / 1024
        elif raw.endswith("Gi"):
            mem_gb = float(raw[:-2])
        elif raw.endswith("k") or raw.endswith("K"):
            mem_gb = int(raw[:-1]) * 1000 / 1024 / 1024 / 1024
        else:
            mem_gb = int(raw) / 1024 / 1024 / 1024
        mem_gb = round(mem_gb, 2)
        log.debug(f"  [k8s API] {hostname} allocatable RAM: {mem_gb} GB")
        return mem_gb
    except Exception as e:
        log.warning(f"  [k8s API] Could not read memory for {hostname}: {e}")
        return 2.0


# ─── Pod / Deployment queries ──────────────────────────────────────────────────

def get_pod_node(deploy_name: str, namespace: str = NAMESPACE) -> str | None:
    """
    Return the node name hosting the first Running pod for a deployment.

    Equivalent to:
        kubectl get pods -l app={deploy_name}
            --field-selector status.phase=Running
            -o jsonpath='{.items[0].spec.nodeName}'

    Returns None if no running pod is found.
    """
    try:
        pods = _core().list_namespaced_pod(
            namespace=namespace,
            label_selector=f"app={deploy_name}",
            field_selector="status.phase=Running",
            _request_timeout=K8S_REQUEST_TIMEOUT_S,
        )
        if pods.items:
            node = pods.items[0].spec.node_name
            log.debug(f"  [k8s API] {deploy_name} running on: {node}")
            return node
        log.warning(f"  [k8s API] No running pod found for app={deploy_name}")
        return None
    except Exception as e:
        log.warning(f"  [k8s API] Could not list pods for {deploy_name}: {e}")
        return None


def get_pod_env(deploy_name: str, namespace: str = NAMESPACE) -> list[dict]:
    """
    Return the environment variable list from the first Running pod of a deployment.

    Each entry is a dict with at least {"name": ..., "value": ...}.

    Equivalent to:
        kubectl get pods -l app={deploy_name}
            --field-selector status.phase=Running
            -o jsonpath='{.items[0].spec.containers[0].env}'

    Returns [] if no pod or env vars found.
    """
    try:
        pods = _core().list_namespaced_pod(
            namespace=namespace,
            label_selector=f"app={deploy_name}",
            field_selector="status.phase=Running",
            _request_timeout=K8S_REQUEST_TIMEOUT_S,
        )
        if not pods.items:
            log.warning(f"  [k8s API] No running pod for {deploy_name} — env empty")
            return []
        env_vars = pods.items[0].spec.containers[0].env or []
        # Convert V1EnvVar objects → plain dicts for uniform handling
        return [{"name": e.name, "value": e.value or ""} for e in env_vars]
    except Exception as e:
        log.warning(f"  [k8s API] Could not read env for {deploy_name}: {e}")
        return []


def parse_cpu_quantity(raw: object) -> float:
    """Parse a Kubernetes CPU quantity into cores."""
    text = str(raw or "0").strip()
    if not text:
        return 0.0
    try:
        if text.endswith("m"):
            return max(0.0, float(text[:-1]) / 1000.0)
        return max(0.0, float(text))
    except ValueError:
        log.warning("  [k8s API] Could not parse CPU quantity: %r", raw)
        return 0.0


def parse_memory_quantity_gb(raw: object) -> float:
    """Parse a Kubernetes memory quantity into GiB-style GB units."""
    text = str(raw or "0").strip()
    if not text:
        return 0.0
    units = {
        "Ki": 1 / 1024 / 1024,
        "Mi": 1 / 1024,
        "Gi": 1.0,
        "Ti": 1024.0,
        "K": 1000 / 1024 / 1024 / 1024,
        "M": 1000**2 / 1024 / 1024 / 1024,
        "G": 1000**3 / 1024 / 1024 / 1024,
        "T": 1000**4 / 1024 / 1024 / 1024,
    }
    try:
        for suffix, factor in units.items():
            if text.endswith(suffix):
                return max(0.0, float(text[: -len(suffix)]) * factor)
        return max(0.0, float(text) / 1024 / 1024 / 1024)
    except ValueError:
        log.warning("  [k8s API] Could not parse memory quantity: %r", raw)
        return 0.0


def get_scenario_load_by_node(
    namespace: str = NAMESPACE,
    label_selector: str = "scenario-load=true",
) -> dict[str, dict]:
    """Return MILP-modeled load from scenario pods, grouped by hostname.

    Only Pending and Running pods are counted. By default the load is based on
    Kubernetes resource requests because those are what scheduler admission
    reserves. Demo pods can override the modeled load with
    ``kltn.io/modeled-cpu`` and ``kltn.io/modeled-memory`` annotations, allowing
    controlled MILP pressure scenarios without asking Kubernetes to admit an
    unschedulable request.
    """
    totals: dict[str, dict] = {}
    try:
        pods = _core().list_namespaced_pod(
            namespace=namespace,
            label_selector=label_selector,
            _request_timeout=K8S_REQUEST_TIMEOUT_S,
        ).items
    except Exception as e:
        log.warning("  [k8s API] Could not list scenario load pods: %s", e)
        return totals

    for pod in pods:
        phase = (pod.status.phase or "").strip()
        if phase not in {"Pending", "Running"}:
            continue

        labels = pod.metadata.labels or {}
        node_name = (
            pod.spec.node_name
            or labels.get("scenario-target-node")
            or labels.get("target-node")
            or ""
        )
        if not node_name:
            continue

        entry = totals.setdefault(
            node_name,
            {"cpu_cores": 0.0, "mem_gb": 0.0, "pods": []},
        )
        pod_cpu = 0.0
        pod_mem = 0.0
        for container in pod.spec.containers or []:
            resources = container.resources
            requests = resources.requests if resources and resources.requests else {}
            pod_cpu += parse_cpu_quantity(requests.get("cpu", "0"))
            pod_mem += parse_memory_quantity_gb(requests.get("memory", "0"))

        # Kubernetes client objects normally expose ``annotations`` as ``None``
        # when it is unset.  Keep this tolerant of lightweight API stubs as
        # well, so a pod with no annotations still contributes its requests.
        annotations = getattr(pod.metadata, "annotations", None) or {}
        modeled_cpu_raw = annotations.get(MODELED_CPU_ANNOTATION)
        modeled_mem_raw = annotations.get(MODELED_MEMORY_ANNOTATION)
        modeled_cpu = parse_cpu_quantity(modeled_cpu_raw) if modeled_cpu_raw else pod_cpu
        modeled_mem = parse_memory_quantity_gb(modeled_mem_raw) if modeled_mem_raw else pod_mem

        entry["cpu_cores"] += modeled_cpu
        entry["mem_gb"] += modeled_mem
        entry["pods"].append(
            {
                "name": pod.metadata.name or "",
                "phase": phase,
                "cpu_cores": round(modeled_cpu, 4),
                "mem_gb": round(modeled_mem, 4),
                "request_cpu_cores": round(pod_cpu, 4),
                "request_mem_gb": round(pod_mem, 4),
                "modeled_override": bool(modeled_cpu_raw or modeled_mem_raw),
            }
        )

    for entry in totals.values():
        entry["cpu_cores"] = round(float(entry["cpu_cores"]), 4)
        entry["mem_gb"] = round(float(entry["mem_gb"]), 4)
    return totals


# ─── Deployment mutations ──────────────────────────────────────────────────────

def patch_deployment_node_selector(
    deploy_name: str,
    node_hostname: str,
    namespace: str = NAMESPACE,
) -> None:
    """
    Set nodeSelector on a deployment so its pods are pinned to a specific node.

    Equivalent to:
        kubectl patch deployment {deploy_name} --type='json' \\
            -p='[{"op":"replace","path":"/spec/template/spec/nodeSelector",
                  "value":{"kubernetes.io/hostname":"{node_hostname}"}}]'
    """
    body = {
        "spec": {
            "template": {
                "spec": {
                    "nodeSelector": {
                        "kubernetes.io/hostname": node_hostname
                    }
                }
            }
        }
    }
    try:
        _apps().patch_namespaced_deployment(deploy_name, namespace, body)
        log.info(f"  [k8s API] {deploy_name}: nodeSelector → {node_hostname}")
    except ApiException as e:
        log.error(f"  [k8s API] patch nodeSelector failed for {deploy_name}: {e}")
        raise


def set_deployment_image(
    deploy_name: str,
    container_name: str,
    image: str,
    namespace: str = NAMESPACE,
) -> None:
    """
    Update the container image for a named container in a deployment.

    Uses a strategic merge patch (dict body, no resourceVersion) instead of
    read → modify → full-object-patch. This avoids 409 Conflict errors that
    occur when the object's resourceVersion changes between the read and the
    patch (e.g., because a concurrent nodeSelector patch or a K8s controller
    update happened in between).

    Kubernetes uses `name` as the strategic merge key for containers, so
    specifying just the container name + image is enough to update only that
    container without touching the rest of the spec.

    Equivalent to:
        kubectl set image deployment/{deploy_name} {container_name}={image}
    """
    patch = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {"name": container_name, "image": image}
                    ]
                }
            }
        }
    }
    try:
        _apps().patch_namespaced_deployment(deploy_name, namespace, patch)
        log.info(f"  [k8s API] {deploy_name}/{container_name}: image → {image}")
    except ApiException as e:
        log.error(f"  [k8s API] set image failed for {deploy_name}: {e}")
        raise


def set_deployment_env(
    deploy_name: str,
    env_key: str,
    env_value: str,
    container_name: str | None = None,
    container_index: int = 0,
    namespace: str = NAMESPACE,
) -> None:
    """
    Set (or add) an environment variable on a deployment's container.

    Uses a strategic merge patch (dict body, no resourceVersion). Kubernetes
    uses `name` as the merge key for both containers and env entries, so this
    patch safely adds or overwrites a single env var without a read-modify-
    write cycle and without risking 409 Conflict errors.

    If container_name is None, the first container is targeted via a fallback
    read (index lookup).  In practice, pass container_name when known.

    Equivalent to:
        kubectl set env deployment/{deploy_name} {env_key}={env_value}
    """
    # Resolve container name for the merge-key if not provided
    if container_name is None:
        try:
            dep = _apps().read_namespaced_deployment(deploy_name, namespace)
            container_name = dep.spec.template.spec.containers[container_index].name
        except ApiException as e:
            log.error(f"  [k8s API] Could not resolve container name for {deploy_name}: {e}")
            raise

    patch = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": container_name,
                        "env": [{"name": env_key, "value": env_value}],
                    }]
                }
            }
        }
    }
    try:
        _apps().patch_namespaced_deployment(deploy_name, namespace, patch)
        log.info(f"  [k8s API] {deploy_name}: env {env_key}={env_value}")
    except ApiException as e:
        log.error(f"  [k8s API] set env failed for {deploy_name}: {e}")
        raise


def rollout_restart_deployment(
    deploy_name: str,
    namespace: str = NAMESPACE,
) -> None:
    """
    Trigger a rolling restart of a deployment by annotating its pod template.

    Equivalent to:
        kubectl rollout restart deployment/{deploy_name}

    Uses the same mechanism as kubectl: sets the
    `kubectl.kubernetes.io/restartedAt` annotation to the current UTC timestamp,
    which causes a new ReplicaSet to be created with fresh pods.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    patch = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "kubectl.kubernetes.io/restartedAt": now
                    }
                }
            }
        }
    }
    try:
        _apps().patch_namespaced_deployment(deploy_name, namespace, patch)
        log.info(f"  [k8s API] {deploy_name}: rollout restart triggered at {now}")
    except ApiException as e:
        log.error(f"  [k8s API] rollout restart failed for {deploy_name}: {e}")
        raise


def wait_rollout_complete(
    deploy_name: str,
    timeout_s: int = 60,
    poll_interval_s: float = 2.0,
    namespace: str = NAMESPACE,
) -> bool:
    """
    Poll until a deployment's rollout is complete or the timeout expires.

    Equivalent to:
        kubectl rollout status deployment/{deploy_name} --timeout={timeout_s}s

    A rollout is considered complete when:
        deployment.status.updated_replicas == deployment.spec.replicas
        deployment.status.available_replicas == deployment.spec.replicas
        deployment.status.unavailable_replicas == 0 (or None)

    Returns True on success, False on timeout.
    """
    deadline = time.monotonic() + timeout_s
    log.info(f"  [k8s API] Waiting for {deploy_name} rollout (timeout={timeout_s}s)...")
    while time.monotonic() < deadline:
        try:
            dep = _apps().read_namespaced_deployment(deploy_name, namespace)
            desired   = dep.spec.replicas or 1
            updated   = dep.status.updated_replicas or 0
            available = dep.status.available_replicas or 0
            unavail   = dep.status.unavailable_replicas or 0
            blocker = _detect_rollout_blocker(deploy_name, namespace)
            if blocker:
                log.error(
                    f"  [k8s API] {deploy_name}: rollout blocked by pod={blocker['pod']} "
                    f"reason={blocker['reason']} message={blocker['message']}"
                )
                return False
            if updated >= desired and available >= desired and unavail == 0:
                log.info(f"  [k8s API] {deploy_name}: rollout complete ✓")
                return True
            log.debug(
                f"  [k8s API] {deploy_name}: "
                f"updated={updated}/{desired} available={available} unavail={unavail}"
            )
        except ApiException as e:
            log.warning(f"  [k8s API] Error polling {deploy_name}: {e}")
        time.sleep(poll_interval_s)

    log.warning(
        f"  [k8s API] {deploy_name}: rollout did NOT complete within {timeout_s}s "
        f"(check pod events: kubectl -n {namespace} describe deploy/{deploy_name})"
    )
    return False


def force_delete_stuck_pods(service: str, namespace: str = NAMESPACE) -> int:
    """
    Force-delete (grace_period=0) any pods belonging to a service that are
    scheduled on nodes which are no longer schedulable / Ready.

    When a node goes down its kubelet becomes unreachable, so the normal
    graceful SIGTERM is never delivered and the pod stays in Terminating
    indefinitely.  That Terminating pod holds a replica slot that the
    rolling-update controller will not exceed, preventing the new pod from
    being created on the healthy replacement node.

    Force-deleting removes the pod from etcd immediately so the replica
    controller can proceed to schedule the replacement.

    Returns the number of pods force-deleted.
    """
    available = set(get_all_worker_nodes())
    try:
        pods = _core().list_namespaced_pod(
            namespace=namespace,
            label_selector=f"app={service}",
        ).items
    except ApiException as e:
        log.warning("  [k8s API] Could not list pods for %s: %s", service, e)
        return 0

    deleted = 0
    for pod in pods:
        node = pod.spec.node_name or ""
        if not node or node in available:
            continue
        name = pod.metadata.name or "unknown"
        try:
            _core().delete_namespaced_pod(
                name,
                namespace,
                grace_period_seconds=0,
            )
            log.warning(
                "  [k8s API] Force-deleted stuck pod %s on unavailable node %s",
                name, node,
            )
            deleted += 1
        except ApiException as e:
            log.warning("  [k8s API] Could not force-delete pod %s: %s", name, e)
    return deleted


def _detect_rollout_blocker(deploy_name: str, namespace: str) -> dict | None:
    """Return a blocking pod reason when rollout cannot progress safely."""
    fatal_reasons = {
        "ErrImagePull",
        "ImagePullBackOff",
        "InvalidImageName",
        "CreateContainerConfigError",
        "CreateContainerError",
        "RunContainerError",
        "CrashLoopBackOff",
    }

    try:
        pods = _core().list_namespaced_pod(
            namespace=namespace,
            label_selector=f"app={deploy_name}",
        ).items
    except ApiException:
        return None

    for pod in pods:
        pod_name = pod.metadata.name or "unknown"
        statuses = pod.status.container_statuses or []
        for status in statuses:
            waiting = status.state.waiting if status.state else None
            if not waiting:
                continue
            reason = waiting.reason or "Unknown"
            if reason in fatal_reasons:
                return {
                    "pod": pod_name,
                    "reason": reason,
                    "message": (waiting.message or "").strip(),
                }

    return None
