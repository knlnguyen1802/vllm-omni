# Executor Comparison Guide

## Overview

This guide compares all available executor backends for the diffusion engine.

## Executor Types

### 1. MultiProcDiffusionExecutor (Default)

**Backend:** `"mp"`

**Description:** Launches worker processes using Python multiprocessing.

**Use Case:** Single-node, multi-GPU setups with simple deployment.

```python
config = OmniDiffusionConfig(
    model="model-path",
    distributed_executor_backend="mp",  # Default
)
```

**Pros:**
- ✅ No external dependencies
- ✅ Simple and reliable
- ✅ Good for single-node setups
- ✅ Direct process control

**Cons:**
- ❌ Single-node only
- ❌ No fault tolerance
- ❌ Workers die with engine

### 2. RayActorExecutor

**Backend:** `"ray_actor"`

**Description:** Creates a Ray actor that manages workers internally.

**Use Case:** Simple Ray integration, single or multi-node setups.

```python
config = OmniDiffusionConfig(
    model="model-path",
    distributed_executor_backend="ray_actor",
)
```

**Pros:**
- ✅ Simple API (like AsyncOmniDiffusion)
- ✅ Multi-node support via Ray
- ✅ Automatic lifecycle management
- ✅ Ray dashboard monitoring
- ✅ Fault tolerance (if Ray configured)

**Cons:**
- ❌ Requires Ray installation
- ❌ Ray overhead
- ❌ Single actor (not fully distributed)

### 3. ExternalDiffusionExecutor

**Backend:** `"external_launcher"` or custom class

**Description:** Connects to externally managed workers (base class).

**Use Case:** Custom orchestration systems, advanced deployments.

```python
config = OmniDiffusionConfig(
    model="model-path",
    distributed_executor_backend=MyCustomExecutor,
)
```

**Pros:**
- ✅ Maximum flexibility
- ✅ Workers persist across engine restarts
- ✅ Custom orchestration (K8s, etc.)
- ✅ Advanced deployment scenarios

**Cons:**
- ❌ Requires custom implementation
- ❌ More complex to set up
- ❌ Manual worker management

### 4. RayExternalDiffusionExecutor

**Backend:** `RayExternalDiffusionExecutor` class

**Description:** Connects to pre-existing Ray actors.

**Use Case:** When Ray actors are managed externally.

```python
config = OmniDiffusionConfig(
    model="model-path",
    distributed_executor_backend=RayExternalDiffusionExecutor,
)
```

**Pros:**
- ✅ Decouple worker lifecycle
- ✅ Workers persist independently
- ✅ Flexible deployment
- ✅ External orchestration

**Cons:**
- ❌ Manual actor creation
- ❌ More setup required
- ❌ Need to manage actor lifecycle

## Feature Comparison

| Feature | MultiProc | RayActor | External | RayExternal |
|---------|-----------|----------|----------|-------------|
| **Installation** | Built-in | Ray | Custom | Ray |
| **Complexity** | Low | Low | High | Medium |
| **Multi-Node** | ❌ | ✅ | ✅ | ✅ |
| **Fault Tolerance** | ❌ | ⚠️ | ✅ | ✅ |
| **Lifecycle Management** | Automatic | Automatic | Manual | Manual |
| **Worker Persistence** | ❌ | ❌ | ✅ | ✅ |
| **Custom Orchestration** | ❌ | ❌ | ✅ | ⚠️ |
| **Monitoring** | Basic | Ray Dashboard | Custom | Ray Dashboard |
| **API Simplicity** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

## When to Use Which

### Use MultiProcDiffusionExecutor when:
- ✅ Single-node deployment
- ✅ Simple setup required
- ✅ No Ray available
- ✅ Direct process control needed

### Use RayActorExecutor when:
- ✅ Want simple API (like AsyncOmniDiffusion)
- ✅ Multi-node capability desired
- ✅ Ray already in use
- ✅ Ray monitoring preferred
- ✅ **Recommended for most use cases**

### Use ExternalDiffusionExecutor when:
- ✅ Custom orchestration system (K8s, etc.)
- ✅ Workers managed separately
- ✅ Advanced deployment requirements
- ✅ Maximum flexibility needed

### Use RayExternalDiffusionExecutor when:
- ✅ Ray actors managed externally
- ✅ Worker persistence required
- ✅ Decouple engine and workers
- ✅ Pre-existing Ray infrastructure

## Quick Start Examples

### MultiProc (Default)

```python
engine = DiffusionEngine(
    OmniDiffusionConfig(model="model-path")
)
result = engine.step([request])
engine.close()
```

### RayActor (Recommended)

```python
engine = DiffusionEngine(
    OmniDiffusionConfig(
        model="model-path",
        distributed_executor_backend="ray_actor",
    )
)
result = engine.step([request])
engine.close()
```

### External

```python
# 1. Create custom executor
class MyExecutor(ExternalDiffusionExecutor):
    def _connect_to_workers(self):
        # Your logic
        pass

# 2. Use it
engine = DiffusionEngine(
    OmniDiffusionConfig(
        model="model-path",
        distributed_executor_backend=MyExecutor,
    )
)
```

### RayExternal

```python
# 1. Create Ray actors first
actors = [create_actor(i) for i in range(num_gpus)]

# 2. Configure executor
os.environ["DIFFUSION_WORKER_ACTOR_NAMES"] = ",".join(actor_names)

# 3. Use external executor
engine = DiffusionEngine(
    OmniDiffusionConfig(
        model="model-path",
        distributed_executor_backend=RayExternalDiffusionExecutor,
    )
)
```

## Architecture Comparison

### MultiProc
```
Engine → Executor → [Process-0, Process-1, ...] → Workers
         (spawns)
```

### RayActor
```
Engine → Executor → Ray Actor → Workers
         (creates)  (manages)
```

### External
```
Engine → Executor → Discovers → [External Workers]
         (connects)             (pre-existing)
```

### RayExternal
```
Engine → Executor → Discovers → [Ray Actors] → Workers
         (connects)             (pre-existing)
```

## Migration Guide

### From MultiProc to RayActor

```python
# Before
config = OmniDiffusionConfig(
    model="model-path",
    num_gpus=2,
)

# After
config = OmniDiffusionConfig(
    model="model-path",
    num_gpus=2,
    distributed_executor_backend="ray_actor",  # Add this line
)
```

### From MultiProc to External

```python
# Before: Direct creation
engine = DiffusionEngine(config)

# After: Custom executor
class MyExecutor(ExternalDiffusionExecutor):
    def _connect_to_workers(self):
        # Implement connection logic
        pass

config.distributed_executor_backend = MyExecutor
engine = DiffusionEngine(config)
```

## Performance Comparison

| Executor | Latency | Throughput | Scalability | Overhead |
|----------|---------|------------|-------------|----------|
| MultiProc | Low | High | Single-node | Minimal |
| RayActor | Medium | High | Multi-node | Low-Medium |
| External | Varies | Varies | Maximum | Varies |
| RayExternal | Medium | High | Multi-node | Medium |

## Recommendations

### Development
**Recommended:** `MultiProcDiffusionExecutor` or `RayActorExecutor`
- Simple setup
- Easy debugging
- Fast iteration

### Production (Single Node)
**Recommended:** `RayActorExecutor`
- Ray monitoring
- Better fault handling
- Cleaner code

### Production (Multi Node)
**Recommended:** `RayActorExecutor` or custom `ExternalDiffusionExecutor`
- Distributed execution
- Fault tolerance
- Monitoring

### Custom Deployments
**Recommended:** `ExternalDiffusionExecutor` subclass
- Maximum control
- Custom orchestration
- Advanced scenarios

## Summary Table

| Need | Executor | Config |
|------|----------|--------|
| Simplest setup | MultiProc | `"mp"` (default) |
| Best API | RayActor | `"ray_actor"` |
| Multi-node | RayActor | `"ray_actor"` |
| Custom orchestration | External | Custom class |
| Worker persistence | RayExternal | `RayExternalDiffusionExecutor` |
| Monitoring | RayActor | `"ray_actor"` |
| No dependencies | MultiProc | `"mp"` |

## Code Examples

See detailed examples in:
- [MultiProc Example](../../tests/e2e/offline_inference/)
- [RayActor Example](../../examples/ray_actor_executor_example.py)
- [RayExternal Example](../../examples/ray_external_executor_example.py)

## Documentation

- [Executor Architecture](diffusion_executor_architecture.md)
- [RayActor Guide](ray_actor_executor_guide.md)
- [External Executor Guide](external_executor_guide.md)
