# SYS-101 — Assessment 3: Federated Averaging (Rust + Candle)

This repository contains a **minimal-but-complete** Federated Learning (FL) system implementing **Federated Averaging (FedAvg)** with a **parameter server** and multiple **federated clients**. The implementation is networked (HTTP/JSON over Axum), concurrent (async + task fan‑out/fan‑in per round), and trains a small **MNIST linear classifier** using the Candle ML stack.

## How it works (design overview)

- **Model**: a single linear layer `784 → 10` with parameters **W** and **b** stored as `Vec<f32>` for easy serialization. At run time, we convert to Candle tensors.
- **Server**:
  - Tracks registered clients and a per‑model state (`status`, `round`, `params`).
  - `/init` initializes the global model (random small weights).
  - `/train` runs `R` rounds. Each round:
    1. Randomly selects `K` clients (IID assumption).
    2. Sends the current global parameters + training config to **each client** concurrently (`tokio::task::JoinSet`).
    3. Receives each client's locally trained parameters and **averages** them (**FedAvg**).
    4. Updates the global model + round counter.
  - `/model/:name` returns status/params/round.
  - `/test` evaluates the current global model on server’s MNIST test split.
- **Client**:
  - Exposes `/train` which **accepts server params** and config, performs **local training** on its own MNIST subset (IID sample controlled via `sample_ratio`) and returns updated params + metrics.
  - `/test` reports local test accuracy; `/model` exposes local params/status.
- **Concurrency & Synchronization**:
  - Shared state on both server and client is guarded by `parking_lot::Mutex` (fast, fair) over small critical sections.
  - Per‑round client calls run concurrently; results are aggregated **after** all tasks finish (fan‑in). No long‑held locks across awaits → **deadlock‑free and race‑free**.
- **Correctness notes**:
  - Each client performs 1+ local epochs with SGD on its sampled data; returns its updated weights.
  - FedAvg improves both local and global performance over rounds (on MNIST this is visible within a few rounds).

## API summary

### Server
- `POST /register` — `{ client_url, model }` → registers the client for that model.
- `POST /init` — `{ model, shape?, seed? }` → resets/initializes the global model.
- `POST /train` — `{ model, rounds, clients_per_round, train }` → runs federated rounds.
- `GET /model/:name` — returns `{ status, params?, round, registered_clients }`.
- `POST /test` — `{ model }` → returns `{ acc }` for the global model.

### Client
- `POST /train` — `{ model, params, train, seed }` → trains locally and returns `{ params, metrics }`.
- `GET /model` — local status/params for debugging.
- `POST /test` — returns `{ acc }` on the client’s test set.

`TrainConfig` fields: `epochs`, `batch_size`, `lr`, `sample_ratio` (fraction of local data per round).

## Build & Run

> Requires Rust toolchain (`cargo`), and the MNIST dataset will be auto-cached by Candle under `~/.cache/mnist` (default). CPU is enough.

### 1) Build
```bash
cargo build --release
```

### 2) Start the server
```bash
ADDR=0.0.0.0:8080 ./target/release/server
```

### 3) Start 2–3 clients (different ports)
```bash
# Client 1
ADDR=0.0.0.0:9001 URL=http://127.0.0.1:9001 SERVER=http://127.0.0.1:8080 ./target/release/client
# Client 2
ADDR=0.0.0.0:9002 URL=http://127.0.0.1:9002 SERVER=http://127.0.0.1:8080 ./target/release/client
# Client 3
ADDR=0.0.0.0:9003 URL=http://127.0.0.1:9003 SERVER=http://127.0.0.1:8080 ./target/release/client
```

> Each client automatically registers itself to the server on boot.

### 4) Initialize the global model
```bash
curl -X POST http://127.0.0.1:8080/init -H "Content-Type: application/json" \
  -d '{"model":"mnist"}'
```

### 5) Run FedAvg for a few rounds
```bash
curl -X POST http://127.0.0.1:8080/train -H "Content-Type: application/json" -d '{
  "model":"mnist",
  "rounds": 3,
  "clients_per_round": 2,
  "train": { "epochs": 1, "batch_size": 64, "lr": 0.1, "sample_ratio": 0.3 }
}'
```

### 6) Inspect status and test accuracy
```bash
curl http://127.0.0.1:8080/model/mnist
curl -X POST http://127.0.0.1:8080/test -H "Content-Type: application/json" -d '{"model":"mnist"}'
```

## Grading criteria mapping

- **Concurrency**: Server fans out per‑client training requests concurrently using async tasks; no shared state is held over `await`, ensuring **deadlock/race‑free** progress.
- **Synchronization**: Minimal critical sections (register/init/update) under `Mutex`. Network interactions are outside the lock. Message‑passing over HTTP embodies **channel‑like** coordination.
- **Correctness**: Clients train on **different IID samples** per round (different seeds) → parameter deltas differ; FedAvg produces a stronger global model over rounds. `POST /test` lets you verify improvements quantitatively.

## Notes & Extensions

- The model abstraction is intentionally simple for clarity; swapping in a deeper network is straightforward.
- You may enable GPU features in Candle by changing features, but CPU is sufficient for this assessment.
- Error handling is pragmatic; productionizing should add retries/backoff for flaky clients.

---

**Author:** Taha Keler — SYS‑101 Spring 2025 — Assessment 3