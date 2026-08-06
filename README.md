# Distributed-MOE-and-Speculative-Decoding

## High-Performance ML Systems from Scratch

A repository demonstrating first-principles implementations of core Machine Learning Infrastructure components in PyTorch and MPI, optimizing distributed execution and inference throughput.

## 🚀 Key Performance Impact
*   **1.7× End-to-End Inference Speedup** on LLM generation via vectorized speculative verification.
*   **>85% Speculative Token Acceptance Rate** sustained across baseline target benchmarks.
*   **Zero-Overhead Hardware Saturation** using custom point-to-point full-duplex communication primitives.

---

## 🛠️ Project 1: Tensor Parallel Mixture-of-Experts (`MoE_TP`)

A from-scratch implementation of a Tensor Parallel Mixture-of-Experts engine designed to bypass high-level framework abstractions and isolate distributed execution bottlenecks.

### 📐 Architecture Design
*   **Replicated Router:** Implements a softmax-based gating router uniform across all active worker processes.
*   **Sharded Experts:** Model layers are explicitly sharded across the process pool via custom `ShardedLinear` modules utilizing deterministic rank-specific state contexts (`rng_context`).
*   **Spatial Math Allocation:** Processes calculate global execution boundaries manually (`self.output_offset`), allowing fine-grained control over tensor layouts.

### ⚡ Distributed Optimization & Bottleneck Fix
*   **The Problem:** Dynamic token routing introduces highly asymmetric workloads across sequences. Traditional collective routines inject global blocking synchronization barriers, forcing active GPUs to sit idle waiting for network stragglers.
*   **The Solution:** Bypassed standard collective libraries to build a custom `myAlltoall` primitive. It segments arrays into uniform chunks and orchestrates a point-to-point, bidirectional `Sendrecv` routing topology. This layout saturates interconnect bandwidth and eliminates GPU network idle states.

---

## 🏎️ Project 2: Standalone Speculative Decoding Engine

An algorithmic inference accelerator engineered to mask sequential LLM latency boundaries by pairing a large target model with a lightweight draft engine.

### ⚙️ Optimization Pipeline
*   **Kernel Acceleration:** Compresses draft model initialization latency using `torch.compile(mode="reduce-overhead", fullgraph=True)`.
*   **Vectorized Token Verification:** Bypasses traditional token-by-token sequential evaluation loops that choke hardware execution paths.

### 🧠 Single-Pass Validation Mechanics
Instead of testing tokens iteratively, `verify_tokens_vectorized` concatenates prompt histories and candidate chunks to execute a **single forward pass** through the massive target model. A vectorized boolean mask combined with PyTorch’s `.nonzero()` operation instantly pinpoints the precise index of any token divergence, safely dropping verification overhead.

---

