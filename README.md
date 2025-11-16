# ⚙️ PicardForge-Rust
### *A fixed-step ODE & PDE time-integration library featuring Picard–Gauss–Seidel implicit solvers and classical explicit methods — implemented in Rust*

---

## ✨ Overview

**PicardForge-Rust** is the Rust implementation of the PicardForge family — a suite of **fixed-step ODE solvers** specially designed for **semi-discretized PDEs** such as diffusion, heat conduction, parabolic systems, and stiff linear operators.

Like the Python / Julia / C++ versions, this Rust backend implements both **explicit** and **implicit** methods, but leverages Rust’s:

- **memory safety**
- **zero-cost abstractions**
- **no garbage collector**
- **predictable performance**
- **excellent FFI options**

to create a solver library that is **fast, safe, and ideal for embedded PDE solvers or HPC pipelines**.

All implicit solvers use **Picard fixed-point iteration with Gauss–Seidel relaxation**, providing a **Jacobian-free** and **matrix-free** nonlinear solve suitable for diffusion-dominated PDEs.

---

## 🚀 Features

### ✔ Full suite of classic numerical integrators

| Family | Methods | Notes |
|-------|---------|-------|
| **Explicit Runge–Kutta** | RK1–RK6 | Fully hard-coded Butcher tables |
| **Adams–Bashforth** | AB2–AB5 | Explicit multistep |
| **Adams–Moulton** | AM2–AM5 | Implicit multistep with Picard |
| **BDF** | BDF1–BDF6 | Implicit, stiff-accurate |
| **SDIRK** | SDIRK2–SDIRK4 | Diagonally implicit RK |
| **Gauss–Legendre IRK** | s = 1–5 | A-stable, symplectic |
| **Radau IIA IRK** | s = 2–5 | L-stable, stiff solvers |
| **Lobatto IIIC IRK** | s = 2–5 | Symmetric, stiffly accurate |

### ✔ Picard–Gauss–Seidel nonlinear iteration
A unified iterative method for all implicit solvers:

- No Jacobian matrices  
- No Newton iterations  
- Stage-by-stage Gauss–Seidel relaxation  
- Ideal for semi-discretized PDE systems  
- Converges rapidly for diffusion-type operators  

### ✔ Full safety with high performance
Rust guarantees:

- no memory leaks  
- no data races  
- no null pointers  
- no uninitialized buffers  

while still compiling to **native machine code** competitive with C++.

### ✔ Ideal for PDE codebases
The design fits:

- finite-difference semi-discretizations  
- implicit diffusion / heat operators  
- multi-layer conduction models  
- large systems that require safe threading

---

## 📁 Repository Structure

PicardForge-Rust/
│
├── rk.rs # RK1–RK6 explicit solvers
├── ab.rs # AB2–AB5 explicit multistep
├── am.rs # AM2–AM5 implicit multistep (Picard)
├── bdf.rs # BDF1–BDF6 implicit multistep
├── sdirk.rs # SDIRK2–SDIRK4 implicit RK
├── irk.rs # Gauss/Radau/Lobatto IRK (Picard)
