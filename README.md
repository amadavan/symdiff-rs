![Logo](assets/symdiff-banner.svg)


[![Crates.io](https://img.shields.io/crates/v/symdiff.svg)](https://crates.io/crates/symdiff)
[![Documentation](https://docs.rs/symdiff/badge.svg)](https://docs.rs/symdiff/latest/symdiff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build](https://github.com/amadavan/symdiff-rs/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/amadavan/symdiff-rs/actions/workflows/rust.yml)

Compile-time symbolic differentiation for Rust via a proc-macro attribute.

Annotate a function with `#[gradient]` and the macro parses its body at compile
time, builds a symbolic expression tree, applies analytical differentiation
rules, simplifies the result, and emits a companion `{fn}_gradient` function
— all with zero runtime overhead.

```rust
use symdiff::gradient;

#[gradient(arg = "x", dim = 2)]
fn rosenbrock(x: &[f64]) -> f64 {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
}

fn main() {
    // Gradient at the minimum (1, 1) is (0, 0).
    let g = rosenbrock_gradient(&[1.0, 1.0]);
    assert!(g[0].abs() < 1e-10);
    assert!(g[1].abs() < 1e-10);
}
```

The generated `rosenbrock_gradient` is a plain Rust function — no closures, no
allocations, no trait objects — just the closed-form derivative expression
inlined directly.

---

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
symdiff = "1.0.0"
```

### Attribute parameters

| Parameter    | Type             | Required | Description                                              |
|--------------|------------------|----------|----------------------------------------------------------|
| `arg`        | `&str`           | yes      | Name of the `&[f64]` parameter to differentiate against  |
| `dim`        | `usize`          | yes      | Number of gradient components (length of output array)   |
| `max_passes` | `usize`          | no       | Maximum simplification passes; default 10                |

### Requirements

- The annotated function must accept its differentiable argument as `&[f64]`
  and return `f64`.
- The function body must be a single expression, optionally preceded by `let`
  bindings.  `let`-bound names are inlined symbolically into subsequent
  expressions.

---

## Supported syntax

| Source form                                  | How it is treated                          |
|----------------------------------------------|--------------------------------------------|
| `x[i]`                                       | Variable — `i`-th component of `arg`       |
| `1.0`, `2`                                   | Constant                                   |
| `e + f`, `e - f`, `e * f`, `e / f`          | Binary arithmetic                          |
| `-e`                                         | Negation                                   |
| `e.sin()`, `e.cos()`                         | Trigonometric functions                    |
| `e.ln()`, `e.exp()`, `e.sqrt()`              | Transcendental functions                   |
| `e.powi(n)` with integer constant `n`        | Integer power                              |
| `let name = expr;`                           | Inlined into subsequent expressions        |
| `(e)`, `e as f64`                            | Transparent — inner expression used        |
| Anything else                                | **Opaque** — derivative assumed zero       |

Opaque sub-expressions are safe when they do not depend on the differentiation
variable (e.g. a call to an external helper).  If an opaque node *does* depend
on the variable, the generated derivative will silently be incorrect.

---

## Simplification

After differentiating, `symdiff` runs multiple passes of algebraic
simplification until a fixed point is reached (or `max_passes` is exhausted).
Rules applied include:

- Constant folding (`3.0 * 2.0` → `6.0`)
- Additive / multiplicative identity removal (`0 + e` → `e`, `1 * e` → `e`)
- Double negation (`--e` → `e`)
- Factor extraction (`a*b + a*c` → `a*(b+c)`)
- Power merging (`x^a * x^b` → `x^(a+b)`, `x^a / x^b` → `x^(a-b)`)
- Exponential merging (`exp(a) * exp(b)` → `exp(a+b)`)
- Logarithm of a power (`ln(x^n)` → `n * ln(x)`)
- Square-root of an even power (`sqrt(x^(2k))` → `x^k`)

---

## Alternatives

### `rust-ad`

[rust-ad](https://github.com/JonathanWoollett-Light/rust-ad) uses the same
architectural approach — a proc-macro that walks a Rust function body AST via
`syn` and emits a transformed function.  The key difference is that `rust-ad`
implements **algorithmic** (forward/reverse mode) AD: it mechanically propagates
derivative values through each operation at runtime.  It does not produce a
simplified closed-form expression.  `symdiff` produces a fully symbolic,
simplified derivative that the compiler can reason about and optimise further.

### `descent_macro`

[descent](https://crates.io/crates/descent) provides an `expr!{}` proc-macro
that emits symbolic derivatives at compile time, similar in spirit to `symdiff`.
The main differences: it operates on a **custom DSL** inside the macro invocation
(not an ordinary annotated `fn`), requires nightly Rust, and is scoped to a
specific nonlinear optimisation solver (Ipopt).  `symdiff` works on standard
stable Rust functions with no special toolchain requirements.

### `#[autodiff]` (rustc + Enzyme)

The nightly `std::autodiff` attribute differentiates functions at the **LLVM IR
level** using [Enzyme](https://enzyme.mit.edu/).  It is not symbolic — no
simplified closed-form is produced, and compile times are expensive because
Enzyme must recover type information from IR.  It supports forward and reverse
mode AD and handles arbitrary Rust code.  `symdiff` is more limited in the
syntax it recognises but produces human-readable, zero-overhead derivative
expressions on stable Rust.

### Runtime CAS (`symbolica`, `symb_anafis`)

[symbolica](https://crates.io/crates/symbolica) and
[symb_anafis](https://crates.io/crates/symb_anafis) are full computer-algebra
systems that perform symbolic differentiation at **runtime** on expression trees.
They support a much richer set of operations than `symdiff`.  The trade-off is
that they require runtime expression construction and evaluation rather than
emitting native Rust code at compile time.

### Summary

| Crate / approach           | Compile-time? | Symbolic / simplified?   | Plain `fn` syntax? | Stable Rust?   |
|----------------------------|:-------------:|:-----------------------:|:------------------:|:--------------:|
| **symdiff** (this crate)   | ✓             | ✓                       | ✓                  | ✓              |
| `rust-ad`                  | ✓             | ✗ (algorithmic AD)      | ✓                  | ✓              |
| `descent_macro`            | ✓             | ✓ (limited DSL)         | ✗                  | ✗ (nightly)    |
| `#[autodiff]` (Enzyme)     | ✓             | ✗ (IR-level AD)         | ✓                  | ✗ (nightly)    |
| `symbolica`                | ✗             | ✓                       | ✗                  | ✓              |
| `symb_anafis`              | ✗             | ✓                       | ✗                  | ✓              |

---

## Limitations

- Only `f64` arithmetic is supported.
- The differentiable argument must be a `&[f64]` slice; scalar `f64` parameters
  are not yet recognised as variables.
- Only `powi` (integer powers) is supported; `powf` and general `pow` are
  treated as opaque.
- Expressions that cannot be parsed symbolically (external function calls,
  closures, control flow) are treated as constants with derivative zero.
- No support for Hessians or higher-order derivatives yet.

---

> *AI Disclosure* — the structure of the library was inspired by AI-tooling, and documentation has been largely AI-generated.
