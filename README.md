![Logo](assets/symdiff-banner.svg)

[![Crates.io](https://img.shields.io/crates/v/symdiff.svg)](https://crates.io/crates/symdiff)
[![Documentation](https://docs.rs/symdiff/badge.svg)](https://docs.rs/symdiff/latest/symdiff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build](https://github.com/amadavan/symdiff-rs/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/amadavan/symdiff-rs/actions/workflows/rust.yml)

Compile-time symbolic differentiation for Rust via a proc-macro attribute.

`#[gradient(dim = N)]` parses a function body at compile time, builds a
symbolic expression tree, differentiates it, runs algebraic simplification,
and emits a companion `{fn}_gradient` function — all before your binary is
compiled, with no runtime cost.

```rust
use symdiff::gradient;

#[gradient(dim = 2)]
fn rosenbrock(x: &[f64]) -> f64 {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
}

fn main() {
    // Gradient at the minimum (1, 1) should be (0, 0).
    let g = rosenbrock_gradient(&[1.0, 1.0]);
    assert!(g[0].abs() < 1e-10);
    assert!(g[1].abs() < 1e-10);
}
```

The generated `rosenbrock_gradient` is a plain Rust function containing just
the closed-form derivative — no allocations, no trait objects, no overhead.

## Usage

```toml
[dependencies]
symdiff = "2.0.1"
```

`#[gradient(...)]` accepts two parameters:

| Parameter    | Type    | Required | Description                                                              |
|--------------|---------|:--------:|--------------------------------------------------------------------------|
| `dim`        | `usize` | yes      | Number of partial derivatives (length of output array)                   |
| `max_passes` | `usize` | no       | Max simplification passes; default 10                                    |
| `sparse`     | `bool`  | no       | Output the gradient as sparse vector                                     |
| `prune`      | `bool`  | no       | Whether to prune the tree after derivative (expensive but lower memory)  |

The annotated function must have signature `fn name(x: &[f64]) -> f64`; the
generated gradient has signature `fn name_gradient(x: &[f64]) -> [f64; dim]` for dense output and
`fn name_gradient(x: &[f64]) => ([f64; reduced_dim], [f64; reduced_dim])`, where the first
corresponds to the indices, and the second corresponds to the values.

The function body may contain `let` bindings followed by a bare tail expression
(no trailing semicolon). Bound names are substituted symbolically before differentiating.

## Supported syntax

| Source form                           | How it is treated                   |
|---------------------------------------|-------------------------------------|
| `x[i]`                                | Variable - `i`-th component of `x`  |
| `1.0`, `2`                            | Constant                            |
| `e + f`, `e - f`, `e * f`, `e / f`    | Binary arithmetic                   |
| `-e`                                  | Negation                            |
| `e.sin()`, `e.cos()`                  | Trigonometric functions             |
| `e.ln()`, `e.exp()`, `e.sqrt()`       | Transcendental functions            |
| `e.powi(n)` with integer constant `n` | Integer power                       |
| `(e)`, `e as f64`, `{ e }`            | Transparent - inner expression used |
| Anything else                         | **Compile error (panic)**           |

## Simplification

After differentiating, simplification runs repeatedly until the expression
stops changing or `max_passes` is reached (default: 10). Each pass applies
constant folding, identity rules (`0 + e = e`, `1 * e = e`, etc.), double
negation, distributive factoring, power and exponential merging, and a handful
of logarithm and square-root identities. Multiple passes let reductions
cascade — for example, `0 * f(x) + 1` needs two passes to reach `1`.

## Limitations

This is an early-stage library with a narrow scope:

- Only `f64` arithmetic. The input must be `x: &[f64]`; scalar parameters
  are not differentiated.
- Only `powi` for powers. `powf` and anything not in the table above causes a
  compile-time panic.
- `powi` exponents must be integer literals, not variables.
- The input slice parameter must be named `x`.
- No support for higher-order derivatives.

## Alternatives

**[rust-ad](https://github.com/JonathanWoollett-Light/rust-ad)** takes the same
proc-macro approach but implements algorithmic AD (forward/reverse mode) rather
than producing a symbolic closed form.

**[descent](https://crates.io/crates/descent)** also generates symbolic
derivatives at compile time via proc-macros ("fixed" form), and additionally
offers a runtime expression tree ("dynamic") form. Both are scoped to the Ipopt
solver and require nightly Rust.

**`#[autodiff]` (Enzyme)** differentiates at the LLVM IR level, which means it
handles arbitrary Rust code but produces no simplified closed form and requires
nightly.

**[symbolica](https://crates.io/crates/symbolica)** and similar runtime CAS
crates do the same symbolic work as `symdiff` — but at runtime, on expression
objects, rather than emitting native Rust at compile time.

>*AI Disclosure* The documentation was initially generated by AI, but edited independently. AI-assistance was used to help design the overall structure and for boilerplate, but all code is our own. Copilot was used to generate a CI run for dev.
