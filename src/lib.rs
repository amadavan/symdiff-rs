//! Compile-time symbolic automatic differentiation.
//!
//! This crate provides the [`gradient`] proc-macro attribute, which parses a
//! Rust function body at compile time, builds a symbolic expression tree,
//! differentiates it analytically, simplifies the result, and emits a
//! companion `{fn}_gradient` function containing the closed-form derivative.
//!
//! # Usage
//!
//! Annotate a function that takes a slice `&[f64]` and returns `f64`:
//!
//! ```rust,ignore
//! use symdiff::gradient;
//!
//! #[gradient(dim = 2)]
//! fn rosenbrock(x: &[f64]) -> f64 {
//!     (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
//! }
//!
//! // The macro emits `rosenbrock` unchanged plus:
//! // fn rosenbrock_gradient(x: &[f64]) -> [f64; 2] { [...] }
//! ```
//!
//! ## Attribute parameters
//!
//! | Parameter    | Type    | Description                                      |
//! |--------------|---------|--------------------------------------------------|
//! | `dim`        | `usize` | Number of components (length of the gradient)    |
//! | `max_passes` | `usize` | Max simplification passes; optional, default 10  |
//!
//! # Supported syntax
//!
//! | Source form                             | Symbolic node       |
//! |-----------------------------------------|---------------------|
//! | `x[i]`                                  | `Var`               |
//! | `1.0`, `2`                              | `Const`             |
//! | `e + f`, `e - f`, `e * f`, `e / f`     | `Add/Sub/Mul/Div`   |
//! | `-e`                                    | `Neg`               |
//! | `e.sin()`, `e.cos()`                    | `Sin`, `Cos`        |
//! | `e.ln()`, `e.exp()`, `e.sqrt()`         | `Ln`, `Exp`, `Sqrt` |
//! | `e.powi(n)` (integer literal `n`)       | `Powi`              |
//! | `(e)`, `e as f64`                       | transparent         |
//! | anything else                           | compile-time panic  |
mod expr;

use expr::*;
use quote::quote;
use syn::{Expr, Pat, Stmt};

/// Walk a function body block and return the symbolic form of its return value.
///
/// The body must consist of zero or more `let` bindings followed by a
/// bare expression (no trailing semicolon).  Each `let` binding is converted
/// to a [`SymExpr`] and stored in a name → expression map so that subsequent
/// uses of the bound name are inlined symbolically.
///
/// Returns `None` if no bare tail expression is found (e.g. the body ends with
/// a semicolon-terminated statement or is empty).
fn parse_body(block: &syn::Block) -> Option<Expr> {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Local(local) => {
                if let Pat::Ident(pat_ident) = &local.pat {
                    let _name = pat_ident.ident.to_string();
                    if let Some(init) = &local.init {
                        return Some(init.expr.as_ref().clone());
                    }
                }
            }
            Stmt::Expr(expr, None) => return Some(expr.clone()),
            _ => {
                // Unsupported statement type (e.g. semi-colon terminated expr, item, macro).
                // For simplicity, we require the function body to be a single expression.
            }
        }
    }

    None
}

/// Parsed attribute arguments for [`gradient`].
#[derive(deluxe::ParseMetaItem)]
struct DerivativeInput {
    /// Number of gradient components, i.e. the length of the output array.
    dim: usize,
    /// Maximum number of simplification passes to perform; defaults to 10 if not specified.
    max_passes: Option<usize>,
}

/// Derive a gradient function for the annotated `fn` at compile time.
///
/// Emits the original function unchanged, plus a companion
/// `{fn_name}_gradient` with the same signature but returning `[f64; dim]`
/// whose `i`-th element is `∂f/∂arg[i]`.
///
/// See the [crate-level documentation](self) for the full attribute syntax and
/// the list of supported source expressions.
#[proc_macro_attribute]
pub fn gradient(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let input_fn = syn::parse_macro_input!(item as syn::ItemFn);
    let fn_name = &input_fn.sig.ident;
    let _params = &input_fn.sig.inputs;
    let body = &input_fn.block;
    let vis = &input_fn.vis;

    let DerivativeInput { dim, max_passes } = deluxe::parse::<DerivativeInput>(attr.into())
        .expect("Failed to parse macro attribute arguments for gradient.");

    let func_def = parse_body(body);
    let mut arena = SymArena::new();

    if func_def.is_none() {
        panic!("Function body must end with a bare expression (no trailing semicolon).");
    }

    let root = parse_syn(&mut arena, &func_def.unwrap());

    let gradient_tokens = (0..dim)
        .map(|i| compile_expression(&mut arena, root, i, max_passes.unwrap_or(10)))
        .collect::<Vec<_>>();

    let grad_name = syn::Ident::new(
        &format!("{}_gradient", fn_name),
        proc_macro2::Span::call_site(),
    );

    let expanded = quote!(
        #input_fn

        #vis fn #grad_name(x: &[f64]) -> [f64; #dim] {
            [#(#gradient_tokens),*]
        }
    );

    expanded.into()
}
