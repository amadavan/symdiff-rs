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
//! #[gradient(arg = "x", dim = 2)]
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
//! | Parameter | Type | Description |
//! |---|---|---|
//! | `arg` | `&str` | Name of the slice parameter to differentiate with respect to |
//! | `dim` | `usize` | Number of components (length of the gradient vector) |
//! | `max_passes` | `usize` (optional) | Maximum simplification passes; defaults to 10 |
//!
//! # Supported syntax
//!
//! | Source form | Symbolic node |
//! |---|---|
//! | `x[i]` | `Var` — the `i`-th component of `arg` |
//! | `1.0`, `2` | `Const` |
//! | `e + f`, `e - f`, `e * f`, `e / f` | `Add`, `Sub`, `Mul`, `Div` |
//! | `-e` | `Neg` |
//! | `e.sin()`, `e.cos()` | `Sin`, `Cos` |
//! | `e.ln()`, `e.exp()`, `e.sqrt()` | `Ln`, `Exp`, `Sqrt` |
//! | `e.powi(n)` (const `n`) | `Powi` |
//! | `let name = expr;` | inlined into subsequent expressions |
//! | `(e)`, `e as f64` | transparent — inner expression used |
//! | anything else | `Opaque` — derivative assumed zero |
//!
//! > *AI Disclosure* The structure and partial implementation of this file was generated with AI-tooling.
mod expr;

use expr::*;
use proc_macro2::TokenStream;
use quote::quote;
use std::collections::HashMap;
use syn::{Pat, Stmt};

/// Walk a function body block and return the symbolic form of its return value.
///
/// The body must consist of zero or more `let` bindings followed by a
/// bare expression (no trailing semicolon).  Each `let` binding is converted
/// to a [`SymExpr`] and stored in a name → expression map so that subsequent
/// uses of the bound name are inlined symbolically.
///
/// Returns `None` if no bare tail expression is found (e.g. the body ends with
/// a semicolon-terminated statement or is empty).
fn parse_body(block: &syn::Block) -> Option<SymExpr> {
    let mut bindings = HashMap::new();

    for stmt in &block.stmts {
        match stmt {
            Stmt::Local(local) => {
                if let Pat::Ident(pat_ident) = &local.pat {
                    let name = pat_ident.ident.to_string();
                    if let Some(init) = &local.init {
                        let sym = syn_to_sym(&init.expr, &bindings);
                        bindings.insert(name, sym);
                    }
                }
            }
            Stmt::Expr(expr, None) => {
                return Some(syn_to_sym(expr, &bindings));
            }
            _ => {
                // Unsupported statement type (e.g. semi-colon terminated expr, item, macro).
                // For simplicity, we require the function body to be a single expression
                // with optional `let` bindings, so we can skip these.
            }
        }
    }

    None
}

/// Parsed attribute arguments for [`gradient`].
#[derive(deluxe::ParseMetaItem)]
struct DerivativeInput {
    /// Name of the slice parameter to differentiate with respect to (e.g. `"x"`).
    arg: String,
    /// Number of gradient components, i.e. the length of the output array.
    dim: usize,
    /// Maximum number of simplification passes.  `None` uses the default (10).
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
    let params = &input_fn.sig.inputs;
    let body = &input_fn.block;
    let vis = &input_fn.vis;

    let DerivativeInput {
        arg,
        dim,
        max_passes,
    } = deluxe::parse::<DerivativeInput>(attr.into())
        .expect("Failed to parse macro attribute arguments for gradient.");

    let sym = match parse_body(body) {
        Some(s) => s,
        None => {
            return syn::Error::new_spanned(
                body,
                "Function body must be a single expression for symbolic differentiation",
            )
            .to_compile_error()
            .into();
        }
    };

    let grad_components: Vec<TokenStream> = (0..dim)
        .map(|i| {
            let d = sym
                .diff(format!("{}[{}]", arg, i))
                .simplify_multipass(max_passes);
            d.into_token_stream()
        })
        .collect();

    let grad_name = syn::Ident::new(
        &format!("{}_gradient", fn_name),
        proc_macro2::Span::call_site(),
    );

    let expanded = quote!(
        #input_fn

        #vis fn #grad_name(#params) -> [f64; #dim] {
            [#(#grad_components),*]
        }
    );

    expanded.into()
}
