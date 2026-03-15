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
mod arena;
mod coordinator;
mod transformers;
mod visitors;

use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Expr, Pat, Stmt};

use arena::{NodeId, SymArena, SymNode};
use transformers::{DiffTransformer, SimplifyTransformer};
use visitors::{RefCountVisitor, ToTokenStreamVisitor};

use crate::coordinator::{Coordinator, GreedyCoordinator};

/// Parse a `syn` expression into the [`SymArena`], returning the root [`NodeId`].
///
/// Supported expression forms:
///
/// | Source                   | Result                        |
/// |--------------------------|-------------------------------|
/// | `1`, `2.0`               | `Const` (f64 bit pattern)     |
/// | `x[i]`                   | `Var(i)`                      |
/// | `a + b`, `a - b`         | `Add`, `Sub`                  |
/// | `a * b`, `a / b`         | `Mul`, `Div`                  |
/// | `-e`                     | `Neg`                         |
/// | `e.powi(n)`              | `Powi` (n must be an integer literal) |
/// | `e.sin()`, `e.cos()`     | `Sin`, `Cos`                  |
/// | `e.ln()`, `e.exp()`      | `Ln`, `Exp`                   |
/// | `e.sqrt()`               | `Sqrt`                        |
/// | `(e)`, `e as T`, `{e}`   | transparent — inner expr used |
///
/// # Panics
///
/// Panics on any unsupported expression form.
fn parse_syn(arena: &mut SymArena, expr: &Expr, env: &HashMap<syn::Ident, NodeId>) -> NodeId {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Int(int_lit) => {
                let value = int_lit.base10_parse::<u64>().unwrap();
                return arena.intern(SymNode::Const((value as f64).to_bits()));
            }
            syn::Lit::Float(flt_lit) => {
                let value = flt_lit.base10_parse::<f64>().unwrap();
                return arena.intern(SymNode::Const(value.to_bits()));
            }
            _ => panic!("Unsupported literal type"),
        },

        Expr::Index(index_expr) => {
            if let Expr::Path(path_expr) = &*index_expr.expr {
                if path_expr.path.is_ident("x") {
                    if let Expr::Lit(lit) = &*index_expr.index {
                        if let syn::Lit::Int(int_lit) = &lit.lit {
                            let idx = int_lit.base10_parse::<NodeId>().unwrap();
                            return arena.intern(SymNode::Var(idx));
                        } else {
                            panic!("Expected integer literal for variable index");
                        }
                    } else {
                        panic!("Expected literal for variable index");
                    }
                } else {
                    panic!("Expected variable name 'x'");
                }
            } else {
                panic!("Expected variable access of the form x[i]");
            }
        }

        Expr::Path(path_expr) => {
            if let Some(ident) = path_expr.path.get_ident() {
                if let Some(node_id) = env.get(ident) {
                    return *node_id;
                } else {
                    panic!("Undefined variable: {}", ident);
                }
            } else {
                panic!("Unsupported path expression");
            }
        }

        Expr::Binary(bin_expr) => {
            let left_id = parse_syn(arena, &bin_expr.left, env);
            let right_id = parse_syn(arena, &bin_expr.right, env);
            return match bin_expr.op {
                syn::BinOp::Add(_) => arena.intern(SymNode::Add(left_id, right_id)),
                syn::BinOp::Sub(_) => arena.intern(SymNode::Sub(left_id, right_id)),
                syn::BinOp::Mul(_) => arena.intern(SymNode::Mul(left_id, right_id)),
                syn::BinOp::Div(_) => arena.intern(SymNode::Div(left_id, right_id)),
                _ => panic!("Unsupported binary operator"),
            };
        }

        Expr::Unary(un) => {
            let operand_id = parse_syn(arena, &un.expr, env);
            return match un.op {
                syn::UnOp::Neg(_) => arena.intern(SymNode::Neg(operand_id)),
                _ => panic!("Unsupported unary operator"),
            };
        }

        Expr::MethodCall(call) => {
            let receiver_id = parse_syn(arena, &call.receiver, env);
            let method_name = call.method.to_string();
            return match method_name.as_str() {
                "powi" => {
                    if call.args.len() != 1 {
                        panic!("powi method call must have exactly one argument");
                    }
                    if let Expr::Lit(lit) = &call.args[0] {
                        if let syn::Lit::Int(int_lit) = &lit.lit {
                            let exp = int_lit.base10_parse::<i32>().unwrap();
                            arena.intern(SymNode::Powi(receiver_id, exp))
                        } else {
                            panic!("Expected integer literal for powi exponent");
                        }
                    } else {
                        panic!("Expected literal for powi exponent");
                    }
                }
                "sin" => arena.intern(SymNode::Sin(receiver_id)),
                "cos" => arena.intern(SymNode::Cos(receiver_id)),
                "ln" => arena.intern(SymNode::Ln(receiver_id)),
                "exp" => arena.intern(SymNode::Exp(receiver_id)),
                "sqrt" => arena.intern(SymNode::Sqrt(receiver_id)),
                _ => panic!("Unsupported method call: {}", method_name),
            };
        }

        Expr::Paren(paren) => parse_syn(arena, &paren.expr, env),
        Expr::Group(group) => parse_syn(arena, &group.expr, env),
        Expr::Cast(c) => parse_syn(arena, &c.expr, env),

        _ => panic!("Unsupported expression type {:?}", expr),
    }
}

/// Differentiate `root_id` symbolically with respect to `var_idx`, simplify
/// the result, and emit it as a `TokenStream`.
///
/// The emitted tokens represent a single `f64` expression (no surrounding
/// function definition).  Common sub-expressions that appear more than once are
/// hoisted into `let` bindings by [`ToTokenStreamVisitor`].
fn compile_expression(
    arena: &mut SymArena,
    root_id: NodeId,
    var_idx: usize,
    max_passes: usize,
) -> (Vec<TokenStream>, TokenStream) {
    let cost_estimates = HashMap::from([
        (SymNode::Const(0), 0),
        (SymNode::Var(0), 0),
        (SymNode::Add(0, 0), 1),
        (SymNode::Sub(0, 0), 1),
        (SymNode::Mul(0, 0), 1),
        (SymNode::Div(0, 0), 15),
        (SymNode::Powi(0, 0), 3),
        (SymNode::Neg(0), 0),
        (SymNode::Sin(0), 100),
        (SymNode::Cos(0), 100),
        (SymNode::Ln(0), 60),
        (SymNode::Exp(0), 60),
        (SymNode::Sqrt(0), 5),
    ]);

    // Differentiate with respect to variable var_idx
    let diff_transformer = DiffTransformer::new(var_idx);
    let mut root_id = arena.transform(root_id, &diff_transformer);

    // Simplify the result
    let simplify_transformer = SimplifyTransformer::new();
    for _ in 0..max_passes {
        let new_root_id = arena.transform(root_id, &simplify_transformer);
        if new_root_id == root_id {
            break; // No further simplification possible
        }
        root_id = new_root_id;
    }

    // Commutative and associative reordering to canonicalize expressions and expose more common sub-expressions.
    let greedy_coordinator = GreedyCoordinator::new(&cost_estimates);
    for _ in 0..max_passes {
        let new_root_id = greedy_coordinator.optimize(root_id, arena);
        if new_root_id == root_id {
            break; // No further optimization possible
        }
        root_id = new_root_id;
    }

    // Reference counting for common sub-expression elimination
    let mut ref_count_visitor = RefCountVisitor::new();
    arena.accept(root_id, &mut ref_count_visitor);
    let ref_counts = ref_count_visitor.get_counts();

    let mut to_tokens_visitor = ToTokenStreamVisitor::new(&ref_counts);
    let gradient_tokens = arena.accept(root_id, &mut to_tokens_visitor);
    let instruction_tokens = to_tokens_visitor.get_instructions().to_vec();

    // Pass instructions and the final expression tokens back to the caller so they can be emitted together.
    (instruction_tokens, gradient_tokens)
}

/// Walk a function body block and return the symbolic form of its return value.
///
/// The body must consist of zero or more `let` bindings followed by a
/// bare expression (no trailing semicolon).  Each `let` binding is converted
/// to a [`SymExpr`] and stored in a name → expression map so that subsequent
/// uses of the bound name are inlined symbolically.
///
/// Returns `None` if no bare tail expression is found (e.g. the body ends with
/// a semicolon-terminated statement or is empty.
fn parse_body(
    arena: &mut SymArena,
    block: &syn::Block,
) -> (HashMap<syn::Ident, NodeId>, Option<Expr>) {
    let mut env = HashMap::new();

    for stmt in &block.stmts {
        match stmt {
            Stmt::Local(local) => {
                let ident = match &local.pat {
                    Pat::Ident(pat_ident) => Some(&pat_ident.ident),
                    Pat::Type(pat_ident) => {
                        if let Pat::Ident(inner_ident) = &*pat_ident.pat {
                            Some(&inner_ident.ident)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if let (Some(name), Some(init)) = (ident, &local.init) {
                    let expr = &init.expr;
                    let node_id = parse_syn(arena, expr, &env);
                    env.insert(name.clone(), node_id);
                }
            }
            Stmt::Expr(expr, None) => return (env, Some(expr.clone())),
            _ => {
                // Unsupported statement type (e.g. semi-colon terminated expr, item, macro).
                // For simplicity, we require the function body to be a single expression.
            }
        }
    }

    (env, None)
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

    let mut arena = SymArena::new();

    let (env, func_def) = parse_body(&mut arena, body);

    if func_def.is_none() {
        panic!("Function body must end with a bare expression (no trailing semicolon).");
    }

    let root = parse_syn(&mut arena, &func_def.unwrap(), &env);

    let tokens = (0..dim)
        .map(|i| compile_expression(&mut arena, root, i, max_passes.unwrap_or(10)))
        .collect::<Vec<_>>();

    let gradient_tokens = tokens
        .iter()
        .map(|(_, expr)| {
            quote! {
                #expr
            }
        })
        .collect::<Vec<_>>();

    let instruction_tokens = tokens
        .iter()
        .flat_map(|(instrs, _)| instrs)
        .collect::<Vec<_>>();

    let grad_name = syn::Ident::new(
        &format!("{}_gradient", fn_name),
        proc_macro2::Span::call_site(),
    );

    let expanded = quote!(
        #input_fn

        #vis fn #grad_name(x: &[f64]) -> [f64; #dim] {
            #(#instruction_tokens)*
            [#(#gradient_tokens),*]
        }
    );

    expanded.into()
}
