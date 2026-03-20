//! Compile-time symbolic differentiation via a proc-macro attribute.
//!
//! The [`gradient`] macro differentiates a Rust function analytically at
//! compile time and emits a companion `{fn}_gradient` that returns the
//! closed-form gradient as a fixed-size array. No numerical approximation,
//! no runtime overhead beyond the arithmetic itself.
//!
//! # How it works
//!
//! At compile time the macro:
//!
//! 1. Parses the function body into a symbolic expression tree.
//! 2. Differentiates each component analytically using standard calculus rules.
//! 3. Simplifies the result (constant folding, identity laws, CSE, etc.).
//! 4. Applies greedy commutative/associative reordering to expose further
//!    common sub-expressions.
//! 5. Emits the gradient function with shared sub-expressions hoisted into
//!    `let` bindings.
//!
//! # Example
//!
//! ```rust,ignore
//! use symdiff::gradient;
//!
//! #[gradient(dim = 2)]
//! fn rosenbrock(x: &[f64]) -> f64 {
//!     let a = 1.0 - x[0];
//!     let b = x[1] - x[0].powi(2);
//!     a.powi(2) + 100.0 * b.powi(2)
//! }
//!
//! // Generates rosenbrock unchanged, plus:
//! // fn rosenbrock_gradient(x: &[f64]) -> [f64; 2] {
//! //     let tmp0 = x[0].powi(1);   // shared sub-expressions hoisted
//! //     ...
//! //     [∂f/∂x[0], ∂f/∂x[1]]
//! // }
//!
//! let g = rosenbrock_gradient(&[1.0, 1.0]);
//! assert_eq!(g, [0.0, 0.0]); // minimum of the Rosenbrock function
//! ```
//!
//! # Attribute parameters
//!
//! - `dim: usize` — number of gradient components; must match the number of
//!   `x[i]` indices used in the function body (required)
//! - `max_passes: usize` — maximum simplification passes; default 10 (optional)
//!
//! # Supported syntax
//!
//! The function body may contain `let` bindings followed by a bare tail
//! expression. Within expressions, the following are supported:
//!
//! - Variables: `x[0]`, `x[1]`, … (integer literal indices only)
//! - Numeric literals: `1`, `2.0`, etc.
//! - Arithmetic: `+`, `-`, `*`, `/`, unary `-`
//! - Methods: `powi(n)` (integer literal exponent), `sin`, `cos`, `ln`,
//!   `exp`, `sqrt`
//! - Transparent: parentheses, `as f64` casts, block expressions
//!
//! Anything else (closures, function calls, loops, …) causes a compile-time
//! panic with a descriptive message.
//!
//! # Limitations
//!
//! - The input slice parameter must be named `x`.
//! - `powi` exponents must be integer literals, not variables.
//! - Conditional expressions and loops cannot be differentiated symbolically.

mod arena;
mod coordinator;
mod transformers;
mod visitors;

use std::collections::HashMap;

use quote::quote;
use syn::{Expr, Ident, Pat, Stmt};

use arena::{NodeId, SymArena, SymNode, VarId};
use transformers::{DiffTransformer, SimplifyTransformer};
use visitors::{RefCountVisitor, ToTokenStreamVisitor};

use crate::coordinator::{Coordinator, GreedyCoordinator};

/// Parse a `syn` expression into the [`SymArena`], returning the root [`NodeId`].
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
                if let Some(var_ident) = path_expr.path.get_ident() {
                    if let Some(var_id) = arena.get_var_id(var_ident) {
                        if let Expr::Lit(lit) = &*index_expr.index {
                            if let syn::Lit::Int(int_lit) = &lit.lit {
                                let idx = int_lit.base10_parse::<NodeId>().unwrap();
                                return arena.intern(SymNode::Var(var_id, idx));
                            } else {
                                panic!("Expected integer literal for variable index");
                            }
                        } else {
                            panic!("Expected literal for variable index");
                        }
                    } else {
                        panic!("Unexpected variable identifier");
                    }
                } else {
                    panic!("Unable to find variable identifier");
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

/// Differentiate `root_id` with respect to `var_idx`, simplify the result, and
/// emit the `root_id` of the simplified expression.
fn compile_expression(
    arena: &mut SymArena,
    root_id: NodeId,
    var_id: VarId,
    var_idx: usize,
    options: &Options,
) -> NodeId {
    let cost_estimates = HashMap::from([
        (SymNode::Const(0), 0),
        (SymNode::Var(0, 0), 0),
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
    let diff_transformer = DiffTransformer::new(var_id, var_idx);
    let mut root_id = arena.transform(root_id, &diff_transformer);

    // Simplify the result
    let simplify_transformer = SimplifyTransformer::new();
    for _ in 0..options.max_passes {
        let new_root_id = arena.transform(root_id, &simplify_transformer);
        if new_root_id == root_id {
            break; // No further simplification possible
        }
        root_id = new_root_id;
    }

    // Commutative and associative reordering to canonicalize expressions and expose more common sub-expressions.
    let greedy_coordinator = GreedyCoordinator::new(&cost_estimates);
    for _ in 0..options.max_passes {
        let new_root_id = greedy_coordinator.optimize(root_id, arena);
        if new_root_id == root_id {
            break; // No further optimization possible
        }
        root_id = new_root_id;
    }

    root_id
}

/// Walk a function body and return its environment and tail expression.
///
/// Expects zero or more `let` bindings followed by a bare return expression.
/// Bound names are stored so later uses are inlined symbolically. Returns
/// `None` for the tail if the body ends with a semicolon or is empty.
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
struct DerivativeOptions {
    // Variable over which to compute the gradient
    var: Ident,
    /// Number of gradient components, i.e. the length of the output array.
    dim: usize,
    /// Maximum simplification passes; defaults to 10.
    max_passes: Option<usize>,
    /// Specify whether to output data as sparse structures
    sparse: Option<bool>,
    /// Whether the input array access should be unchecked
    unchecked: Option<bool>,
    /// Specify whether to prune the tree after simplification
    /// This operation may be expensive so diabled by default
    prune: Option<bool>,
}

struct Options {
    max_passes: usize,
    sparse: bool,
    unchecked: bool,
    prune: bool,
}

impl From<DerivativeOptions> for Options {
    fn from(value: DerivativeOptions) -> Self {
        Self {
            max_passes: value.max_passes.unwrap_or(10),
            sparse: value.sparse.unwrap_or(false),
            unchecked: value.unchecked.unwrap_or(false),
            prune: value.prune.unwrap_or(false),
        }
    }
}

/// Emit the original function unchanged, plus `{fn}_gradient(x: &[f64]) -> [f64; dim]`
/// where element `i` is `∂f/∂x[i]` in closed form.
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

    let derivative_options = deluxe::parse::<DerivativeOptions>(attr.into())
        .expect("Failed to parse macro attribute arguments for gradient.");
    let dim = derivative_options.dim;
    let var = derivative_options.var.clone();
    let options = Options::from(derivative_options);

    let mut arena = SymArena::new();

    // Add all parameters as variables
    let param_idents = params.iter().filter_map(|param| {
        if let syn::FnArg::Typed(pat_type) = param {
            if let syn::Pat::Ident(pat_ident) = *pat_type.pat.clone() {
                return Some(pat_ident.ident);
            }
        }
        None
    });

    param_idents.for_each(|ident| {
        arena.intern_var_ident(&ident);
    });

    // Find the id of the variable of interest
    let var_id = arena.get_var_id(&var).expect("Unable to find variable");

    let (env, func_def) = parse_body(&mut arena, body);

    if func_def.is_none() {
        panic!("Function body must end with a bare expression (no trailing semicolon).");
    }

    let root = parse_syn(&mut arena, &func_def.unwrap(), &env);

    if options.prune {
        // Prune the expression tree to remove any nodes that are not ancestors of the root.
        arena.prune(root);
    }

    let tokens = (0..dim)
        .map(|i| compile_expression(&mut arena, root, var_id, i, &options))
        .collect::<Vec<_>>();

    // Count all references to each node to determine which sub-expressions are shared and should be hoisted into `let` bindings.
    let mut ref_count = RefCountVisitor::new();
    tokens.iter().for_each(|t| arena.accept(*t, &mut ref_count));

    let mut cache = HashMap::new();
    let mut instructions = Vec::new();

    // Generate tokens for the gradient expression, hoisting any shared sub-expressions into `let` bindings and caching their tokens for reuse.
    let mut to_tokens_visitor = ToTokenStreamVisitor::new(
        &ref_count.get_counts(),
        &mut cache,
        &mut instructions,
        &options,
    );

    let (dim, gradient_token) = {
        if options.sparse {
            // For sparse output, we must emit both the values and indices
            let sparse_tokens = tokens
                .iter()
                .enumerate()
                .filter(|(_, t)| &SymNode::Const(0) != arena.get_node(**t))
                .map(|(i, t)| (i, arena.accept(*t, &mut to_tokens_visitor)))
                .collect::<Vec<_>>();
            let indices = sparse_tokens
                .iter()
                .map(|(i, _)| quote! { #i })
                .collect::<Vec<_>>();
            let values = sparse_tokens
                .iter()
                .map(|(_, t)| t.clone())
                .collect::<Vec<_>>();
            (
                sparse_tokens.len(),
                quote! { ([#(#indices),*], [#(#values),*]) },
            )
        } else {
            let dense_tokens = tokens
                .iter()
                .map(|t| arena.accept(*t, &mut to_tokens_visitor))
                .collect::<Vec<_>>();
            (dim, quote! { [#(#dense_tokens),*] })
        }
    };

    // Get the emitted `let` bindings in the order they must be emitted.
    let instruction_tokens = to_tokens_visitor.get_instructions().to_vec();

    let return_type = if options.sparse {
        quote! { ([usize; #dim], [f64; #dim]) }
    } else {
        quote! { [f64; #dim] }
    };

    let grad_name = syn::Ident::new(
        &format!("{}_gradient", fn_name),
        proc_macro2::Span::call_site(),
    );

    let expanded = quote!(
        #input_fn

        #vis fn #grad_name(#params) -> #return_type {
            unsafe {
                #(#instruction_tokens)*
                #gradient_token
            }
        }
    );

    expanded.into()
}
