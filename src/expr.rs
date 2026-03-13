use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::Expr;

const DEFAULT_MAX_PASSES: usize = 10;

// ─────────────────────────────────────────────────────────────────────────────
// Symbolic expression tree
// ─────────────────────────────────────────────────────────────────────────────

/// A compile-time symbolic expression tree.
///
/// Variables are represented by their 0-based index into the function's `f64`
/// parameter list rather than by name, so that [`diff`](SymExpr::diff) and
/// [`into_token_stream`](SymExpr::into_token_stream) can work purely by index
/// without carrying string state.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum SymExpr {
    /// A numeric constant.
    Const(f64),
    /// The `i`-th `f64` parameter of the function being differentiated.
    Var(String, i32),
    /// `lhs + rhs`
    Add(Box<SymExpr>, Box<SymExpr>),
    /// `lhs - rhs`
    Sub(Box<SymExpr>, Box<SymExpr>),
    /// `lhs * rhs`
    Mul(Box<SymExpr>, Box<SymExpr>),
    /// `lhs / rhs`
    Div(Box<SymExpr>, Box<SymExpr>),
    /// Integer power: `base.powi(exp)`.
    Powi(Box<SymExpr>, i32),
    /// Negation: `-e`
    Neg(Box<SymExpr>),
    /// `e.sin()`
    Sin(Box<SymExpr>),
    /// `e.cos()`
    Cos(Box<SymExpr>),
    /// Natural logarithm: `e.ln()`
    Ln(Box<SymExpr>),
    /// Natural exponential: `e.exp()`
    Exp(Box<SymExpr>),
    /// `e.sqrt()`
    Sqrt(Box<SymExpr>),
    /// An expression that could not be parsed symbolically.
    ///
    /// Its derivative with respect to any variable is assumed to be zero.
    /// This is safe for sub-expressions that do not depend on the
    /// differentiation variable (e.g. a call to an opaque helper function),
    /// but will silently produce an incorrect derivative if the expression
    /// actually does depend on that variable.
    Opaque(String),
}

impl SymExpr {
    /// Algebraically simplify the expression by one bottom-up pass.
    ///
    /// Applies constant folding and structural identities:
    ///
    /// **Arithmetic**
    /// - `0 + e = e`, `e + 0 = e`
    /// - `0 * e = 0`, `1 * e = e`, `e * 1 = e`
    /// - `e / 1 = e`
    /// - `--e = e`
    ///
    /// **Factor extraction** (for `+` and `-`)
    /// - `(a*b) + (a*c) = a * (b+c)`, `(a*b) - (a*c) = a * (b-c)`
    ///
    /// **Power / exponential**
    /// - `base^a * base^b = base^(a+b)`, `base^a / base^b = base^(a-b)`
    /// - `exp(a) * exp(b) = exp(a+b)`, `exp(a) / exp(b) = exp(a-b)`
    /// - `sqrt(base^(2k)) = base^k`
    ///
    /// **Logarithm**
    /// - `ln(base^n) = n * ln(base)`
    ///
    /// **Constant folding** — any node whose children are all [`Const`](SymExpr::Const)
    /// is evaluated to a single `Const`.
    ///
    /// Use [`simplify_multipass`](Self::simplify_multipass) to reach a fixed point
    /// when rules may cascade across multiple levels.
    pub fn simplify(&self) -> Self {
        match self {
            SymExpr::Const(c) => SymExpr::Const(*c),
            SymExpr::Var(name, i) => SymExpr::Var(name.clone(), *i),
            SymExpr::Add(e1, e2) => {
                let se1 = e1.simplify();
                let se2 = e2.simplify();
                match (&se1, &se2) {
                    (_, SymExpr::Const(0.0)) => se1,
                    (SymExpr::Const(0.0), _) => se2,
                    (SymExpr::Const(c1), SymExpr::Const(c2)) => SymExpr::Const(c1 + c2),
                    (SymExpr::Mul(e1a, e1b), SymExpr::Mul(e2a, e2b))
                        if e1a == e2a || e1a == e2b =>
                    {
                        // (e1 * e2) + (e1 * e3) = e1 * (e2 + e3)
                        SymExpr::Mul(
                            Box::new(*e1a.clone()),
                            Box::new(SymExpr::Add(e1b.clone(), e2b.clone())),
                        )
                    }
                    (SymExpr::Mul(e1a, e1b), SymExpr::Mul(e2a, e2b))
                        if e1b == e2a || e1b == e2b =>
                    {
                        // (e1 * e2) + (e3 * e2) = (e1 + e3) * e2
                        SymExpr::Mul(
                            Box::new(SymExpr::Add(e1a.clone(), e2a.clone())),
                            Box::new(*e1b.clone()),
                        )
                    }
                    _ => SymExpr::Add(Box::new(se1), Box::new(se2)),
                }
            }
            SymExpr::Sub(e1, e2) => {
                let se1 = e1.simplify();
                let se2 = e2.simplify();
                match (&se1, &se2) {
                    (SymExpr::Const(0.0), _) => SymExpr::Neg(Box::new(se2)),
                    (_, SymExpr::Const(0.0)) => se1,
                    (SymExpr::Mul(e1a, e1b), SymExpr::Mul(e2a, e2b))
                        if e1a == e2a || e1a == e2b =>
                    {
                        // (e1 * e2) - (e1 * e3) = e1 * (e2 - e3)
                        SymExpr::Mul(
                            Box::new(*e1a.clone()),
                            Box::new(SymExpr::Sub(e1b.clone(), e2b.clone())),
                        )
                    }
                    (SymExpr::Mul(e1a, e1b), SymExpr::Mul(e2a, e2b))
                        if e1b == e2a || e1b == e2b =>
                    {
                        // (e1 * e2) - (e3 * e2) = (e1 - e3) * e2
                        SymExpr::Mul(
                            Box::new(SymExpr::Sub(e1a.clone(), e2a.clone())),
                            Box::new(*e1b.clone()),
                        )
                    }
                    (SymExpr::Const(c1), SymExpr::Const(c2)) => SymExpr::Const(c1 - c2),
                    _ => SymExpr::Sub(Box::new(se1), Box::new(se2)),
                }
            }
            SymExpr::Mul(e1, e2) => {
                let se1 = e1.simplify();
                let se2 = e2.simplify();
                match (&se1, &se2) {
                    (SymExpr::Const(0.0), _) | (_, SymExpr::Const(0.0)) => SymExpr::Const(0.0),
                    (SymExpr::Const(1.0), _) => se2,
                    (_, SymExpr::Const(1.0)) => se1,
                    (SymExpr::Const(c1), SymExpr::Const(c2)) => SymExpr::Const(c1 * c2),
                    (SymExpr::Powi(base1, exp1), SymExpr::Powi(base2, exp2)) if base1 == base2 => {
                        SymExpr::Powi(base1.clone(), exp1 + exp2)
                    } // base^a * base^b = base^(a+b)
                    (SymExpr::Exp(e1), SymExpr::Exp(e2)) => {
                        SymExpr::Exp(Box::new(SymExpr::Add(e1.clone(), e2.clone())))
                    } // exp(a) * exp(b) = exp(a + b)
                    _ => SymExpr::Mul(Box::new(se1), Box::new(se2)),
                }
            }
            SymExpr::Div(e1, e2) => {
                let se1 = e1.simplify();
                let se2 = e2.simplify();
                match (&se1, &se2) {
                    (SymExpr::Const(c1), SymExpr::Const(c2)) if *c2 != 0.0 => {
                        SymExpr::Const(c1 / c2)
                    }
                    (SymExpr::Const(0.0), _) => SymExpr::Const(0.0),
                    (_, SymExpr::Const(1.0)) => se1,
                    (SymExpr::Powi(base1, exp1), SymExpr::Powi(base2, exp2)) if base1 == base2 => {
                        SymExpr::Powi(base1.clone(), exp1 - exp2)
                    } // base^a / base^b = base^(a-b)
                    (SymExpr::Exp(e1), SymExpr::Exp(e2)) => {
                        SymExpr::Exp(Box::new(SymExpr::Sub(e1.clone(), e2.clone())))
                    } // exp(a) / exp(b) = exp(a - b)
                    _ => SymExpr::Div(Box::new(se1), Box::new(se2)),
                }
            }
            SymExpr::Powi(e, exp) => {
                let se = e.simplify();
                match &se {
                    SymExpr::Const(c) => SymExpr::Const(c.powi(*exp)),
                    _ => SymExpr::Powi(Box::new(se), *exp),
                }
            }
            SymExpr::Neg(e) => {
                let se = e.simplify();
                match &se {
                    SymExpr::Const(c) => SymExpr::Const(-c),
                    SymExpr::Neg(inner) => *inner.clone(), // --e = e
                    _ => SymExpr::Neg(Box::new(se)),
                }
            }
            SymExpr::Sin(e) => {
                let se = e.simplify();
                match &se {
                    SymExpr::Const(c) => SymExpr::Const(c.sin()),
                    _ => SymExpr::Sin(Box::new(se)),
                }
            }
            SymExpr::Cos(e) => {
                let se = e.simplify();
                match &se {
                    SymExpr::Const(c) => SymExpr::Const(c.cos()),
                    _ => SymExpr::Cos(Box::new(se)),
                }
            }
            SymExpr::Ln(e) => {
                let se = e.simplify();
                match &se {
                    SymExpr::Const(c) if *c > 0.0 => SymExpr::Const(c.ln()),
                    SymExpr::Powi(base, exp) => SymExpr::Mul(
                        Box::new(SymExpr::Const(*exp as f64)),
                        Box::new(SymExpr::Ln(base.clone())),
                    ), // ln(base^exp) = exp * ln(base)
                    _ => SymExpr::Ln(Box::new(se)),
                }
            }
            SymExpr::Exp(e) => {
                let se = e.simplify();
                match &se {
                    SymExpr::Const(c) => SymExpr::Const(c.exp()),
                    _ => SymExpr::Exp(Box::new(se)),
                }
            }
            SymExpr::Sqrt(e) => {
                let se = e.simplify();
                match &se {
                    SymExpr::Const(c) if *c >= 0.0 => SymExpr::Const(c.sqrt()),
                    SymExpr::Powi(base, exp) if *exp % 2 == 0 => {
                        // sqrt(base^(2k)) = base^k
                        SymExpr::Powi(base.clone(), *exp / 2)
                    }
                    _ => SymExpr::Sqrt(Box::new(se)),
                }
            }
            SymExpr::Opaque(e) => SymExpr::Opaque(e.clone()),
        }
    }

    /// Repeatedly call [`simplify`](Self::simplify) until the expression stops changing or
    /// `max_passes` iterations are reached (default: [`DEFAULT_MAX_PASSES`]).
    ///
    /// A single pass of `simplify` rewrites the tree bottom-up, so some identities
    /// (e.g. constant folding inside a newly-simplified sub-expression) may only
    /// become visible after the parent is revisited.  Multiple passes converge to
    /// a fixed point for all rules currently implemented.
    pub fn simplify_multipass(&self, max_passes: Option<usize>) -> Self {
        let mut simplified = self.clone();
        let max_passes = max_passes.unwrap_or(DEFAULT_MAX_PASSES);
        for _ in 0..max_passes {
            let new_simplified = simplified.simplify();
            if new_simplified == simplified {
                break;
            }
            simplified = new_simplified;
        }
        simplified
    }

    /// Symbolically differentiate with respect to parameter index `var`.
    ///
    /// Applies the standard differentiation rules:
    /// - **Sum rule**: `(f ± g)' = f' ± g'`
    /// - **Product rule**: `(fg)' = f'g + fg'`
    /// - **Quotient rule**: `(f/g)' = f'/g - fg'/g²`
    /// - **Power rule**: `(fⁿ)' = n·fⁿ⁻¹·f'`
    /// - **Chain rule**: applied to sin, cos, ln, exp, sqrt
    ///
    /// [`Opaque`](SymExpr::Opaque) sub-expressions are treated as constants
    /// (derivative = 0).  See the variant's documentation for the implications.
    pub fn diff(&self, var: String) -> Self {
        match self {
            SymExpr::Const(_) => SymExpr::Const(0.0),
            SymExpr::Var(name, i) => {
                if format!("{}[{}]", name, i) == var {
                    SymExpr::Const(1.0)
                } else {
                    SymExpr::Const(0.0)
                }
            }
            SymExpr::Add(e1, e2) => SymExpr::Add(
                Box::new(e1.diff(var.clone())),
                Box::new(e2.diff(var.clone())),
            ),
            SymExpr::Sub(e1, e2) => SymExpr::Sub(
                Box::new(e1.diff(var.clone())),
                Box::new(e2.diff(var.clone())),
            ),
            // Product rule: (e1*e2)' = e1'*e2 + e1*e2'
            SymExpr::Mul(e1, e2) => SymExpr::Add(
                Box::new(SymExpr::Mul(Box::new(e1.diff(var.clone())), e2.clone())),
                Box::new(SymExpr::Mul(e1.clone(), Box::new(e2.diff(var.clone())))),
            ),
            // Quotient rule: (e1/e2)' = e1'/e2 - e1*e2'/e2²
            SymExpr::Div(e1, e2) => SymExpr::Sub(
                Box::new(SymExpr::Div(Box::new(e1.diff(var.clone())), e2.clone())),
                Box::new(SymExpr::Div(
                    Box::new(SymExpr::Mul(e1.clone(), Box::new(e2.diff(var.clone())))),
                    Box::new(SymExpr::Powi(e2.clone(), 2)),
                )),
            ),
            // Power rule: (base^n)' = n * base^(n-1) * base'
            SymExpr::Powi(e1, exp) => {
                if *exp == 0 {
                    SymExpr::Const(0.0)
                } else if let SymExpr::Const(_) = **e1 {
                    // (c^n)' = 0
                    SymExpr::Const(0.0)
                } else {
                    let new_exp = *exp - 1;
                    let coeff = *exp as f64;
                    SymExpr::Mul(
                        Box::new(SymExpr::Mul(
                            Box::new(SymExpr::Const(coeff)),
                            Box::new(SymExpr::Powi(e1.clone(), new_exp)),
                        )),
                        Box::new(e1.diff(var)),
                    )
                }
            }
            SymExpr::Neg(e) => SymExpr::Neg(Box::new(e.diff(var))),
            // sin'(e) = cos(e) * e'
            SymExpr::Sin(e) => {
                let se = e.simplify();
                match &se {
                    SymExpr::Const(c) => SymExpr::Const(c.cos()),
                    _ => SymExpr::Mul(Box::new(SymExpr::Cos(Box::new(se))), Box::new(e.diff(var))),
                }
            }
            // cos'(e) = -sin(e) * e'
            SymExpr::Cos(e) => {
                let se = e.simplify();
                match &se {
                    SymExpr::Const(c) => SymExpr::Const(-c.sin()),
                    _ => SymExpr::Mul(
                        Box::new(SymExpr::Mul(
                            Box::new(SymExpr::Const(-1.0)),
                            Box::new(SymExpr::Sin(Box::new(se))),
                        )),
                        Box::new(e.diff(var)),
                    ),
                }
            }
            // ln'(e) = e' / e
            SymExpr::Ln(e) => {
                let se = e.simplify();
                match &se {
                    SymExpr::Const(c) if *c > 0.0 => SymExpr::Const(1.0 / c),
                    _ => SymExpr::Div(Box::new(e.diff(var)), Box::new(se)),
                }
            }
            // exp'(e) = exp(e) * e'
            SymExpr::Exp(e) => {
                let se = e.simplify();
                match &se {
                    SymExpr::Const(c) => SymExpr::Const(c.exp()),
                    _ => SymExpr::Mul(Box::new(SymExpr::Exp(Box::new(se))), Box::new(e.diff(var))),
                }
            }
            // sqrt'(e) = e' / (2 * sqrt(e))
            SymExpr::Sqrt(e) => {
                let se = e.simplify();
                match &se {
                    SymExpr::Const(c) if *c >= 0.0 => SymExpr::Const(0.5 / c.sqrt()),
                    _ => SymExpr::Div(
                        Box::new(e.diff(var)),
                        Box::new(SymExpr::Mul(
                            Box::new(SymExpr::Sqrt(Box::new(se))),
                            Box::new(SymExpr::Const(2.0)),
                        )),
                    ),
                }
            }
            SymExpr::Opaque(_) => SymExpr::Const(0.0),
        }
    }

    /// Emit this expression as a [`proc_macro2::TokenStream`] suitable for
    /// splicing into generated code.
    ///
    /// [`Var(name, i)`](SymExpr::Var) nodes emit `name[i]`, matching the slice
    /// parameter in the annotated function.  [`Const`](SymExpr::Const) values are
    /// emitted as `f64`-suffixed literals (e.g. `3f64`) to avoid type-inference
    /// failures in the generated function body.
    pub fn into_token_stream(self) -> TokenStream {
        match self {
            SymExpr::Const(c) => {
                let lit = proc_macro2::Literal::f64_suffixed(c);
                quote! { #lit }
            }
            SymExpr::Var(name, i) => {
                let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
                let idx = i as usize;
                quote! { #ident[#idx] }
            }
            SymExpr::Add(e1, e2) => {
                let ts1 = e1.into_token_stream();
                let ts2 = e2.into_token_stream();
                quote! { (#ts1 + #ts2) }
            }
            SymExpr::Sub(e1, e2) => {
                let ts1 = e1.into_token_stream();
                let ts2 = e2.into_token_stream();
                quote! { (#ts1 - #ts2) }
            }
            SymExpr::Mul(e1, e2) => {
                let ts1 = e1.into_token_stream();
                let ts2 = e2.into_token_stream();
                quote! { (#ts1 * #ts2) }
            }
            SymExpr::Div(e1, e2) => {
                let ts1 = e1.into_token_stream();
                let ts2 = e2.into_token_stream();
                quote! { (#ts1 / #ts2) }
            }
            SymExpr::Powi(e, exp) => {
                let ts = e.into_token_stream();
                quote! { (#ts).powi(#exp) }
            }
            SymExpr::Neg(e) => {
                let ts = e.into_token_stream();
                quote! { -(#ts) }
            }
            SymExpr::Sin(e) => {
                let ts = e.into_token_stream();
                quote! { (#ts).sin() }
            }
            SymExpr::Cos(e) => {
                let ts = e.into_token_stream();
                quote! { (#ts).cos() }
            }
            SymExpr::Ln(e) => {
                let ts = e.into_token_stream();
                quote! { (#ts).ln() }
            }
            SymExpr::Exp(e) => {
                let ts = e.into_token_stream();
                quote! { (#ts).exp() }
            }
            SymExpr::Sqrt(e) => {
                let ts = e.into_token_stream();
                quote! { (#ts).sqrt() }
            }
            SymExpr::Opaque(ts) => ts.into_token_stream(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// syn::Expr → SymExpr
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a parsed [`syn::Expr`] into a [`SymExpr`].
///
/// `bindings` maps local names to their symbolic form and is used to inline
/// `let`-bindings encountered earlier in the function body.  Slice indexing
/// (`x[i]`) is recognised as [`SymExpr::Var`]; identifiers present in
/// `bindings` are substituted; everything else becomes [`SymExpr::Opaque`].
///
/// # Supported syntax
///
/// | Source form | Result |
/// |---|---|
/// | `1.0`, `2` | [`Const`](SymExpr::Const) |
/// | `x[i]` | [`Var("x", i)`](SymExpr::Var) |
/// | `a` (in bindings) | substituted [`SymExpr`] |
/// | `e + f`, `e - f`, `e * f`, `e / f` | [`Add`](SymExpr::Add) / [`Sub`](SymExpr::Sub) / [`Mul`](SymExpr::Mul) / [`Div`](SymExpr::Div) |
/// | `-e` | [`Neg`](SymExpr::Neg) |
/// | `e.sin()`, `e.cos()`, `e.ln()`, `e.exp()`, `e.sqrt()` | trig/transcendental nodes |
/// | `e.powi(n)` (const `n`) | [`Powi`](SymExpr::Powi) |
/// | `(e)`, `e as f64` | transparent — inner expr used |
/// | anything else | [`Opaque`](SymExpr::Opaque) |
pub(super) fn syn_to_sym(expr: &Expr, bindings: &HashMap<String, SymExpr>) -> SymExpr {
    match expr {
        Expr::Lit(lit) => {
            if let syn::Lit::Float(f) = &lit.lit {
                SymExpr::Const(f.base10_parse::<f64>().unwrap())
            } else if let syn::Lit::Int(i) = &lit.lit {
                SymExpr::Const(i.base10_parse::<f64>().unwrap())
            } else {
                SymExpr::Opaque(quote! { #lit }.to_string())
            }
        }

        Expr::Path(path) => {
            if let Some(ident) = path.path.get_ident() {
                let name = ident.to_string();
                if let Some(sym) = bindings.get(&name) {
                    sym.clone()
                } else {
                    SymExpr::Opaque(quote! { #path }.to_string())
                }
            } else {
                SymExpr::Opaque(quote! { #path }.to_string())
            }
        }

        Expr::Index(index) => SymExpr::Var(
            index.expr.to_token_stream().to_string(),
            index
                .index
                .to_token_stream()
                .to_string()
                .parse::<i32>()
                .unwrap(),
        ),

        Expr::Binary(bin) => {
            let left = syn_to_sym(&bin.left, bindings);
            let right = syn_to_sym(&bin.right, bindings);
            match bin.op {
                syn::BinOp::Add(_) => SymExpr::Add(Box::new(left), Box::new(right)),
                syn::BinOp::Sub(_) => SymExpr::Sub(Box::new(left), Box::new(right)),
                syn::BinOp::Mul(_) => SymExpr::Mul(Box::new(left), Box::new(right)),
                syn::BinOp::Div(_) => SymExpr::Div(Box::new(left), Box::new(right)),
                _ => SymExpr::Opaque(quote! { #bin }.to_string()),
            }
        }

        Expr::Unary(un) => {
            let operand = syn_to_sym(&un.expr, bindings);
            match un.op {
                syn::UnOp::Neg(_) => SymExpr::Neg(Box::new(operand)),
                _ => SymExpr::Opaque(quote! { #un }.to_string()),
            }
        }

        // Method calls: `x.sin()`, `x.pow(n)`, etc.
        Expr::MethodCall(call) => {
            let receiver = syn_to_sym(&call.receiver, bindings);
            let args: Vec<SymExpr> = call
                .args
                .iter()
                .map(|arg| syn_to_sym(arg, bindings))
                .collect();
            match call.method.to_string().as_str() {
                "sin" if args.is_empty() => SymExpr::Sin(Box::new(receiver)),
                "cos" if args.is_empty() => SymExpr::Cos(Box::new(receiver)),
                "ln" if args.is_empty() => SymExpr::Ln(Box::new(receiver)),
                "exp" if args.is_empty() => SymExpr::Exp(Box::new(receiver)),
                "sqrt" if args.is_empty() => SymExpr::Sqrt(Box::new(receiver)),
                "powi" if args.len() == 1 => {
                    if let SymExpr::Const(exp) = &args[0] {
                        SymExpr::Powi(Box::new(receiver), *exp as i32)
                    } else {
                        SymExpr::Opaque(quote! { #call }.to_string())
                    }
                }
                _ => SymExpr::Opaque(quote! { #call }.to_string()),
            }
        }

        // Free-function calls: `sin(x)`, `f64::sin(x)`, etc.
        Expr::Call(call) => {
            if let Expr::Path(path) = &*call.func {
                if let Some(ident) = path.path.get_ident() {
                    let func_name = ident.to_string();
                    let args: Vec<SymExpr> = call
                        .args
                        .iter()
                        .map(|arg| syn_to_sym(arg, bindings))
                        .collect();
                    match func_name.as_str() {
                        "sin" if args.len() == 1 => return SymExpr::Sin(Box::new(args[0].clone())),
                        "cos" if args.len() == 1 => return SymExpr::Cos(Box::new(args[0].clone())),
                        "ln" if args.len() == 1 => return SymExpr::Ln(Box::new(args[0].clone())),
                        "exp" if args.len() == 1 => return SymExpr::Exp(Box::new(args[0].clone())),
                        "sqrt" if args.len() == 1 => {
                            return SymExpr::Sqrt(Box::new(args[0].clone()));
                        }
                        "powi" if args.len() == 2 => {
                            if let SymExpr::Const(exp) = &args[1] {
                                return SymExpr::Powi(Box::new(args[0].clone()), *exp as i32);
                            }
                        }
                        _ => {}
                    }
                }
            }
            SymExpr::Opaque(quote! { #call }.to_string())
        }

        // Transparent wrappers — recurse into the inner expression.
        Expr::Paren(paren) => syn_to_sym(&paren.expr, bindings),
        Expr::Group(group) => syn_to_sym(&group.expr, bindings),
        // `x as f64` — strip the cast and differentiate the inner expression.
        Expr::Cast(c) => syn_to_sym(&c.expr, bindings),

        _ => SymExpr::Opaque(quote! { #expr }.to_string()),
    }
}
