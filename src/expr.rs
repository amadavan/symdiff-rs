//! Symbolic expression tree, arena, and compile-time differentiation.
//!
//! # Architecture
//!
//! Expressions are stored as nodes in a [`SymArena`] and referenced by integer
//! [`NodeId`]s.  The arena de-duplicates nodes on insertion (hash-consing), so
//! structurally identical sub-expressions always share the same id.
//!
//! Two traversal patterns are provided:
//!
//! - **[`SymVisitor`]** – a read-only, bottom-up tree walk.  Implementors
//!   receive pre-computed child results together with the arena so they can
//!   inspect sibling nodes if needed.
//! - **[`SymTransformer`]** – a rewriting pass that maps every node to a new
//!   node id.  [`SymArena::transform`] drives it in topological order, passing
//!   a `diff_map: HashMap<old_id, new_id>` so each handler can look up the
//!   already-transformed children.
//!
//! ## Constant representation
//!
//! [`SymNode::Const`] stores its value as the raw IEEE-754 bit pattern of an
//! `f64` (`u64::from_bits` / `f64::to_bits`).  All code that creates or
//! inspects constant nodes must go through `f64::to_bits` / `f64::from_bits`
//! rather than using integer literals directly (except `0`, whose bit pattern
//! happens to equal `0u64`).

#![allow(unused)]

use std::collections::{HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{Expr, Ident};

/// Index into a [`SymArena`]; uniquely identifies a [`SymNode`].
type NodeId = usize;

/// A single node in the symbolic expression tree.
///
/// All child references are [`NodeId`]s into the owning [`SymArena`].
/// Constants are encoded as the raw `f64` bit pattern (`f64::to_bits`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SymNode {
    /// A numeric constant; the payload is `value.to_bits()`.
    Const(u64),
    /// The `idx`-th component of the input slice `x`, i.e. `x[idx]`.
    Var(NodeId),
    /// Addition: `left + right`.
    Add(NodeId, NodeId),
    /// Subtraction: `left - right`.
    Sub(NodeId, NodeId),
    /// Multiplication: `left * right`.
    Mul(NodeId, NodeId),
    /// Division: `left / right`.
    Div(NodeId, NodeId),
    /// Integer power: `base.powi(exp)`.
    Powi(NodeId, i32),
    /// Arithmetic negation: `-operand`.
    Neg(NodeId),
    /// Sine: `operand.sin()`.
    Sin(NodeId),
    /// Cosine: `operand.cos()`.
    Cos(NodeId),
    /// Natural logarithm: `operand.ln()`.
    Ln(NodeId),
    /// Natural exponential: `operand.exp()`.
    Exp(NodeId),
    /// Square root: `operand.sqrt()`.
    Sqrt(NodeId),
}

/// An interning arena for [`SymNode`]s.
///
/// Nodes are stored in a flat `Vec` and looked up by index ([`NodeId`]).
/// [`SymArena::intern`] guarantees that each structurally distinct node is
/// stored at most once, so two equal nodes always get the same `NodeId`.
pub struct SymArena {
    nodes: Vec<SymNode>,
    lookup: HashMap<SymNode, usize>,
}

impl SymArena {
    /// Create an empty arena.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            lookup: HashMap::new(),
        }
    }

    /// Return a reference to the node stored at `node_id`.
    pub fn get_node(&self, node_id: NodeId) -> &SymNode {
        &self.nodes[node_id]
    }

    /// Dispatch `node_id` to the matching `visitor` method and return the result.
    pub fn accept<V, T>(&self, node_id: NodeId, visitor: &mut V) -> T
    where
        V: SymVisitor<T>,
    {
        let node = &self.nodes[node_id];
        match node {
            SymNode::Const(value) => visitor.visit_const(*value, self),
            SymNode::Var(idx) => visitor.visit_var(*idx, self),
            SymNode::Add(left, right) => visitor.visit_add(*left, *right, self),
            SymNode::Sub(left, right) => visitor.visit_sub(*left, *right, self),
            SymNode::Mul(left, right) => visitor.visit_mul(*left, *right, self),
            SymNode::Div(left, right) => visitor.visit_div(*left, *right, self),
            SymNode::Powi(base, exp) => visitor.visit_powi(*base, *exp, self),
            SymNode::Neg(operand) => visitor.visit_neg(*operand, self),
            SymNode::Sin(operand) => visitor.visit_sin(*operand, self),
            SymNode::Cos(operand) => visitor.visit_cos(*operand, self),
            SymNode::Ln(operand) => visitor.visit_ln(*operand, self),
            SymNode::Exp(operand) => visitor.visit_exp(*operand, self),
            SymNode::Sqrt(operand) => visitor.visit_sqrt(*operand, self),
        }
    }

    /// Insert `node` into the arena, returning its `NodeId`.
    ///
    /// If an identical node already exists, its existing id is returned instead
    /// (hash-consing / structural sharing).
    pub fn intern(&mut self, node: SymNode) -> usize {
        if let Some(&idx) = self.lookup.get(&node) {
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(node.clone());
            self.lookup.insert(node, idx);
            idx
        }
    }

    /// Apply `transformer` to every node reachable from `root` in bottom-up
    /// topological order, returning the new `NodeId` that corresponds to `root`.
    ///
    /// `diff` is an *output* parameter populated by the caller (pass an empty
    /// `HashMap`).  On return it maps each original `NodeId` to the transformed
    /// `NodeId` produced by the transformer.
    fn transform<T: SymTransformer>(
        &mut self,
        root: NodeId,
        transformer: &T,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let order = self.get_topological_order(root);

        let mut diff_map = HashMap::new();

        for i in order {
            let node = self.nodes[i];
            let diff = match node {
                SymNode::Const(value) => transformer.process_const(value, self, &mut diff_map),
                SymNode::Var(idx) => transformer.process_var(idx, self, &mut diff_map),
                SymNode::Add(left, right) => {
                    transformer.process_add(left, right, self, &mut diff_map)
                }
                SymNode::Sub(left, right) => {
                    transformer.process_sub(left, right, self, &mut diff_map)
                }
                SymNode::Mul(left, right) => {
                    transformer.process_mul(left, right, self, &mut diff_map)
                }
                SymNode::Div(left, right) => {
                    transformer.process_div(left, right, self, &mut diff_map)
                }
                SymNode::Powi(base, exp) => {
                    transformer.process_powi(base, exp, self, &mut diff_map)
                }
                SymNode::Neg(operand) => transformer.process_neg(operand, self, &mut diff_map),
                SymNode::Sin(operand) => transformer.process_sin(operand, self, &mut diff_map),
                SymNode::Cos(operand) => transformer.process_cos(operand, self, &mut diff_map),
                SymNode::Ln(operand) => transformer.process_ln(operand, self, &mut diff_map),
                SymNode::Exp(operand) => transformer.process_exp(operand, self, &mut diff_map),
                SymNode::Sqrt(operand) => transformer.process_sqrt(operand, self, &mut diff_map),
            };
            diff_map.insert(i, diff);
        }

        diff_map[&root]
    }

    /// Return the nodes reachable from `root_id` in bottom-up (post-order)
    /// topological order, visiting each node exactly once.
    pub fn get_topological_order(&self, root_id: NodeId) -> Vec<NodeId> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        let mut stack = vec![(root_id, false)];

        while let Some((node_id, visited_children)) = stack.pop() {
            // Check if all children have been visited
            if visited_children {
                order.push(node_id);
            } else if visited.insert(node_id) {
                // Otherwise we need to add children first
                let node = &self.nodes[node_id];
                stack.push((node_id, true));
                // Add children
                match node {
                    SymNode::Const(_) | SymNode::Var(_) => {}
                    SymNode::Add(left, right)
                    | SymNode::Sub(left, right)
                    | SymNode::Mul(left, right)
                    | SymNode::Div(left, right) => {
                        stack.push((*left, false));
                        stack.push((*right, false));
                    }
                    SymNode::Powi(base, _)
                    | SymNode::Neg(base)
                    | SymNode::Sin(base)
                    | SymNode::Cos(base)
                    | SymNode::Ln(base)
                    | SymNode::Exp(base)
                    | SymNode::Sqrt(base) => {
                        stack.push((*base, false));
                    }
                }
            }
        }

        order
    }
}

/// A rewriting pass over the expression tree.
///
/// [`SymArena::transform`] calls one method per node in bottom-up topological
/// order.  Each method receives the original child `NodeId`s plus a `diff` map
/// that already contains the transformed ids for every child; implementors look
/// up `diff[&child_id]` to obtain the rewritten child before building the new
/// node.  The returned `NodeId` is then recorded in `diff` under the current
/// node's id so parent nodes can find it in turn.
pub trait SymTransformer {
    fn process_const(
        &self,
        value: u64,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId;
    fn process_var(
        &self,
        idx: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId;
    fn process_add(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId;
    fn process_sub(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId;
    fn process_mul(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId;
    fn process_div(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId;
    fn process_powi(
        &self,
        base: NodeId,
        exp: i32,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId;
    fn process_neg(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId;
    fn process_sin(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId;
    fn process_cos(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId;
    fn process_ln(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId;
    fn process_exp(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId;
    fn process_sqrt(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId;
}

/// A read-only visitor over the expression tree.
///
/// [`SymArena::accept`] dispatches to the method matching the node kind.
/// Unlike [`SymTransformer`], the visitor is responsible for recursing into
/// children itself (by calling `arena.accept(child_id, self)`) if it needs
/// their values.
pub trait SymVisitor<T> {
    fn visit_const(&mut self, value: u64, arena: &SymArena) -> T;
    fn visit_var(&mut self, idx: NodeId, arena: &SymArena) -> T;
    fn visit_add(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> T;
    fn visit_sub(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> T;
    fn visit_mul(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> T;
    fn visit_div(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> T;
    fn visit_powi(&mut self, base: NodeId, exp: i32, arena: &SymArena) -> T;
    fn visit_neg(&mut self, operand: NodeId, arena: &SymArena) -> T;
    fn visit_sin(&mut self, operand: NodeId, arena: &SymArena) -> T;
    fn visit_cos(&mut self, operand: NodeId, arena: &SymArena) -> T;
    fn visit_ln(&mut self, operand: NodeId, arena: &SymArena) -> T;
    fn visit_exp(&mut self, operand: NodeId, arena: &SymArena) -> T;
    fn visit_sqrt(&mut self, operand: NodeId, arena: &SymArena) -> T;
}

/// A [`SymTransformer`] that computes the symbolic derivative of every node
/// with respect to a single variable `var` (identified by its [`NodeId`]).
///
/// The rules applied are the standard calculus identities: sum rule, product
/// rule, quotient rule, power rule, and chain rule for each transcendental.
struct DiffTransformer {
    /// The variable we are differentiating with respect to.
    var: NodeId,
}

impl DiffTransformer {
    pub fn new(var: NodeId) -> DiffTransformer {
        DiffTransformer { var }
    }
}

impl SymTransformer for DiffTransformer {
    fn process_const(
        &self,
        value: u64,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        // Derivative of a constant is zero
        arena.intern(SymNode::Const(0))
    }

    fn process_var(
        &self,
        idx: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        if idx == self.var {
            // Derivative of the variable with respect to itself is one
            arena.intern(SymNode::Const(1.0_f64.to_bits()))
        } else {
            // Derivative of other variables is zero
            arena.intern(SymNode::Const(0.0_f64.to_bits()))
        }
    }

    fn process_add(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_diff = diff[&left];
        let right_diff = diff[&right];
        arena.intern(SymNode::Add(left_diff, right_diff))
    }

    fn process_sub(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_diff = diff[&left];
        let right_diff = diff[&right];
        arena.intern(SymNode::Sub(left_diff, right_diff))
    }

    fn process_mul(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_diff = diff[&left];
        let right_diff = diff[&right];

        // Product rule: (f * g)' = f' * g + f * g'
        let term1 = arena.intern(SymNode::Mul(left_diff, right));
        let term2 = arena.intern(SymNode::Mul(left, right_diff));
        arena.intern(SymNode::Add(term1, term2))
    }

    fn process_div(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_diff = diff[&left];
        let right_diff = diff[&right];

        // Quotient rule: (f / g)' = (f' * g - f * g') / (g * g)
        let numerator1 = arena.intern(SymNode::Mul(left_diff, right));
        let numerator2 = arena.intern(SymNode::Mul(left, right_diff));
        let numerator = arena.intern(SymNode::Sub(numerator1, numerator2));
        let denominator = arena.intern(SymNode::Mul(right, right));
        arena.intern(SymNode::Div(numerator, denominator))
    }

    fn process_powi(
        &self,
        base: NodeId,
        exp: i32,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let base_diff = diff[&base];
        // Power rule: (f^n)' = n * f^(n-1) * f'
        let exp_node = arena.intern(SymNode::Const((exp as f64).to_bits()));
        let pow_node = arena.intern(SymNode::Powi(base, exp - 1));
        let term = arena.intern(SymNode::Mul(exp_node, pow_node));
        arena.intern(SymNode::Mul(term, base_diff))
    }

    fn process_neg(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let operand_diff = diff[&operand];
        arena.intern(SymNode::Neg(operand_diff))
    }

    fn process_sin(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        // Chain rule: (sin(f))' = cos(f) * f'
        let operand_diff = diff[&operand];
        let cos_operand = arena.intern(SymNode::Cos(operand));
        arena.intern(SymNode::Mul(operand_diff, cos_operand))
    }

    fn process_cos(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        // Chain rule: (cos(f))' = -sin(f) * f'
        let operand_diff = diff[&operand];
        let sin_operand = arena.intern(SymNode::Sin(operand));
        let neg_sin_operand = arena.intern(SymNode::Neg(sin_operand));
        arena.intern(SymNode::Mul(operand_diff, neg_sin_operand))
    }

    fn process_exp(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        // Chain rule: (exp(f))' = exp(f) * f'
        let operand_diff = diff[&operand];
        let exp_operand = arena.intern(SymNode::Exp(operand));
        arena.intern(SymNode::Mul(operand_diff, exp_operand))
    }

    fn process_ln(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        // Chain rule: (ln(f))' = f' / f
        let operand_diff = diff[&operand];
        arena.intern(SymNode::Div(operand_diff, operand))
    }

    fn process_sqrt(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        // Chain rule: (sqrt(f))' = f' / (2 * sqrt(f))
        let operand_diff = diff[&operand];
        let sqrt_operand = arena.intern(SymNode::Sqrt(operand));
        let two = arena.intern(SymNode::Const(2.0_f64.to_bits()));
        let denominator = arena.intern(SymNode::Mul(two, sqrt_operand));
        arena.intern(SymNode::Div(operand_diff, denominator))
    }
}

/// A [`SymTransformer`] that algebraically simplifies an expression tree.
///
/// Each `process_*` method receives the already-simplified child ids (via the
/// `diff` map) and applies a set of local rewrite rules before rebuilding the
/// node.  Rules include:
///
/// - **Constant folding**: if all operands are `Const`, evaluate the operation
///   at compile time and return a single `Const`.
/// - **Identity / annihilator laws**: `0 + e = e`, `e * 1 = e`, `0 * e = 0`,
///   `0 / e = 0`, `e / 1 = e`, `0 - e = -e`, `e - 0 = e`.
/// - **Double negation**: `-(-e) = e`.
/// - **Distributive factoring for `+`/`-`**: `a*b ± a*c = a*(b ± c)`.
/// - **Power merging**: `b^m * b^n = b^(m+n)`, `b^m / b^n = b^(m-n)`.
/// - **Exponential merging**: `exp(a) * exp(b) = exp(a+b)`,
///   `exp(a) / exp(b) = exp(a-b)`.
/// - **Logarithm of a power**: `ln(b^n) = n * ln(b)`.
/// - **Square root of an even power**: `sqrt(b^(2k)) = b^k`.
struct SimplifyTransformer {}

impl SimplifyTransformer {
    pub fn new() -> SimplifyTransformer {
        SimplifyTransformer {}
    }
}

impl SymTransformer for SimplifyTransformer {
    fn process_const(
        &self,
        value: u64,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        // Constants are already simplified
        arena.intern(SymNode::Const(value))
    }

    fn process_var(
        &self,
        idx: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        // Variables are already simplified
        arena.intern(SymNode::Var(idx))
    }

    fn process_add(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        // Simplify addition
        let left_id = diff[&left];
        let right_id = diff[&right];
        let left_node = arena.nodes[left_id];
        let right_node = arena.nodes[right_id];

        match (left_node, right_node) {
            (SymNode::Const(value1), SymNode::Const(value2)) => {
                // Constant folding: if both sides are constants, evaluate them
                let v1_bits = f64::from_bits(value1);
                let v2_bits = f64::from_bits(value2);
                let result_value = (v1_bits + v2_bits).to_bits();
                return arena.intern(SymNode::Const(result_value));
            }
            (SymNode::Const(v), _) if v == 0.0_f64.to_bits() => {
                // 0 + e = e
                return right_id;
            }
            (_, SymNode::Const(v)) if v == 0.0_f64.to_bits() => {
                // e + 0 = e
                return left_id;
            }
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1a == e2a => {
                // (a*b) + (a*c) = a * (b+c)
                let factored = arena.intern(SymNode::Add(e1b, e2b));
                return arena.intern(SymNode::Mul(e1a, factored));
            }
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1a == e2b => {
                // (a*b) + (c*a) = a * (b+c)
                let factored = arena.intern(SymNode::Add(e1b, e2a));
                return arena.intern(SymNode::Mul(e1a, factored));
            }
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1b == e2a => {
                // (b*a) + (a*c) = a * (b+c)
                let factored = arena.intern(SymNode::Add(e1a, e2b));
                return arena.intern(SymNode::Mul(e1b, factored));
            }
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1b == e2b => {
                // (b*a) + (c*a) = a * (b+c)
                let factored = arena.intern(SymNode::Add(e1a, e2a));
                return arena.intern(SymNode::Mul(e1b, factored));
            }
            _ => {}
        }
        arena.intern(SymNode::Add(left_id, right_id))
    }

    fn process_sub(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_id = diff[&left];
        let right_id = diff[&right];
        let left_node = arena.nodes[left_id];
        let right_node = arena.nodes[right_id];

        match (left_node, right_node) {
            (SymNode::Const(value1), SymNode::Const(value2)) => {
                // Constant folding: if both sides are constants, evaluate them
                let v1_bits = f64::from_bits(value1);
                let v2_bits = f64::from_bits(value2);
                let result_value = (v1_bits - v2_bits).to_bits();
                return arena.intern(SymNode::Const(result_value));
            }
            (SymNode::Const(v), _) if v == 0.0_f64.to_bits() => {
                // 0 - e = -e
                return arena.intern(SymNode::Neg(right_id));
            }
            (_, SymNode::Const(v)) if v == 0.0_f64.to_bits() => {
                // e - 0 = e
                return left_id;
            }
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1a == e2a => {
                // (a*b) - (a*c) = a * (b-c)
                let factored = arena.intern(SymNode::Sub(e1b, e2b));
                return arena.intern(SymNode::Mul(e1a, factored));
            }
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1a == e2b => {
                // (a*b) - (c*a) = a * (b-c)
                let factored = arena.intern(SymNode::Sub(e1b, e2a));
                return arena.intern(SymNode::Mul(e1a, factored));
            }
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1b == e2a => {
                // (b*a) - (a*c) = a * (b-c)
                let factored = arena.intern(SymNode::Sub(e1a, e2b));
                return arena.intern(SymNode::Mul(e1b, factored));
            }
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1b == e2b => {
                // (b*a) - (c*a) = a * (b-c)
                let factored = arena.intern(SymNode::Sub(e1a, e2a));
                return arena.intern(SymNode::Mul(e1b, factored));
            }
            _ => {}
        }
        arena.intern(SymNode::Sub(left_id, right_id))
    }

    fn process_mul(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_id = diff[&left];
        let right_id = diff[&right];
        let left_node = arena.nodes[left_id];
        let right_node = arena.nodes[right_id];

        match (left_node, right_node) {
            (SymNode::Const(value1), SymNode::Const(value2)) => {
                // Constant folding: if both sides are constants, evaluate them
                let v1_bits = f64::from_bits(value1);
                let v2_bits = f64::from_bits(value2);
                let result_value = (v1_bits * v2_bits).to_bits();
                return arena.intern(SymNode::Const(result_value));
            }
            (SymNode::Const(v), _) | (_, SymNode::Const(v)) if v == 0.0_f64.to_bits() => {
                // 0 * e = 0, e * 0 = 0
                return arena.intern(SymNode::Const(0.0_f64.to_bits()));
            }
            (SymNode::Const(v), _) if v == 1.0_f64.to_bits() => {
                // 1 * e = e
                return right_id;
            }
            (_, SymNode::Const(v)) if v == 1.0_f64.to_bits() => {
                // e * 1 = e
                return left_id;
            }
            (SymNode::Powi(base1, exp1), SymNode::Powi(base2, exp2)) if base1 == base2 => {
                // base^a * base^b = base^(a+b)
                let new_exp = exp1 + exp2;
                return arena.intern(SymNode::Powi(base1, new_exp));
            }
            (SymNode::Exp(e1), SymNode::Exp(e2)) => {
                // exp(a) * exp(b) = exp(a+b)
                let new_exp = arena.intern(SymNode::Add(e1, e2));
                return arena.intern(SymNode::Exp(new_exp));
            }
            _ => {}
        }
        arena.intern(SymNode::Mul(left_id, right_id))
    }

    fn process_div(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_id = diff[&left];
        let right_id = diff[&right];
        let left_node = arena.nodes[left_id];
        let right_node = arena.nodes[right_id];

        match (left_node, right_node) {
            (SymNode::Const(value1), SymNode::Const(value2)) if value2 != 0.0_f64.to_bits() => {
                // Constant folding: if both sides are constants, evaluate them
                let v1_bits = f64::from_bits(value1);
                let v2_bits = f64::from_bits(value2);
                let result_value = (v1_bits / v2_bits).to_bits();
                return arena.intern(SymNode::Const(result_value));
            }
            (SymNode::Const(v), _) if v == 0.0_f64.to_bits() => {
                // 0 / e = 0
                return arena.intern(SymNode::Const(0.0_f64.to_bits()));
            }
            (_, SymNode::Const(v)) if v == 1.0_f64.to_bits() => {
                // e / 1 = e
                return left_id;
            }
            (SymNode::Exp(e1), SymNode::Exp(e2)) => {
                // exp(a) / exp(b) = exp(a-b)
                let new_exp = arena.intern(SymNode::Sub(e1, e2));
                return arena.intern(SymNode::Exp(new_exp));
            }
            (SymNode::Powi(base1, exp1), SymNode::Powi(base2, exp2)) if base1 == base2 => {
                // base^a / base^b = base^(a-b)
                let new_exp = exp1 - exp2;
                return arena.intern(SymNode::Powi(base1, new_exp));
            }
            _ => {}
        }
        arena.intern(SymNode::Div(left_id, right_id))
    }

    fn process_powi(
        &self,
        base: NodeId,
        exp: i32,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let base_id = diff[&base];
        let base_node = arena.nodes[base_id];

        match base_node {
            SymNode::Const(value) => {
                // Constant folding: if the base is a constant, evaluate it
                let base_value = f64::from_bits(value);
                let result_value = base_value.powi(exp).to_bits();
                return arena.intern(SymNode::Const(result_value));
            }
            _ => {}
        }
        arena.intern(SymNode::Powi(base_id, exp))
    }

    fn process_neg(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let operand_id = diff[&operand];
        let operand_node = arena.nodes[operand_id];

        match operand_node {
            SymNode::Const(value) => {
                // -c = (-c) for constants
                let c = f64::from_bits(value);
                let neg_c = -c;
                return arena.intern(SymNode::Const(neg_c.to_bits()));
            }
            SymNode::Neg(inner) => {
                // --e = e
                return inner;
            }
            _ => {}
        }
        arena.intern(SymNode::Neg(operand_id))
    }

    fn process_sin(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let operand_id = diff[&operand];
        let operand_node = arena.nodes[operand_id];

        match operand_node {
            SymNode::Const(value) => {
                // sin(c) = c.sin() for constants
                let c = f64::from_bits(value);
                let sin_c = c.sin();
                return arena.intern(SymNode::Const(sin_c.to_bits()));
            }
            _ => {}
        }
        arena.intern(SymNode::Sin(operand_id))
    }

    fn process_cos(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let operand_id = diff[&operand];
        let operand_node = arena.nodes[operand_id];

        match operand_node {
            SymNode::Const(value) => {
                // cos(c) = c.cos() for constants
                let c = f64::from_bits(value);
                let cos_c = c.cos();
                return arena.intern(SymNode::Const(cos_c.to_bits()));
            }
            _ => {}
        }
        arena.intern(SymNode::Cos(operand_id))
    }

    fn process_ln(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let operand_id = diff[&operand];
        let operand_node = arena.nodes[operand_id];

        match operand_node {
            SymNode::Const(value) => {
                // ln(c) = c.ln() for positive constants
                let c = f64::from_bits(value);
                if c > 0.0 {
                    let ln_c = c.ln();
                    return arena.intern(SymNode::Const(ln_c.to_bits()));
                }
            }
            SymNode::Powi(base, exp) => {
                // ln(base^exp) = exp * ln(base)
                let ln_base = arena.intern(SymNode::Ln(base));
                let exp_node = arena.intern(SymNode::Const((exp as f64).to_bits()));
                return arena.intern(SymNode::Mul(exp_node, ln_base));
            }
            _ => {}
        }
        arena.intern(SymNode::Ln(operand_id))
    }

    fn process_exp(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let operand_id = diff[&operand];
        let operand_node = arena.nodes[operand_id];

        match operand_node {
            SymNode::Const(value) => {
                // exp(c) = c.exp() for constants
                let c = f64::from_bits(value);
                let exp_c = c.exp();
                return arena.intern(SymNode::Const(exp_c.to_bits()));
            }
            _ => {}
        }
        arena.intern(SymNode::Exp(operand_id))
    }

    fn process_sqrt(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let operand_id = diff[&operand];
        let operand_node = arena.nodes[operand_id];

        match operand_node {
            SymNode::Const(value) => {
                // sqrt(c) = c.sqrt() for positive constants
                let c = f64::from_bits(value);
                if c >= 0.0 {
                    let sqrt_c = c.sqrt();
                    return arena.intern(SymNode::Const(sqrt_c.to_bits()));
                }
            }
            SymNode::Powi(base, exp) if exp % 2 == 0 => {
                // sqrt(base^(2k)) = base^k
                let new_exp = exp / 2;
                return arena.intern(SymNode::Powi(base, new_exp));
            }
            _ => {}
        }
        arena.intern(SymNode::Sqrt(operand_id))
    }
}

struct CostEstimateVisitor<'a> {
    arena: &'a SymArena,
    costs: HashMap<SymNode, usize>,
}

impl CostEstimateVisitor<'_> {
    pub fn new(arena: &'_ SymArena, costs: HashMap<SymNode, usize>) -> CostEstimateVisitor<'_> {
        CostEstimateVisitor { arena, costs }
    }
}

impl SymVisitor<usize> for CostEstimateVisitor<'_> {
    fn visit_const(&mut self, value: u64, arena: &SymArena) -> usize {
        *self.costs.get(&SymNode::Const(0)).unwrap_or(&0)
    }

    fn visit_var(&mut self, idx: NodeId, arena: &SymArena) -> usize {
        *self.costs.get(&SymNode::Var(0)).unwrap_or(&0)
    }

    fn visit_add(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> usize {
        let left_cost = arena.accept(left, self);
        let right_cost = arena.accept(right, self);
        *self.costs.get(&SymNode::Add(0, 0)).unwrap_or(&1) + left_cost + right_cost
    }

    fn visit_sub(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> usize {
        let left_cost = arena.accept(left, self);
        let right_cost = arena.accept(right, self);
        *self.costs.get(&SymNode::Sub(0, 0)).unwrap_or(&1) + left_cost + right_cost
    }

    fn visit_mul(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> usize {
        let left_cost = arena.accept(left, self);
        let right_cost = arena.accept(right, self);
        *self.costs.get(&SymNode::Mul(0, 0)).unwrap_or(&2) + left_cost + right_cost
    }

    fn visit_div(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> usize {
        let left_cost = arena.accept(left, self);
        let right_cost = arena.accept(right, self);
        *self.costs.get(&SymNode::Div(0, 0)).unwrap_or(&3) + left_cost + right_cost
    }

    fn visit_powi(&mut self, base: NodeId, exp: i32, arena: &SymArena) -> usize {
        let base_cost = arena.accept(base, self);
        *self.costs.get(&SymNode::Powi(0, 0)).unwrap_or(&3) + base_cost
    }

    fn visit_neg(&mut self, operand: NodeId, arena: &SymArena) -> usize {
        let operand_cost = arena.accept(operand, self);
        *self.costs.get(&SymNode::Neg(0)).unwrap_or(&1) + operand_cost
    }

    fn visit_sin(&mut self, operand: NodeId, arena: &SymArena) -> usize {
        let operand_cost = arena.accept(operand, self);
        *self.costs.get(&SymNode::Sin(0)).unwrap_or(&4) + operand_cost
    }

    fn visit_cos(&mut self, operand: NodeId, arena: &SymArena) -> usize {
        let operand_cost = arena.accept(operand, self);
        *self.costs.get(&SymNode::Cos(0)).unwrap_or(&4) + operand_cost
    }

    fn visit_ln(&mut self, operand: NodeId, arena: &SymArena) -> usize {
        let operand_cost = arena.accept(operand, self);
        *self.costs.get(&SymNode::Ln(0)).unwrap_or(&4) + operand_cost
    }

    fn visit_exp(&mut self, operand: NodeId, arena: &SymArena) -> usize {
        let operand_cost = arena.accept(operand, self);
        *self.costs.get(&SymNode::Exp(0)).unwrap_or(&4) + operand_cost
    }

    fn visit_sqrt(&mut self, operand: NodeId, arena: &SymArena) -> usize {
        let operand_cost = arena.accept(operand, self);
        *self.costs.get(&SymNode::Sqrt(0)).unwrap_or(&4) + operand_cost
    }
}

/// A [`SymVisitor`] that counts how many times each [`NodeId`] is referenced
/// in the sub-tree rooted at the visited node.
///
/// The resulting counts are used by [`ToTokenStreamVisitor`] to decide which
/// sub-expressions are worth hoisting into a `let` binding for common
/// sub-expression elimination (CSE).
struct RefCountVisitor<'a> {
    arena: &'a SymArena,
    counts: HashMap<NodeId, usize>,
}

impl RefCountVisitor<'_> {
    pub fn new(arena: &'_ SymArena) -> RefCountVisitor<'_> {
        RefCountVisitor {
            arena,
            counts: HashMap::new(),
        }
    }

    /// Return the reference-count map populated during the walk.
    pub fn get_counts(&self) -> &HashMap<NodeId, usize> {
        &self.counts
    }
}

impl SymVisitor<()> for RefCountVisitor<'_> {
    fn visit_const(&mut self, value: u64, arena: &SymArena) -> () {}

    fn visit_var(&mut self, idx: NodeId, arena: &SymArena) -> () {
        self.counts.entry(idx).and_modify(|c| *c += 1).or_insert(1);
    }

    fn visit_add(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> () {
        arena.accept(left, self);
        arena.accept(right, self);
        self.counts.entry(left).and_modify(|c| *c += 1).or_insert(1);
        self.counts
            .entry(right)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    fn visit_sub(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> () {
        arena.accept(left, self);
        arena.accept(right, self);
        self.counts.entry(left).and_modify(|c| *c += 1).or_insert(1);
        self.counts
            .entry(right)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    fn visit_mul(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> () {
        arena.accept(left, self);
        arena.accept(right, self);
        self.counts.entry(left).and_modify(|c| *c += 1).or_insert(1);
        self.counts
            .entry(right)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    fn visit_div(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> () {
        arena.accept(left, self);
        arena.accept(right, self);
        self.counts.entry(left).and_modify(|c| *c += 1).or_insert(1);
        self.counts
            .entry(right)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    fn visit_powi(&mut self, base: NodeId, exp: i32, arena: &SymArena) -> () {
        arena.accept(base, self);
        self.counts.entry(base).and_modify(|c| *c += 1).or_insert(1);
    }

    fn visit_neg(&mut self, operand: NodeId, arena: &SymArena) -> () {
        arena.accept(operand, self);
        self.counts
            .entry(operand)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    fn visit_sin(&mut self, operand: NodeId, arena: &SymArena) -> () {
        arena.accept(operand, self);
        self.counts
            .entry(operand)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    fn visit_cos(&mut self, operand: NodeId, arena: &SymArena) -> () {
        arena.accept(operand, self);
        self.counts
            .entry(operand)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    fn visit_ln(&mut self, operand: NodeId, arena: &SymArena) -> () {
        arena.accept(operand, self);
        self.counts
            .entry(operand)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    fn visit_exp(&mut self, operand: NodeId, arena: &SymArena) -> () {
        arena.accept(operand, self);
        self.counts
            .entry(operand)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    fn visit_sqrt(&mut self, operand: NodeId, arena: &SymArena) -> () {
        arena.accept(operand, self);
        self.counts
            .entry(operand)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }
}

/// A [`SymVisitor`] that emits a `proc_macro2::TokenStream` for an expression
/// tree, performing simple common sub-expression elimination (CSE).
///
/// Nodes whose reference count (from [`RefCountVisitor`]) exceeds 1 are hoisted
/// into `let tmpN = …;` bindings stored in `instructions`, and subsequent uses
/// of the same node emit a reference to that temporary rather than recomputing
/// the expression.
struct ToTokenStreamVisitor<'a> {
    arena: &'a SymArena,
    /// Per-node reference counts used to decide when CSE is worthwhile.
    counts: &'a HashMap<NodeId, usize>,
    /// Cache mapping a `NodeId` to the token stream that represents it (either
    /// the full expression or a reference to the hoisted temporary).
    cache: HashMap<NodeId, TokenStream>,
    /// Accumulated `let` bindings for hoisted sub-expressions, in emission order.
    instructions: Vec<TokenStream>,
}

impl<'a> ToTokenStreamVisitor<'a> {
    pub fn new(
        arena: &'a SymArena,
        counts: &'a HashMap<NodeId, usize>,
    ) -> ToTokenStreamVisitor<'a> {
        ToTokenStreamVisitor {
            arena,
            counts,
            cache: HashMap::new(),
            instructions: Vec::new(),
        }
    }

    /// Given the tokens for `node_id`, either return them directly (if the node
    /// is only used once) or hoist them into a `let` binding and return a
    /// reference to the temporary (if the node is used more than once).
    pub fn write_token(&mut self, node_id: NodeId, tokens: TokenStream) -> TokenStream {
        // Check if the value is already cached
        if let Some(token) = self.cache.get(&node_id) {
            return token.clone();
        }

        // Check if multiple references to this node exist and if so, store the instruction in a temporary variable and cache resul
        if *self.counts.get(&node_id).unwrap_or(&0) > 1 {
            let temp_var = format!("tmp{}", node_id);
            let temp_token = quote! { let #temp_var = #tokens; };
            self.instructions.push(temp_token);
            let temp_token_stream = quote! { #temp_var };
            self.cache.insert(node_id, temp_token_stream.clone());
            return temp_token_stream;
        }

        tokens
    }
}

impl SymVisitor<TokenStream> for ToTokenStreamVisitor<'_> {
    fn visit_const(&mut self, value: u64, arena: &SymArena) -> TokenStream {
        let f = f64::from_bits(value);
        quote! { #f }
    }

    fn visit_var(&mut self, idx: NodeId, arena: &SymArena) -> TokenStream {
        let var_name = Ident::new("x", proc_macro2::Span::call_site());
        quote! { #var_name[#idx] }
    }

    fn visit_add(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> TokenStream {
        let left_tokens = arena.accept(left, self);
        let right_tokens = arena.accept(right, self);
        quote! { (#left_tokens + #right_tokens) }
    }

    fn visit_sub(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> TokenStream {
        let left_tokens = arena.accept(left, self);
        let right_tokens = arena.accept(right, self);
        quote! { (#left_tokens - #right_tokens) }
    }

    fn visit_mul(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> TokenStream {
        let left_tokens = arena.accept(left, self);
        let right_tokens = arena.accept(right, self);
        quote! { (#left_tokens * #right_tokens) }
    }

    fn visit_div(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> TokenStream {
        let left_tokens = arena.accept(left, self);
        let right_tokens = arena.accept(right, self);
        quote! { (#left_tokens / #right_tokens) }
    }

    fn visit_powi(&mut self, base: NodeId, exp: i32, arena: &SymArena) -> TokenStream {
        let base_tokens = arena.accept(base, self);
        quote! { (#base_tokens.powi(#exp)) }
    }

    fn visit_neg(&mut self, operand: NodeId, arena: &SymArena) -> TokenStream {
        let operand_tokens = arena.accept(operand, self);
        quote! { (-#operand_tokens) }
    }

    fn visit_sin(&mut self, operand: NodeId, arena: &SymArena) -> TokenStream {
        let operand_tokens = arena.accept(operand, self);
        quote! { (#operand_tokens.sin()) }
    }

    fn visit_cos(&mut self, operand: NodeId, arena: &SymArena) -> TokenStream {
        let operand_tokens = arena.accept(operand, self);
        quote! { (#operand_tokens.cos()) }
    }

    fn visit_ln(&mut self, operand: NodeId, arena: &SymArena) -> TokenStream {
        let operand_tokens = arena.accept(operand, self);
        quote! { (#operand_tokens.ln()) }
    }

    fn visit_exp(&mut self, operand: NodeId, arena: &SymArena) -> TokenStream {
        let operand_tokens = arena.accept(operand, self);
        quote! { (#operand_tokens.exp()) }
    }

    fn visit_sqrt(&mut self, operand: NodeId, arena: &SymArena) -> TokenStream {
        let operand_tokens = arena.accept(operand, self);
        quote! { (#operand_tokens.sqrt()) }
    }
}

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
pub fn parse_syn(arena: &mut SymArena, expr: &Expr) -> NodeId {
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

        Expr::Binary(bin_expr) => {
            let left_id = parse_syn(arena, &bin_expr.left);
            let right_id = parse_syn(arena, &bin_expr.right);
            return match bin_expr.op {
                syn::BinOp::Add(_) => arena.intern(SymNode::Add(left_id, right_id)),
                syn::BinOp::Sub(_) => arena.intern(SymNode::Sub(left_id, right_id)),
                syn::BinOp::Mul(_) => arena.intern(SymNode::Mul(left_id, right_id)),
                syn::BinOp::Div(_) => arena.intern(SymNode::Div(left_id, right_id)),
                _ => panic!("Unsupported binary operator"),
            };
        }

        Expr::Unary(un) => {
            let operand_id = parse_syn(arena, &un.expr);
            return match un.op {
                syn::UnOp::Neg(_) => arena.intern(SymNode::Neg(operand_id)),
                _ => panic!("Unsupported unary operator"),
            };
        }

        Expr::MethodCall(call) => {
            let receiver_id = parse_syn(arena, &call.receiver);
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

        Expr::Paren(paren) => parse_syn(arena, &paren.expr),
        Expr::Group(group) => parse_syn(arena, &group.expr),
        Expr::Cast(c) => parse_syn(arena, &c.expr),

        _ => panic!("Unsupported expression type {:?}", expr),
    }
}

/// Differentiate `root_id` symbolically with respect to `var_idx`, simplify
/// the result, and emit it as a `TokenStream`.
///
/// The emitted tokens represent a single `f64` expression (no surrounding
/// function definition).  Common sub-expressions that appear more than once are
/// hoisted into `let` bindings by [`ToTokenStreamVisitor`].
pub fn compile_expression(
    arena: &mut SymArena,
    root_id: NodeId,
    var_idx: usize,
    max_passes: usize,
) -> TokenStream {
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
    let mut root_id = arena.transform(root_id, &diff_transformer, &mut HashMap::new());

    // Simplify the result
    let simplify_transformer = SimplifyTransformer::new();
    for _ in 0..max_passes {
        let new_root_id = arena.transform(root_id, &simplify_transformer, &mut HashMap::new());
        if new_root_id == root_id {
            break; // No further simplification possible
        }
        root_id = new_root_id;
    }

    // Reference counting for common sub-expression elimination
    let mut ref_count_visitor = RefCountVisitor::new(arena);
    arena.accept(root_id, &mut ref_count_visitor);
    let ref_counts = ref_count_visitor.get_counts();

    let mut to_tokens_visitor = ToTokenStreamVisitor::new(arena, &ref_counts);
    arena.accept(root_id, &mut to_tokens_visitor)
}
