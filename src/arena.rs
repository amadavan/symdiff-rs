use std::collections::{HashMap, HashSet};

pub type NodeId = usize;

/// A node in the symbolic expression tree.
///
/// Children are [`NodeId`]s into the owning [`SymArena`].
/// Constants are stored as raw `f64` bit patterns (`f64::to_bits`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SymNode {
    /// Numeric constant; payload is `value.to_bits()`.
    Const(u64),
    /// Input variable `x[idx]`.
    Var(NodeId),
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Div(NodeId, NodeId),
    /// `base.powi(exp)` — exponent is a literal `i32`, not a node.
    Powi(NodeId, i32),
    Neg(NodeId),
    Sin(NodeId),
    Cos(NodeId),
    Ln(NodeId),
    Exp(NodeId),
    Sqrt(NodeId),
}

/// Interning arena for [`SymNode`]s.
///
/// Nodes live in a flat `Vec` indexed by [`NodeId`]. Equal nodes share a single
/// id — see [`intern`](SymArena::intern).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymArena {
    nodes: Vec<SymNode>,
    lookup: HashMap<SymNode, NodeId>,
}

impl SymArena {
    /// Create an empty arena.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            lookup: HashMap::new(),
        }
    }

    pub fn get_node(&self, node_id: NodeId) -> &SymNode {
        &self.nodes[node_id]
    }

    /// Look up the id of an already-interned node, if present.
    pub fn get_id(&self, sym: &SymNode) -> Option<NodeId> {
        self.lookup.get(sym).copied()
    }

    /// Dispatch a single node to the matching visitor method.
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

    /// Insert `node`, returning its id. Returns the existing id if already present.
    pub fn intern(&mut self, node: SymNode) -> NodeId {
        if let Some(&idx) = self.lookup.get(&node) {
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(node.clone());
            self.lookup.insert(node, idx);
            idx
        }
    }

    /// Apply `transformer` to `node_id` in isolation, without recursing into children.
    pub fn transform_single_node<T: SymTransformer>(
        &mut self,
        node_id: NodeId,
        transformer: &T,
    ) -> NodeId {
        let node = self.nodes[node_id];
        match node {
            SymNode::Const(value) => transformer.process_const(value, self, &mut HashMap::new()),
            SymNode::Var(idx) => transformer.process_var(idx, self, &mut HashMap::new()),
            SymNode::Add(left, right) => {
                transformer.process_add(left, right, self, &mut HashMap::new())
            }
            SymNode::Sub(left, right) => {
                transformer.process_sub(left, right, self, &mut HashMap::new())
            }
            SymNode::Mul(left, right) => {
                transformer.process_mul(left, right, self, &mut HashMap::new())
            }
            SymNode::Div(left, right) => {
                transformer.process_div(left, right, self, &mut HashMap::new())
            }
            SymNode::Powi(base, exp) => {
                transformer.process_powi(base, exp, self, &mut HashMap::new())
            }
            SymNode::Neg(operand) => transformer.process_neg(operand, self, &mut HashMap::new()),
            SymNode::Sin(operand) => transformer.process_sin(operand, self, &mut HashMap::new()),
            SymNode::Cos(operand) => transformer.process_cos(operand, self, &mut HashMap::new()),
            SymNode::Ln(operand) => transformer.process_ln(operand, self, &mut HashMap::new()),
            SymNode::Exp(operand) => transformer.process_exp(operand, self, &mut HashMap::new()),
            SymNode::Sqrt(operand) => transformer.process_sqrt(operand, self, &mut HashMap::new()),
        }
    }

    /// Apply `transformer` to every node reachable from `root` in bottom-up
    /// topological order, returning the transformed id for `root`.
    pub fn transform<T: SymTransformer>(&mut self, root: NodeId, transformer: &T) -> NodeId {
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

    /// Nodes reachable from `root_id` in post-order, each appearing once.
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
/// [`SymArena::transform`] calls one method per node bottom-up. The `diff` map
/// holds already-transformed ids for all children; look up `diff[&child]` to
/// get the rewritten child before constructing the new node.
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
/// Unlike [`SymTransformer`], the visitor recurses manually — call
/// `arena.accept(child_id, self)` from within each method as needed.
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
