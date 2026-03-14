use std::collections::HashMap;

use crate::arena::{NodeId, SymArena, SymNode, SymTransformer};

/// A [`SymTransformer`] that computes the symbolic derivative of every node
/// with respect to a single variable `var` (identified by its [`NodeId`]).
///
/// The rules applied are the standard calculus identities: sum rule, product
/// rule, quotient rule, power rule, and chain rule for each transcendental.
pub struct DiffTransformer {
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
        _value: u64,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        // Derivative of a constant is zero
        arena.intern(SymNode::Const(0))
    }

    fn process_var(
        &self,
        idx: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
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
pub struct SimplifyTransformer {}

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
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        // Constants are already simplified
        arena.intern(SymNode::Const(value))
    }

    fn process_var(
        &self,
        idx: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
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
        let left_node = *arena.get_node(left_id);
        let right_node = *arena.get_node(right_id);

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
        let left_node = *arena.get_node(left_id);
        let right_node = *arena.get_node(right_id);

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
        let left_node = *arena.get_node(left_id);
        let right_node = *arena.get_node(right_id);

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
        let left_node = *arena.get_node(left_id);
        let right_node = *arena.get_node(right_id);

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
        let base_node = *arena.get_node(base_id);

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
        let operand_node = *arena.get_node(operand_id);

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
        let operand_node = *arena.get_node(operand_id);

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
        let operand_node = *arena.get_node(operand_id);

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
        let operand_node = *arena.get_node(operand_id);

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
        let operand_node = *arena.get_node(operand_id);

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
        let operand_node = *arena.get_node(operand_id);

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
