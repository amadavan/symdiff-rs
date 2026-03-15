use std::collections::HashMap;

use crate::arena::{NodeId, SymArena, SymNode, SymTransformer};

/// Computes the symbolic derivative of every node with respect to one variable.
///
/// Applies standard calculus rules: sum, product, quotient, power, and chain
/// rule for each transcendental.
pub struct DiffTransformer {
    /// Index of the variable to differentiate with respect to.
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
        arena.intern(SymNode::Const(0))
    }

    fn process_var(
        &self,
        idx: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        if idx == self.var {
            arena.intern(SymNode::Const(1.0_f64.to_bits()))
        } else {
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

/// Algebraically simplifies an expression tree via local rewrite rules.
///
/// Rules applied (where applicable): constant folding, identity/annihilator
/// laws (`0+e`, `1*e`, `0*e`, etc.), double negation, distributive factoring
/// (`a*b ± a*c = a*(b±c)`), power merging (`b^m * b^n = b^(m+n)`),
/// exponential merging (`exp(a)*exp(b) = exp(a+b)`), `ln(b^n) = n*ln(b)`,
/// and `sqrt(b^(2k)) = b^k`.
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
        arena.intern(SymNode::Const(value))
    }

    fn process_var(
        &self,
        idx: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Var(idx))
    }

    fn process_add(
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
                let result_value = (f64::from_bits(value1) + f64::from_bits(value2)).to_bits();
                return arena.intern(SymNode::Const(result_value));
            }
            (SymNode::Const(v), _) if v == 0.0_f64.to_bits() => return right_id,
            (_, SymNode::Const(v)) if v == 0.0_f64.to_bits() => return left_id,
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1a == e2a => {
                let factored = arena.intern(SymNode::Add(e1b, e2b));
                return arena.intern(SymNode::Mul(e1a, factored));
            }
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1a == e2b => {
                let factored = arena.intern(SymNode::Add(e1b, e2a));
                return arena.intern(SymNode::Mul(e1a, factored));
            }
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1b == e2a => {
                let factored = arena.intern(SymNode::Add(e1a, e2b));
                return arena.intern(SymNode::Mul(e1b, factored));
            }
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1b == e2b => {
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
                let result_value = (f64::from_bits(value1) - f64::from_bits(value2)).to_bits();
                return arena.intern(SymNode::Const(result_value));
            }
            (SymNode::Const(v), _) if v == 0.0_f64.to_bits() => {
                return arena.intern(SymNode::Neg(right_id));
            }
            (_, SymNode::Const(v)) if v == 0.0_f64.to_bits() => return left_id,
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1a == e2a => {
                let factored = arena.intern(SymNode::Sub(e1b, e2b));
                return arena.intern(SymNode::Mul(e1a, factored));
            }
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1a == e2b => {
                let factored = arena.intern(SymNode::Sub(e1b, e2a));
                return arena.intern(SymNode::Mul(e1a, factored));
            }
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1b == e2a => {
                let factored = arena.intern(SymNode::Sub(e1a, e2b));
                return arena.intern(SymNode::Mul(e1b, factored));
            }
            (SymNode::Mul(e1a, e1b), SymNode::Mul(e2a, e2b)) if e1b == e2b => {
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
                let result_value = (f64::from_bits(value1) * f64::from_bits(value2)).to_bits();
                return arena.intern(SymNode::Const(result_value));
            }
            (SymNode::Const(v), _) | (_, SymNode::Const(v)) if v == 0.0_f64.to_bits() => {
                return arena.intern(SymNode::Const(0.0_f64.to_bits()));
            }
            (SymNode::Const(v), _) if v == 1.0_f64.to_bits() => return right_id,
            (_, SymNode::Const(v)) if v == 1.0_f64.to_bits() => return left_id,
            (SymNode::Powi(base1, exp1), SymNode::Powi(base2, exp2)) if base1 == base2 => {
                return arena.intern(SymNode::Powi(base1, exp1 + exp2));
            }
            (SymNode::Exp(e1), SymNode::Exp(e2)) => {
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
                let result_value = (f64::from_bits(value1) / f64::from_bits(value2)).to_bits();
                return arena.intern(SymNode::Const(result_value));
            }
            (SymNode::Const(v), _) if v == 0.0_f64.to_bits() => {
                return arena.intern(SymNode::Const(0.0_f64.to_bits()));
            }
            (_, SymNode::Const(v)) if v == 1.0_f64.to_bits() => return left_id,
            (SymNode::Exp(e1), SymNode::Exp(e2)) => {
                let new_exp = arena.intern(SymNode::Sub(e1, e2));
                return arena.intern(SymNode::Exp(new_exp));
            }
            (SymNode::Powi(base1, exp1), SymNode::Powi(base2, exp2)) if base1 == base2 => {
                return arena.intern(SymNode::Powi(base1, exp1 - exp2));
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
                return arena.intern(SymNode::Const(f64::from_bits(value).powi(exp).to_bits()));
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
                return arena.intern(SymNode::Const((-f64::from_bits(value)).to_bits()));
            }
            SymNode::Neg(inner) => return inner,
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
                return arena.intern(SymNode::Const(f64::from_bits(value).sin().to_bits()));
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
                return arena.intern(SymNode::Const(f64::from_bits(value).cos().to_bits()));
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
                let c = f64::from_bits(value);
                if c > 0.0 {
                    return arena.intern(SymNode::Const(c.ln().to_bits()));
                }
            }
            SymNode::Powi(base, exp) => {
                // ln(b^n) = n * ln(b)
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
                return arena.intern(SymNode::Const(f64::from_bits(value).exp().to_bits()));
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
                let c = f64::from_bits(value);
                if c >= 0.0 {
                    return arena.intern(SymNode::Const(c.sqrt().to_bits()));
                }
            }
            SymNode::Powi(base, exp) if exp % 2 == 0 => {
                return arena.intern(SymNode::Powi(base, exp / 2));
            }
            _ => {}
        }
        arena.intern(SymNode::Sqrt(operand_id))
    }
}

/// Swaps operands of commutative operations (`+`, `*`) and rewrites division
/// chains to expose more opportunities for common sub-expression elimination.
pub struct CommutativeTransformer {}

impl CommutativeTransformer {
    pub fn new() -> CommutativeTransformer {
        CommutativeTransformer {}
    }
}

impl SymTransformer for CommutativeTransformer {
    fn process_const(
        &self,
        value: u64,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Const(value))
    }

    fn process_var(
        &self,
        idx: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Var(idx))
    }

    fn process_add(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_node = *arena.get_node(left);
        let right_node = *arena.get_node(right);
        match (left_node, right_node) {
            (SymNode::Const(_), _)
            | (_, SymNode::Const(_))
            | (SymNode::Var(_), _)
            | (_, SymNode::Var(_))
            | (SymNode::Add(_, _), _)
            | (_, SymNode::Add(_, _))
            | (SymNode::Sub(_, _), _)
            | (_, SymNode::Sub(_, _)) => {
                return arena.intern(SymNode::Add(right, left));
            }
            _ => arena.intern(SymNode::Add(left, right)),
        }
    }

    fn process_sub(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_node = *arena.get_node(left);
        let right_node = *arena.get_node(right);
        match (left_node, right_node) {
            (SymNode::Const(_), _)
            | (_, SymNode::Const(_))
            | (SymNode::Var(_), _)
            | (_, SymNode::Var(_))
            | (SymNode::Add(_, _), _)
            | (_, SymNode::Add(_, _))
            | (SymNode::Sub(_, _), _)
            | (_, SymNode::Sub(_, _)) => {
                let new_right = arena.intern(SymNode::Neg(right));
                return arena.intern(SymNode::Add(new_right, left));
            }
            _ => arena.intern(SymNode::Sub(left, right)),
        }
    }

    fn process_mul(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_node = *arena.get_node(left);
        let right_node = *arena.get_node(right);
        match (left_node, right_node) {
            (SymNode::Const(_), _)
            | (_, SymNode::Const(_))
            | (SymNode::Var(_), _)
            | (_, SymNode::Var(_))
            | (SymNode::Mul(_, _), _)
            | (_, SymNode::Mul(_, _))
            | (SymNode::Div(_, _), _)
            | (_, SymNode::Div(_, _)) => {
                return arena.intern(SymNode::Mul(right, left));
            }
            _ => arena.intern(SymNode::Mul(left, right)),
        }
    }

    fn process_div(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_node = *arena.get_node(left);
        let right_node = *arena.get_node(right);
        match (left_node, right_node) {
            (SymNode::Mul(l1, l2), _) => {
                let new_left = arena.intern(SymNode::Div(l1, right));
                return arena.intern(SymNode::Mul(new_left, l2));
            }
            (_, SymNode::Mul(r1, _)) => {
                let new_right = arena.intern(SymNode::Div(right, r1));
                return arena.intern(SymNode::Div(left, new_right));
            }
            (SymNode::Div(l1, l2), _) => {
                let new_right = arena.intern(SymNode::Mul(l2, right));
                return arena.intern(SymNode::Div(l1, new_right));
            }
            (_, SymNode::Div(r1, r2)) => {
                let new_left = arena.intern(SymNode::Mul(left, r2));
                return arena.intern(SymNode::Div(new_left, r1));
            }
            _ => arena.intern(SymNode::Div(left, right)),
        }
    }

    fn process_powi(
        &self,
        base: NodeId,
        exp: i32,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Powi(base, exp))
    }

    fn process_neg(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Neg(operand))
    }

    fn process_sin(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Sin(operand))
    }

    fn process_cos(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Cos(operand))
    }

    fn process_ln(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Ln(operand))
    }

    fn process_exp(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Exp(operand))
    }

    fn process_sqrt(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Sqrt(operand))
    }
}

/// Re-brackets associative operations (`+`, `-`, `*`, `/`) to expose more
/// opportunities for common sub-expression elimination.
pub struct AssociativeTransformer {}

impl AssociativeTransformer {
    pub fn new() -> AssociativeTransformer {
        AssociativeTransformer {}
    }
}

impl SymTransformer for AssociativeTransformer {
    fn process_const(
        &self,
        value: u64,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Const(value))
    }

    fn process_var(
        &self,
        idx: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Var(idx))
    }

    fn process_add(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_node = *arena.get_node(left);
        let right_node = *arena.get_node(right);

        match (left_node, right_node) {
            (SymNode::Add(l1, l2), _) => {
                let new_right = arena.intern(SymNode::Add(l2, right));
                return arena.intern(SymNode::Add(l1, new_right));
            }
            (_, SymNode::Add(r1, r2)) => {
                let new_left = arena.intern(SymNode::Add(left, r1));
                return arena.intern(SymNode::Add(new_left, r2));
            }
            (SymNode::Sub(l1, l2), _) => {
                let new_right = arena.intern(SymNode::Sub(l2, right));
                return arena.intern(SymNode::Sub(l1, new_right));
            }
            (_, SymNode::Sub(r1, r2)) => {
                let new_left = arena.intern(SymNode::Add(left, r1));
                return arena.intern(SymNode::Sub(new_left, r2));
            }
            _ => {}
        }

        arena.intern(SymNode::Add(left, right))
    }

    fn process_sub(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_node = *arena.get_node(left);
        let right_node = *arena.get_node(right);

        match (left_node, right_node) {
            (SymNode::Add(l1, l2), _) => {
                let new_right = arena.intern(SymNode::Sub(l2, right));
                return arena.intern(SymNode::Add(l1, new_right));
            }
            (_, SymNode::Add(r1, r2)) => {
                let new_left = arena.intern(SymNode::Sub(left, r1));
                return arena.intern(SymNode::Sub(new_left, r2));
            }
            (SymNode::Sub(l1, l2), _) => {
                let new_right = arena.intern(SymNode::Add(l2, right));
                return arena.intern(SymNode::Sub(l1, new_right));
            }
            (_, SymNode::Sub(r1, r2)) => {
                let new_left = arena.intern(SymNode::Sub(left, r1));
                return arena.intern(SymNode::Add(new_left, r2));
            }
            _ => arena.intern(SymNode::Sub(left, right)),
        }
    }

    fn process_mul(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_node = *arena.get_node(left);
        let right_node = *arena.get_node(right);

        match (left_node, right_node) {
            (SymNode::Mul(l1, l2), _) => {
                let new_right = arena.intern(SymNode::Mul(l2, right));
                return arena.intern(SymNode::Mul(l1, new_right));
            }
            (_, SymNode::Mul(r1, r2)) => {
                let new_left = arena.intern(SymNode::Mul(left, r1));
                return arena.intern(SymNode::Mul(new_left, r2));
            }
            (SymNode::Div(l1, l2), _) => {
                let new_right = arena.intern(SymNode::Div(right, l2));
                return arena.intern(SymNode::Mul(l1, new_right));
            }
            (_, SymNode::Div(r1, r2)) => {
                let new_left = arena.intern(SymNode::Mul(left, r1));
                return arena.intern(SymNode::Div(new_left, r2));
            }
            _ => arena.intern(SymNode::Mul(left, right)),
        }
    }

    fn process_div(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_node = *arena.get_node(left);
        let right_node = *arena.get_node(right);

        match (left_node, right_node) {
            (SymNode::Mul(l1, l2), _) => {
                let new_right = arena.intern(SymNode::Div(l2, right));
                return arena.intern(SymNode::Mul(l1, new_right));
            }
            (_, SymNode::Mul(r1, r2)) => {
                let new_left = arena.intern(SymNode::Div(left, r1));
                return arena.intern(SymNode::Div(new_left, r2));
            }
            (SymNode::Div(l1, l2), _) => {
                let new_right = arena.intern(SymNode::Mul(l2, right));
                return arena.intern(SymNode::Div(l1, new_right));
            }
            (_, SymNode::Div(r1, r2)) => {
                let new_left = arena.intern(SymNode::Mul(left, r2));
                return arena.intern(SymNode::Div(new_left, r1));
            }
            _ => arena.intern(SymNode::Div(left, right)),
        }
    }

    fn process_powi(
        &self,
        base: NodeId,
        exp: i32,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Powi(base, exp))
    }

    fn process_neg(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let operand_node = *arena.get_node(operand);

        match operand_node {
            SymNode::Neg(inner) => return inner,
            SymNode::Const(value) => {
                return arena.intern(SymNode::Const((-f64::from_bits(value)).to_bits()));
            }
            SymNode::Sub(l1, l2) => return arena.intern(SymNode::Sub(l2, l1)),
            _ => arena.intern(SymNode::Neg(operand)),
        }
    }

    fn process_sin(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Sin(operand))
    }

    fn process_cos(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Cos(operand))
    }

    fn process_ln(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Ln(operand))
    }

    fn process_exp(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Exp(operand))
    }

    fn process_sqrt(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Sqrt(operand))
    }
}

/// Replaces node ids according to a fixed mapping. Used by [`GreedyCoordinator`]
/// to propagate rewritten ids back through the tree after each optimization step.
pub struct RemapTransformer<'a> {
    mapping: &'a HashMap<NodeId, NodeId>,
}

impl<'a> RemapTransformer<'a> {
    pub fn new(mapping: &'a HashMap<NodeId, NodeId>) -> RemapTransformer<'a> {
        RemapTransformer { mapping }
    }
}

impl<'a> SymTransformer for RemapTransformer<'a> {
    fn process_const(
        &self,
        value: u64,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        arena.intern(SymNode::Const(value))
    }

    fn process_var(
        &self,
        idx: NodeId,
        arena: &mut SymArena,
        _diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        if let Some(&new_idx) = self.mapping.get(&idx) {
            arena.intern(SymNode::Var(new_idx))
        } else {
            arena.intern(SymNode::Var(idx))
        }
    }

    fn process_add(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_id = *self.mapping.get(&left).unwrap_or(&diff[&left]);
        let right_id = *self.mapping.get(&right).unwrap_or(&diff[&right]);
        arena.intern(SymNode::Add(left_id, right_id))
    }

    fn process_sub(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_id = *self.mapping.get(&left).unwrap_or(&diff[&left]);
        let right_id = *self.mapping.get(&right).unwrap_or(&diff[&right]);
        arena.intern(SymNode::Sub(left_id, right_id))
    }

    fn process_mul(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_id = *self.mapping.get(&left).unwrap_or(&diff[&left]);
        let right_id = *self.mapping.get(&right).unwrap_or(&diff[&right]);
        arena.intern(SymNode::Mul(left_id, right_id))
    }

    fn process_div(
        &self,
        left: NodeId,
        right: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let left_id = *self.mapping.get(&left).unwrap_or(&diff[&left]);
        let right_id = *self.mapping.get(&right).unwrap_or(&diff[&right]);
        arena.intern(SymNode::Div(left_id, right_id))
    }

    fn process_powi(
        &self,
        base: NodeId,
        exp: i32,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let base_id = *self.mapping.get(&base).unwrap_or(&diff[&base]);
        arena.intern(SymNode::Powi(base_id, exp))
    }

    fn process_neg(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let operand_id = *self.mapping.get(&operand).unwrap_or(&diff[&operand]);
        arena.intern(SymNode::Neg(operand_id))
    }

    fn process_sin(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let operand_id = *self.mapping.get(&operand).unwrap_or(&diff[&operand]);
        arena.intern(SymNode::Sin(operand_id))
    }

    fn process_cos(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let operand_id = *self.mapping.get(&operand).unwrap_or(&diff[&operand]);
        arena.intern(SymNode::Cos(operand_id))
    }

    fn process_ln(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let operand_id = *self.mapping.get(&operand).unwrap_or(&diff[&operand]);
        arena.intern(SymNode::Ln(operand_id))
    }

    fn process_exp(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let operand_id = *self.mapping.get(&operand).unwrap_or(&diff[&operand]);
        arena.intern(SymNode::Exp(operand_id))
    }

    fn process_sqrt(
        &self,
        operand: NodeId,
        arena: &mut SymArena,
        diff: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        let operand_id = *self.mapping.get(&operand).unwrap_or(&diff[&operand]);
        arena.intern(SymNode::Sqrt(operand_id))
    }
}
