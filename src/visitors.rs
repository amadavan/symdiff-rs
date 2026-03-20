use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::quote;

use crate::{
    Options,
    arena::{NodeId, SymArena, SymNode, SymVisitor, VarId},
};

/// Estimates the total evaluation cost of an expression tree by summing
/// per-node costs from a provided table (see `compile_expression` in `lib.rs`).
pub struct CostEstimateVisitor<'a> {
    costs: &'a HashMap<SymNode, usize>,
}

impl<'a> CostEstimateVisitor<'a> {
    pub fn new(costs: &'a HashMap<SymNode, usize>) -> CostEstimateVisitor<'a> {
        CostEstimateVisitor { costs }
    }
}

impl<'a> SymVisitor<usize> for CostEstimateVisitor<'a> {
    fn visit_const(&mut self, _value: u64, _arena: &SymArena) -> usize {
        *self.costs.get(&SymNode::Const(0)).unwrap_or(&0)
    }

    fn visit_var(&mut self, _id: VarId, _idx: NodeId, _arena: &SymArena) -> usize {
        *self.costs.get(&SymNode::Var(0, 0)).unwrap_or(&0)
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
        *self.costs.get(&SymNode::Mul(0, 0)).unwrap_or(&1) + left_cost + right_cost
    }

    fn visit_div(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> usize {
        let left_cost = arena.accept(left, self);
        let right_cost = arena.accept(right, self);
        *self.costs.get(&SymNode::Div(0, 0)).unwrap_or(&15) + left_cost + right_cost
    }

    fn visit_powi(&mut self, base: NodeId, _exp: i32, arena: &SymArena) -> usize {
        let base_cost = arena.accept(base, self);
        *self.costs.get(&SymNode::Powi(0, 0)).unwrap_or(&3) + base_cost
    }

    fn visit_neg(&mut self, operand: NodeId, arena: &SymArena) -> usize {
        let operand_cost = arena.accept(operand, self);
        *self.costs.get(&SymNode::Neg(0)).unwrap_or(&0) + operand_cost
    }

    fn visit_sin(&mut self, operand: NodeId, arena: &SymArena) -> usize {
        let operand_cost = arena.accept(operand, self);
        *self.costs.get(&SymNode::Sin(0)).unwrap_or(&100) + operand_cost
    }

    fn visit_cos(&mut self, operand: NodeId, arena: &SymArena) -> usize {
        let operand_cost = arena.accept(operand, self);
        *self.costs.get(&SymNode::Cos(0)).unwrap_or(&100) + operand_cost
    }

    fn visit_ln(&mut self, operand: NodeId, arena: &SymArena) -> usize {
        let operand_cost = arena.accept(operand, self);
        *self.costs.get(&SymNode::Ln(0)).unwrap_or(&60) + operand_cost
    }

    fn visit_exp(&mut self, operand: NodeId, arena: &SymArena) -> usize {
        let operand_cost = arena.accept(operand, self);
        *self.costs.get(&SymNode::Exp(0)).unwrap_or(&60) + operand_cost
    }

    fn visit_sqrt(&mut self, operand: NodeId, arena: &SymArena) -> usize {
        let operand_cost = arena.accept(operand, self);
        *self.costs.get(&SymNode::Sqrt(0)).unwrap_or(&5) + operand_cost
    }
}

/// Counts how many times each [`NodeId`] is referenced in a subtree.
/// [`ToTokenStreamVisitor`] uses these counts to decide what to hoist into a
/// `let` binding for CSE.
pub struct RefCountVisitor {
    counts: HashMap<NodeId, usize>,
}

impl RefCountVisitor {
    pub fn new() -> RefCountVisitor {
        RefCountVisitor {
            counts: HashMap::new(),
        }
    }

    pub fn get_counts(&self) -> &HashMap<NodeId, usize> {
        &self.counts
    }
}

impl SymVisitor<()> for RefCountVisitor {
    fn visit_const(&mut self, _value: u64, _arena: &SymArena) -> () {}

    // Variables should always be referenced at least once and will be inlined
    fn visit_var(&mut self, _id: VarId, _idx: NodeId, _arena: &SymArena) -> () {}

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

    fn visit_powi(&mut self, base: NodeId, _exp: i32, arena: &SymArena) -> () {
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

/// Emits a `TokenStream` for an expression tree, hoisting nodes referenced
/// more than once into `let tmpN = …;` bindings (CSE).
pub struct ToTokenStreamVisitor<'a> {
    counts: &'a HashMap<NodeId, usize>,
    /// Maps a `NodeId` to its emitted tokens (inline expr or tmp reference).
    cache: &'a mut HashMap<NodeId, TokenStream>,
    /// Accumulated `let` bindings in emission order.
    instructions: &'a mut Vec<TokenStream>,
    options: &'a Options,
}

impl<'a> ToTokenStreamVisitor<'a> {
    pub fn new(
        counts: &'a HashMap<NodeId, usize>,
        cache: &'a mut HashMap<NodeId, TokenStream>,
        instructions: &'a mut Vec<TokenStream>,
        options: &'a Options,
    ) -> ToTokenStreamVisitor<'a> {
        ToTokenStreamVisitor {
            counts,
            cache,
            instructions,
            options,
        }
    }

    /// The accumulated `let` bindings, in the order they must be emitted.
    pub fn get_instructions(&self) -> &[TokenStream] {
        &self.instructions
    }

    /// Returns tokens for `node_id`, hoisting into a `let` tmp if ref count > 1.
    fn write_token(&mut self, node_id: NodeId, tokens: TokenStream) -> TokenStream {
        if let Some(token) = self.cache.get(&node_id) {
            return token.clone();
        }

        if *self.counts.get(&node_id).unwrap_or(&0) > 1 {
            let temp_var =
                syn::Ident::new(&format!("tmp{}", node_id), proc_macro2::Span::call_site());

            let temp_token_stream = quote! { #temp_var };
            // If the instruction has not already been cached, cache it and add a let binding for it.
            if let None = self.cache.insert(node_id, temp_token_stream.clone()) {
                self.instructions.push(quote! { let #temp_var = #tokens; });
            }
            return temp_token_stream;
        }

        tokens
    }
}

impl SymVisitor<TokenStream> for ToTokenStreamVisitor<'_> {
    fn visit_const(&mut self, value: u64, _arena: &SymArena) -> TokenStream {
        let f = f64::from_bits(value);
        quote! { #f }
    }

    fn visit_var(&mut self, id: VarId, idx: NodeId, arena: &SymArena) -> TokenStream {
        let var_ident = arena.get_var_ident(id);
        if self.options.unchecked {
            quote! { (*#var_ident.get_unchecked(#idx)) }
        } else {
            quote! { #var_ident[#idx] }
        }
    }

    fn visit_add(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> TokenStream {
        let left_tokens = arena.accept(left, self);
        let right_tokens = arena.accept(right, self);
        self.write_token(
            arena.get_id(&SymNode::Add(left, right)).unwrap(),
            quote! { (#left_tokens + #right_tokens) },
        )
    }

    fn visit_sub(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> TokenStream {
        let left_tokens = arena.accept(left, self);
        let right_tokens = arena.accept(right, self);
        self.write_token(
            arena.get_id(&SymNode::Sub(left, right)).unwrap(),
            quote! { (#left_tokens - #right_tokens) },
        )
    }

    fn visit_mul(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> TokenStream {
        let left_tokens = arena.accept(left, self);
        let right_tokens = arena.accept(right, self);
        self.write_token(
            arena.get_id(&SymNode::Mul(left, right)).unwrap(),
            quote! { (#left_tokens * #right_tokens) },
        )
    }

    fn visit_div(&mut self, left: NodeId, right: NodeId, arena: &SymArena) -> TokenStream {
        let left_tokens = arena.accept(left, self);
        let right_tokens = arena.accept(right, self);
        self.write_token(
            arena.get_id(&SymNode::Div(left, right)).unwrap(),
            quote! { (#left_tokens / #right_tokens) },
        )
    }

    fn visit_powi(&mut self, base: NodeId, exp: i32, arena: &SymArena) -> TokenStream {
        let base_tokens = arena.accept(base, self);
        self.write_token(
            arena.get_id(&SymNode::Powi(base, exp)).unwrap(),
            quote! { (#base_tokens.powi(#exp)) },
        )
    }

    fn visit_neg(&mut self, operand: NodeId, arena: &SymArena) -> TokenStream {
        let operand_tokens = arena.accept(operand, self);
        self.write_token(
            arena.get_id(&SymNode::Neg(operand)).unwrap(),
            quote! { (-#operand_tokens) },
        )
    }

    fn visit_sin(&mut self, operand: NodeId, arena: &SymArena) -> TokenStream {
        let operand_tokens = arena.accept(operand, self);
        self.write_token(
            arena.get_id(&SymNode::Sin(operand)).unwrap(),
            quote! { (#operand_tokens.sin()) },
        )
    }

    fn visit_cos(&mut self, operand: NodeId, arena: &SymArena) -> TokenStream {
        let operand_tokens = arena.accept(operand, self);
        self.write_token(
            arena.get_id(&SymNode::Cos(operand)).unwrap(),
            quote! { (#operand_tokens.cos()) },
        )
    }

    fn visit_ln(&mut self, operand: NodeId, arena: &SymArena) -> TokenStream {
        let operand_tokens = arena.accept(operand, self);
        self.write_token(
            arena.get_id(&SymNode::Ln(operand)).unwrap(),
            quote! { (#operand_tokens.ln()) },
        )
    }

    fn visit_exp(&mut self, operand: NodeId, arena: &SymArena) -> TokenStream {
        let operand_tokens = arena.accept(operand, self);
        self.write_token(
            arena.get_id(&SymNode::Exp(operand)).unwrap(),
            quote! { (#operand_tokens.exp()) },
        )
    }

    fn visit_sqrt(&mut self, operand: NodeId, arena: &SymArena) -> TokenStream {
        let operand_tokens = arena.accept(operand, self);
        self.write_token(
            arena.get_id(&SymNode::Sqrt(operand)).unwrap(),
            quote! { (#operand_tokens.sqrt()) },
        )
    }
}
