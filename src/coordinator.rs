use std::collections::HashMap;

use crate::{
    SimplifyTransformer,
    arena::{NodeId, SymArena, SymNode},
    transformers::{AssociativeTransformer, CommutativeTransformer, RemapTransformer},
    visitors::CostEstimateVisitor,
};

pub trait Coordinator {
    fn optimize(&self, root: NodeId, arena: &mut SymArena) -> NodeId;
}

pub struct GreedyCoordinator<'a> {
    costs: &'a HashMap<SymNode, usize>,
}

impl<'a> GreedyCoordinator<'a> {
    pub fn new(costs: &'a HashMap<SymNode, usize>) -> GreedyCoordinator<'a> {
        GreedyCoordinator { costs }
    }
}

impl<'a> Coordinator for GreedyCoordinator<'a> {
    fn optimize(&self, root: NodeId, arena: &mut SymArena) -> NodeId {
        let order = arena.get_topological_order(root);
        let simplify_transformer = SimplifyTransformer::new();
        let commutative_transformer = CommutativeTransformer::new();
        let associative_transformer = AssociativeTransformer::new();
        let mut cost_visitor = CostEstimateVisitor::new(self.costs);

        let mut remapping = HashMap::new();

        for old_id in order {
            let mut trial_arena = arena.clone();
            let mut candidates = vec![(root, &mut trial_arena)];

            // Consider commutation
            let mut comm_arena = arena.clone();
            let comm_root = comm_arena.transform_single_node(old_id, &commutative_transformer);
            candidates.push((comm_root, &mut comm_arena));

            // Consider association
            let mut assoc_arena = arena.clone();
            let assoc_root = assoc_arena.transform_single_node(old_id, &associative_transformer);
            candidates.push((assoc_root, &mut assoc_arena));

            // Consider commutation of the association
            let mut comm_assoc_arena = arena.clone();
            let comm_assoc_root =
                comm_assoc_arena.transform_single_node(assoc_root, &commutative_transformer);
            candidates.push((comm_assoc_root, &mut comm_assoc_arena));

            // Consider association of the commutation
            let mut assoc_comm_arena = arena.clone();
            let assoc_comm_root =
                assoc_comm_arena.transform_single_node(comm_root, &associative_transformer);
            candidates.push((assoc_comm_root, &mut assoc_comm_arena));

            let top_candidate = candidates
                .into_iter()
                .map(|(candidate, arena)| {
                    (arena.transform(candidate, &simplify_transformer), arena)
                })
                .min_by_key(|(candidate, arena)| arena.accept(*candidate, &mut cost_visitor))
                .unwrap();

            // Add corresponding trial arena to arena
            *arena = top_candidate.1.clone();

            remapping.insert(old_id, top_candidate.0);
            let remap_transformer = RemapTransformer::new(&remapping);

            // Transform the lower ids to the new ids
            arena.transform(old_id, &remap_transformer);
        }

        root
    }
}
