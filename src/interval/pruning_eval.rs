use std::collections::BTreeMap;

use crate::{
    eval::ValueType,
    tree::{Node, Tree},
};

use super::Interval;

pub struct PruningEvaluator<T>
where
    T: ValueType,
{
    ops: Vec<(Node, usize)>,
    regs: Vec<T>,
    vars: Vec<(char, T)>,
    root_regs: Vec<usize>,
    outputs: Vec<T>,
    bounds: BTreeMap<char, Interval>,
}

impl<T> PruningEvaluator<T>
where
    T: ValueType,
{
    pub fn new(tree: &Tree, bounds: BTreeMap<char, Interval>) {}
}
