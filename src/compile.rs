use crate::tree::{Node, Node::*};
use std::ops::Range;

/*
Compile the tree into instructions. In theory one can just walk the nodes of the
// tree and evaluate it without compiling it, but that requires allocating as
// many registers as there are nodes in the tree. Here we compile the tree into
// instructions that reuse registers using Matt Keeter's Solid State Register
// Allocator: https://www.mattkeeter.com/blog/2022-10-04-ssra/
 */

pub struct CompileOutput<'a> {
    pub ops: &'a mut Vec<(Node, usize)>,
    pub out_regs: &'a mut Vec<usize>,
}

#[derive(Default)]
pub struct CompileCache {
    valregs: Vec<Option<usize>>, // Track registers occupied by node values during compilation.
    alive: Vec<bool>,            // Track which registers are in use.
}

pub fn compile(
    nodes: &[Node],
    roots: Range<usize>,
    cache: &mut CompileCache,
    mut out: CompileOutput,
) -> usize {
    let valregs = &mut cache.valregs;
    let alive = &mut cache.alive;
    valregs.clear();
    valregs.resize(nodes.len(), None);
    alive.clear();
    let ops = &mut out.ops;
    let num_ops = ops.len();
    let outregs = &mut out.out_regs;
    let num_outregs = outregs.len();
    // Iterate in reverse.
    for (index, node) in nodes.iter().enumerate().rev() {
        let outreg = get_register(valregs, alive, index);
        if roots.contains(&index) {
            outregs.push(outreg);
        }
        /*
        We immediately mark the output register as not alive, i.e. not in
        use, because we want this register to be re-used by one of the
        inputs. For example, a long chain of unary ops can all be
        performed on the same register without ever allocating a second
        register.

        This optimization is only possible when a tree has one output. With one
        output, we can assume that all instructions before the last instruction
        are just dependencies of the last instruction. And we can reuse the
        registers as we wish. When you have multiple outputs, you can track
        multiple paths of dependencies through the tree, each starting at a root
        node. A simple solution for this is to keep the root registers alive
        when the tree has multiple outputs.
        */
        alive[outreg] = roots.len() > 1 && roots.contains(&index);
        valregs[index] = None;
        let op = match node {
            Constant(val) => (Constant(*val), outreg),
            Symbol(label) => (Symbol(*label), outreg),
            Unary(op, input) => {
                let ireg = get_register(valregs, alive, *input);
                (Unary(*op, ireg), outreg)
            }
            Binary(op, lhs, rhs) => {
                let lreg = get_register(valregs, alive, *lhs);
                let rreg = get_register(valregs, alive, *rhs);
                (Binary(*op, lreg, rreg), outreg)
            }
            Ternary(op, a, b, c) => {
                let areg = get_register(valregs, alive, *a);
                let breg = get_register(valregs, alive, *b);
                let creg = get_register(valregs, alive, *c);
                (Ternary(*op, areg, breg, creg), outreg)
            }
        };
        ops.push(op);
    }
    outregs[num_outregs..].reverse();
    ops[num_ops..].reverse();
    alive.len()
}

/// Get the first register that isn't alive, i.e. is not in use. If all
/// registers are in use, i.e. alive, a new register is allocated.
fn get_register(valregs: &mut [Option<usize>], alive: &mut Vec<bool>, index: usize) -> usize {
    if let Some(val) = valregs[index] {
        return val;
    }
    let reg = match alive.iter().position(|flag| !*flag) {
        Some(reg) => {
            alive[reg] = true;
            reg
        }
        None => {
            let reg = alive.len();
            alive.push(true);
            reg
        }
    };
    valregs[index] = Some(reg);
    reg
}
