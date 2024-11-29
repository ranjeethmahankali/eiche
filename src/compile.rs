use crate::tree::{Node, Node::*, Tree};

/*
Compile the tree into instructions. In theory one can just walk the nodes of the
// tree and evaluate it without compiling it, but that requires allocating as
// many registers as there are nodes in the tree. Here we compile the tree into
// instructions that reuse registers using Matt Keeter's Solid State Register
// Allocator: https://www.mattkeeter.com/blog/2022-10-04-ssra/
 */

pub struct Instructions {
    pub ops: Vec<(Node, usize)>,
    pub num_regs: usize,
    pub out_regs: Vec<usize>,
}

/// We "compile" the tree into a set of ops that closely mirror the nodes of the
/// tree itself. The difference between the compiled ops and the tree nodes is
/// that the former reference register indices, where the registers are
/// reused. An 'instruction' is a tuple of a Node and a usize index pointing to
/// the register that the output of the instruction should be written into.
pub fn compile(tree: &Tree) -> Instructions {
    let roots = tree.root_indices();
    let mut root_regs = Vec::new();
    let mut valregs = vec![None; tree.len()];
    let mut alive: Vec<bool> = Vec::new();
    let mut ops = Vec::new();
    // Iterate in reverse.
    for (index, node) in tree.nodes().iter().enumerate().rev() {
        let outreg = get_register(&mut valregs, &mut alive, index);
        if roots.contains(&index) {
            root_regs.push(outreg);
        }
        // We immediately mark the output register as not alive, i.e. not in
        // use, because we want this register to be re-used by one of the
        // inputs. For example, a long chain of unary ops can all be
        // performed on the same register without ever allocating a second
        // register.
        alive[outreg] = false;
        valregs[index] = None;
        let op = match node {
            Constant(val) => (Constant(*val), outreg),
            Symbol(label) => (Symbol(*label), outreg),
            Unary(op, input) => {
                let ireg = get_register(&mut valregs, &mut alive, *input);
                (Unary(*op, ireg), outreg)
            }
            Binary(op, lhs, rhs) => {
                let lreg = get_register(&mut valregs, &mut alive, *lhs);
                let rreg = get_register(&mut valregs, &mut alive, *rhs);
                (Binary(*op, lreg, rreg), outreg)
            }
            Ternary(op, a, b, c) => {
                let areg = get_register(&mut valregs, &mut alive, *a);
                let breg = get_register(&mut valregs, &mut alive, *b);
                let creg = get_register(&mut valregs, &mut alive, *c);
                (Ternary(*op, areg, breg, creg), outreg)
            }
        };
        ops.push(op);
    }
    root_regs.reverse();
    ops.reverse();
    Instructions {
        ops,
        num_regs: alive.len(),
        out_regs: root_regs,
    }
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
