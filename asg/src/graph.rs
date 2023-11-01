use crate::tree::{Node, Node::*, Tree};

struct OutgoingEdge {
    vertex: usize,
    next: Option<usize>,
}

struct Vertex {
    node: Node,
    outgoing: Option<usize>,
}

struct Graph {
    root_index: usize,
    vertices: Vec<Vertex>,
    halfedges: Vec<OutgoingEdge>,
}

impl Graph {
    pub fn from(tree: &Tree) -> Graph {
        let mut graph = Graph {
            root_index: tree.root_index(),
            vertices: tree
                .nodes()
                .iter()
                .map(|&node| Vertex {
                    node,
                    outgoing: None,
                })
                .collect(),
            halfedges: Vec::with_capacity(
                // Count the edges in the tree.
                tree.nodes()
                    .iter()
                    .map(|node| match node {
                        Constant(_) | Symbol(_) => 0,
                        Unary(_, _) => 1,
                        Binary(_, _, _) => 2,
                    })
                    .sum(),
            ),
        };
        for (index, maybe_parent) in tree.iter_depth(true) {
            if let Some(parent) = maybe_parent {
                graph.add_incoming(parent, index);
            }
        }
        return graph;
    }

    fn add_incoming(&mut self, parent: usize, child: usize) {
        self.vertices[child].outgoing = Some({
            let prev = self.vertices[child].outgoing;
            let index = self.halfedges.len();
            self.halfedges.push(OutgoingEdge {
                vertex: parent,
                next: prev,
            });
            index
        });
    }

    pub fn iter_depth(&self, mirrored: bool) -> GraphDepthIterator {
        GraphDepthIterator::from(&self, mirrored)
    }

    pub fn to_tree(&self) -> Result<Tree, GraphConversionError> {
        let indices = {
            // We'll use the mirrored depth first iterator because
            // we'll later reverse the whole vector. This will produce
            // a node order consistent with the original tree that was
            // used to create this graph. Either way the tree should
            // be computationally equivalent, this is just a minor
            // detail to produce the same tree for round trip
            // conversions.
            let mut out: Vec<usize> = self.iter_depth(true).collect();
            out.reverse();
            out.into_boxed_slice()
        };
        let map = {
            let mut map = vec![0; indices.len()].into_boxed_slice();
            for i in 0..indices.len() {
                map[indices[i]] = i;
            }
            map
        };
        return Ok(Tree::from({
            let mut nodes: Vec<Node> = Vec::with_capacity(indices.len());
            for &index in indices.iter() {
                let vertex = &self.vertices[index];
                nodes.push(match vertex.node {
                    Constant(val) => Constant(val),
                    Symbol(label) => Symbol(label),
                    Unary(op, input) => Unary(op, map[input]),
                    Binary(op, lhs, rhs) => Binary(op, map[lhs], map[rhs]),
                })
            }
            nodes
        }));
    }
}

#[derive(Debug)]
struct GraphConversionError;

struct GraphDepthIterator<'a> {
    mirrored: bool,
    graph: &'a Graph,
    stack: Vec<usize>,
    visited: Box<[bool]>,
}

impl<'a> GraphDepthIterator<'a> {
    fn from(graph: &Graph, mirrored: bool) -> GraphDepthIterator {
        let mut iter = GraphDepthIterator {
            mirrored,
            graph,
            stack: Vec::with_capacity(graph.vertices.len()),
            visited: vec![false; graph.vertices.len()].into_boxed_slice(),
        };
        iter.stack.push(graph.root_index);
        return iter;
    }
}

impl<'a> Iterator for GraphDepthIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let vindex = {
            let mut vindex = self.stack.pop()?;
            while self.visited[vindex] {
                vindex = self.stack.pop()?;
            }
            vindex
        };
        match &self.graph.vertices[vindex].node {
            Constant(_) | Symbol(_) => {} // Do nothing.
            Unary(_op, input) => {
                self.stack.push(*input);
            }
            Binary(_op, lhs, rhs) => {
                if self.mirrored {
                    self.stack.push(*lhs);
                    self.stack.push(*rhs);
                } else {
                    self.stack.push(*rhs);
                    self.stack.push(*lhs);
                }
            }
        }
        return Some(vindex);
    }
}

pub struct SimplificationError;

impl Tree {
    pub fn simplify(self) -> Result<Tree, SimplificationError> {
        let graph = Graph::from(&self);
        // Do the simplification.
        let _result = match graph.to_tree() {
            Ok(tree) => Ok(tree),
            Err(_) => Err(SimplificationError),
        };
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::*;

    #[test]
    fn round_trip_conversion() {
        let trees = vec![
            {
                let x: Tree = 'x'.into();
                x + 'y'.into()
            },
            min(sqrt('x'.into()), sqrt('y'.into())),
            (pow(sin('x'.into()), 2.0.into())
                + pow(cos('x'.into()), 2.0.into())
                + ((cos('x'.into()) * sin('x'.into())) * 2.0.into()))
                / (pow(sin('y'.into()), 2.0.into())
                    + pow(cos('y'.into()), 2.0.into())
                    + ((cos('y'.into()) * sin('y'.into())) * 2.0.into())),
            pow(log(sin('x'.into()) + 2.0.into()), 3.0.into()) / (cos('x'.into()) + 2.0.into()),
        ];
        for i in 0..trees.len() {
            println!("Checking round trip conversion for tree {}...", i);
            let converted = Graph::from(&trees[i])
                .to_tree()
                .expect("Failed to convert graph to tree.");
            assert_eq!(trees[i], converted);
        }
    }
}
