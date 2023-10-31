use crate::tree::{Node, Node::*, Tree};

struct HalfEdge {
    vertex: usize,
    next: Option<usize>,
}

struct Vertex {
    node: Node,
    down: Option<usize>,
    up: Option<usize>,
}

struct Graph {
    root_index: usize,
    vertices: Vec<Vertex>,
    halfedges: Vec<HalfEdge>,
}

impl Graph {
    pub fn from(tree: &Tree) -> Graph {
        let mut graph = Graph {
            root_index: tree.root_index(),
            vertices: tree
                .nodes()
                .iter()
                .map(|n| Vertex {
                    node: n.clone(),
                    down: None,
                    up: None,
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
                graph.add_edge(parent, index);
            }
        }
        return graph;
    }

    fn add_halfedge(&mut self, vertex: usize) -> usize {
        let index = self.halfedges.len();
        self.halfedges.push(HalfEdge { vertex, next: None });
        return index;
    }

    fn add_edge(&mut self, parent: usize, child: usize) {
        // Add halfedge going parent to child.
        match self.vertices[parent].down {
            Some(he) => {
                let mut he = he;
                while let Some(next) = self.halfedges[he].next {
                    he = next;
                }
                self.halfedges[he].next = Some(self.add_halfedge(child));
            }
            None => {
                self.vertices[parent].down = Some(self.add_halfedge(child));
            }
        };
        // Add halfedge from child to parent.
        match self.vertices[child].up {
            Some(he) => {
                let mut he = he;
                while let Some(next) = self.halfedges[he].next {
                    he = next;
                }
                self.halfedges[he].next = Some(self.add_halfedge(parent));
            }
            None => {
                self.vertices[parent].up = Some(self.add_halfedge(parent));
            }
        };
    }

    pub fn iter_depth(&self) -> GraphDepthIterator {
        GraphDepthIterator::from(&self)
    }

    pub fn to_tree(&self) -> Result<Tree, GraphConversionError> {
        let indices = {
            let mut out: Vec<usize> = self.iter_depth().collect();
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
                    Unary(op, _) => {
                        if let Some(he) = vertex.down {
                            Unary(op, map[self.halfedges[he].vertex])
                        } else {
                            return Err(GraphConversionError);
                        }
                    }
                    Binary(op, _, _) => {
                        if let Some(he) = vertex.down {
                            let lhs = map[self.halfedges[he].vertex];
                            if let Some(he2) = self.halfedges[he].next {
                                Binary(op, lhs, map[self.halfedges[he2].vertex])
                            } else {
                                return Err(GraphConversionError);
                            }
                        } else {
                            return Err(GraphConversionError);
                        }
                    }
                })
            }
            nodes
        }));
    }
}

#[derive(Debug)]
struct GraphConversionError;

struct GraphDepthIterator<'a> {
    graph: &'a Graph,
    stack: Vec<usize>,
    visited: Box<[bool]>,
}

impl<'a> GraphDepthIterator<'a> {
    fn from(graph: &Graph) -> GraphDepthIterator {
        let mut iter = GraphDepthIterator {
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
        let vertex = &self.graph.vertices[vindex];
        match vertex.down {
            Some(he) => {
                self.stack.push(self.graph.halfedges[he].vertex);
                let mut he = he;
                while let Some(next) = self.graph.halfedges[he].next {
                    he = next;
                    self.stack.push(self.graph.halfedges[he].vertex);
                }
            }
            None => {}
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
        let trees = vec![min(sqrt('x'.into()), sqrt('y'.into()))];
        for i in 0..trees.len() {
            println!("Checking round trip conversion for tree {}...", i);
            let converted = Graph::from(&trees[i])
                .to_tree()
                .expect("Failed to convert graph to tree.");
            assert_eq!(trees[i], converted);
        }
    }
}
