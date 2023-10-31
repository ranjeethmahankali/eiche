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
    vertices: Vec<Vertex>,
    halfedges: Vec<HalfEdge>,
}

impl Graph {
    pub fn from(tree: &Tree) -> Graph {
        let mut graph = Graph {
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
        tree.traverse_depth(
            |index, maybe_parent| -> Result<(), ()> {
                if let Some(parent) = maybe_parent {
                    graph.add_edge(parent, index);
                }
                return Ok(());
            },
            true,
        )
        .expect("Unreachable codepath: Unable to construct a graph.");
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
}

impl Tree {
    pub fn simplify(self) -> Tree {
        let _graph = Graph::from(&self);
        todo!();
    }
}
