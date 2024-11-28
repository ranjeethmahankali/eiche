use crate::tree::{Node, Tree};
use std::ops::Range;

struct Edge(usize, usize);

#[derive(Default, Clone)]
struct Vertex {
    parents: Range<usize>,
    children: Range<usize>,
}

struct Graph {
    edges: Vec<Edge>,
    vertices: Vec<Vertex>,
    incoming: Vec<usize>,
}

impl Graph {
    pub fn from_tree(tree: &Tree) -> Self {
        let (edges, num_incoming) = {
            let mut edges: Vec<Edge> = Vec::new();
            let mut num_incoming: Vec<usize> = vec![0usize; tree.len()];
            for (index, node) in tree.nodes().iter().enumerate() {
                match node {
                    Node::Constant(_) | Node::Symbol(_) => {} // Do nothing.
                    Node::Unary(_, a) => {
                        edges.push(Edge(index, *a));
                        num_incoming[*a] += 1;
                    }
                    Node::Binary(_, a, b) => {
                        for child in [a, b] {
                            edges.push(Edge(index, *child));
                            num_incoming[*child] += 1;
                        }
                    }
                    Node::Ternary(_, a, b, c) => {
                        for child in [a, b, c] {
                            edges.push(Edge(index, *child));
                            num_incoming[*child] += 1;
                        }
                    }
                }
            }
            (edges, num_incoming)
        };
        let vertices: Vec<Vertex> = tree
            .nodes()
            .iter()
            .zip(num_incoming.iter())
            .scan(Vertex::default(), |prev, (node, incoming)| {
                Some(Vertex {
                    parents: prev.parents.end..(prev.parents.end + *incoming),
                    children: prev.children.end
                        ..(prev.children.end
                            + match node {
                                Node::Constant(_) | Node::Symbol(_) => 0,
                                Node::Unary(_, _) => 1,
                                Node::Binary(_, _, _) => 2,
                                Node::Ternary(_, _, _, _) => 3,
                            }),
                })
            })
            .collect();
        let mut offsets = vec![0usize; tree.len()];
        let mut incoming =
            vec![usize::MAX; vertices.last().map(|v| v.parents.end).unwrap_or(0usize)];
        for (ei, Edge(_, child)) in edges.iter().enumerate() {
            let offset = &mut offsets[*child];
            let span = &vertices[*child].parents;
            assert!(*offset < span.len()); // Cannot write out of bounds.
            incoming[span.start + *offset] = ei;
            *offset += 1;
        }
        // Make sure all positions in the array are written to.
        assert!(incoming.iter().all(|ei| *ei != usize::MAX));
        Graph {
            edges,
            vertices,
            incoming,
        }
    }
}
