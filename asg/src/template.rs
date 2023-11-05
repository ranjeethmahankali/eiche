use crate::tree::{BinaryOp, Node, UnaryOp};

pub struct Template {
    ping: Vec<Node>,
    pong: Vec<Node>,
}

impl Template {
    pub fn from(ping: Vec<Node>, pong: Vec<Node>) -> Template {
        Template { ping, pong }
    }
}
