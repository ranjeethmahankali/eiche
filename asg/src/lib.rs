pub mod eval;
pub mod io;
pub mod lisp;
pub mod latex;
pub mod reduce;
pub mod tree;

mod dedup;
mod fold;
mod macros;
mod mutate;
mod prune;
mod sort;
mod template;
mod walk;

#[cfg(test)]
mod test;
