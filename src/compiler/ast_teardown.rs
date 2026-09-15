//! Dropping an AST subtree without recursing.
//!
//! When the stack guard in `compile_expr` bails, it still owns the entire un-compiled remainder of
//! the program. Letting that go out of scope runs the derived drop glue, which reaches into each
//! nesting level in turn and costs stack proportional to the *program's* depth — not to the depth
//! at which compilation actually stopped. So the guard would fire correctly and then the process
//! would die cleaning up, which is exactly what it was trying to prevent.
//!
//! This tears the subtree down against an explicit heap worklist instead: pop a node, move its
//! children onto the worklist, let the childless remainder drop. Nothing recurses, so no depth of
//! nesting can exhaust the stack.
//!
//! Every `match` here is exhaustive on purpose — no `_` arm. A new AST variant must fail to
//! compile rather than silently reintroduce a recursive drop path.

use crate::parser::ast::Node;
use crate::parser::ast::{
    Abort, Assignment, Block, Container, Expr, FunctionCall, FunctionClosure, IfStatement, Op,
    Predicate, Query, QueryTarget, Return, Unary,
};

/// Drops `root` and everything below it iteratively.
pub(super) fn drop_expr(root: Node<Expr>) {
    let mut worklist = vec![root.into_inner()];
    while let Some(expr) = worklist.pop() {
        push_children(expr, &mut worklist);
    }
}

/// Moves every `Expr` directly beneath `expr` onto `worklist`. Whatever is left of `expr` — spans,
/// identifiers, operators — is shallow and drops normally when this returns.
fn push_children(expr: Expr, worklist: &mut Vec<Expr>) {
    match expr {
        // Leaves. A string literal can hold template segments, but those are identifiers, not
        // expressions, so there is nothing nested to unwind here.
        Expr::Literal(_) | Expr::Variable(_) => {}

        Expr::Container(node) => push_container(node.into_inner(), worklist),

        Expr::IfStatement(node) => {
            let IfStatement {
                predicate,
                if_node,
                else_node,
            } = node.into_inner();
            push_predicate(predicate.into_inner(), worklist);
            push_block(if_node.into_inner(), worklist);
            if let Some(block) = else_node {
                push_block(block.into_inner(), worklist);
            }
        }

        Expr::Op(node) => {
            let Op(lhs, _opcode, rhs) = node.into_inner();
            push_boxed(lhs, worklist);
            push_boxed(rhs, worklist);
        }

        Expr::Assignment(node) => match node.into_inner() {
            Assignment::Single { expr, .. } | Assignment::Infallible { expr, .. } => {
                push_boxed(expr, worklist);
            }
        },

        Expr::Query(node) => {
            let Query { target, path: _ } = node.into_inner();
            match target.into_inner() {
                QueryTarget::Internal(_) | QueryTarget::External(_) => {}
                QueryTarget::FunctionCall(call) => push_function_call(call, worklist),
                QueryTarget::Container(container) => push_container(container, worklist),
            }
        }

        Expr::FunctionCall(node) => push_function_call(node.into_inner(), worklist),

        Expr::Unary(node) => match node.into_inner() {
            Unary::Not(not) => {
                let (_span, expr) = not.into_inner().take();
                push_boxed(expr, worklist);
            }
        },

        Expr::Abort(node) => {
            let Abort { message } = node.into_inner();
            if let Some(expr) = message {
                push_boxed(expr, worklist);
            }
        }

        Expr::Return(node) => {
            let Return { expr } = node.into_inner();
            push_boxed(expr, worklist);
        }
    }
}

fn push_container(container: Container, worklist: &mut Vec<Expr>) {
    match container {
        Container::Group(group) => worklist.push((*group).into_inner().into_inner().into_inner()),
        Container::Block(block) => push_block(block.into_inner(), worklist),
        Container::Array(array) => {
            worklist.extend(array.into_inner().0.into_iter().map(Node::into_inner));
        }
        Container::Object(object) => {
            worklist.extend(object.into_inner().0.into_values().map(Node::into_inner));
        }
    }
}

fn push_block(block: Block, worklist: &mut Vec<Expr>) {
    worklist.extend(block.into_inner().into_iter().map(Node::into_inner));
}

fn push_predicate(predicate: Predicate, worklist: &mut Vec<Expr>) {
    match predicate {
        Predicate::One(expr) => push_boxed(expr, worklist),
        Predicate::Many(exprs) => worklist.extend(exprs.into_iter().map(Node::into_inner)),
    }
}

fn push_function_call(call: FunctionCall, worklist: &mut Vec<Expr>) {
    let FunctionCall {
        ident: _,
        abort_on_error: _,
        arguments,
        closure,
    } = call;

    worklist.extend(
        arguments
            .into_iter()
            .map(|argument| argument.into_inner().expr.into_inner()),
    );

    if let Some(closure) = closure {
        let FunctionClosure {
            variables: _,
            block,
        } = closure.into_inner();
        push_block(block.into_inner(), worklist);
    }
}

fn push_boxed(expr: Box<Node<Expr>>, worklist: &mut Vec<Expr>) {
    worklist.push((*expr).into_inner());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses `source` on a generous stack, then tears the AST down on a deliberately small one.
    /// Returns normally only if the teardown never recursed; a recursive drop aborts the process.
    fn parse_then_drop_on_small_stack(source: String, stack: usize) {
        let program = std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(move || crate::parser::parse(&source).expect("parse"))
            .expect("spawn")
            .join()
            .expect("join");

        std::thread::Builder::new()
            .stack_size(stack)
            .spawn(move || {
                for root in program.0 {
                    if let crate::parser::ast::RootExpr::Expr(expr) = root.into_inner() {
                        drop_expr(expr);
                    }
                }
            })
            .expect("spawn")
            .join()
            .expect("join");
    }

    #[test]
    fn tears_down_deeply_nested_unary_without_recursing() {
        parse_then_drop_on_small_stack("!".repeat(30_000) + "true", 512 * 1024);
    }

    #[test]
    fn tears_down_deeply_nested_containers_without_recursing() {
        let depth = 10_000;
        parse_then_drop_on_small_stack("[".repeat(depth) + &"]".repeat(depth), 512 * 1024);
    }

    #[test]
    fn tears_down_deeply_nested_groups_without_recursing() {
        let depth = 5_000;
        parse_then_drop_on_small_stack("(".repeat(depth) + "true" + &")".repeat(depth), 512 * 1024);
    }
}
