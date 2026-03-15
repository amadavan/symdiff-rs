#![allow(unused)]

use symdiff::gradient;

#[gradient(dim = 2)]
fn add(x: &[f64]) -> f64 {
    3.0 * x[0] + 4.0 * x[1]
}

#[gradient(dim = 2, max_passes = 5)]
fn subtract(x: &[f64]) -> f64 {
    3.0 * x[0] - 4.0 * x[1]
}

#[gradient(dim = 2)]
fn linear(x: &[f64]) -> f64 {
    3.0 * x[0] + 4.0 * x[1]
}

#[gradient(dim = 2)]
fn quadratic(x: &[f64]) -> f64 {
    x[0].powi(2) + x[0] * x[1] + x[1].powi(3)
}

#[gradient(dim = 2)]
fn multiply(x: &[f64]) -> f64 {
    x[0] * x[1]
}

#[gradient(dim = 2)]
fn divide(x: &[f64]) -> f64 {
    x[0] / x[1]
}

#[gradient(dim = 2)]
fn rosenbrock(x: &[f64]) -> f64 {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
}

#[gradient(dim = 2, max_passes = 0)]
fn rosenbrock_unsimplified(x: &[f64]) -> f64 {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
}

#[gradient(dim = 1)]
fn trig(x: &[f64]) -> f64 {
    x[0].sin() + x[0].cos().powi(2)
}

// let-binding inlining
#[gradient(dim = 2)]
fn with_lets(x: &[f64]) -> f64 {
    let a = x[0].powi(2);
    let b = 2.0 * x[1];
    a + b
}

mod test {
    use super::*;

    #[test]
    fn test_add() {
        let f = super::add(&[2.0, 3.0]);
        assert!((f - 18.0).abs() < 1e-10, "f = {}", f);
        let g = add_gradient(&[2.0, 3.0]);
        assert!((g[0] - 3.0).abs() < 1e-10, "df/dx = {}", g[0]);
        assert!((g[1] - 4.0).abs() < 1e-10, "df/dy = {}", g[1]);
    }

    #[test]
    fn add_grad() {
        let g = add_gradient(&[2.0, 3.0]);
        assert!((g[0] - 3.0).abs() < 1e-10, "df/dx = {}", g[0]);
        assert!((g[1] - 4.0).abs() < 1e-10, "df/dy = {}", g[1]);
    }

    #[test]
    fn subtract_grad() {
        let g = subtract_gradient(&[2.0, 3.0]);
        assert!((g[0] - 3.0).abs() < 1e-10, "df/dx = {}", g[0]);
        assert!((g[1] + 4.0).abs() < 1e-10, "df/dy = {}", g[1]);
    }

    #[test]
    fn multiply_grad() {
        let g = multiply_gradient(&[2.0, 3.0]);
        assert!((g[0] - 3.0).abs() < 1e-10, "df/dx = {}", g[0]);
        assert!((g[1] - 2.0).abs() < 1e-10, "df/dy = {}", g[1]);
    }

    #[test]
    fn divide_grad() {
        let g = divide_gradient(&[2.0, 3.0]);
        assert!((g[0] - 1.0 / 3.0).abs() < 1e-10, "df/dx = {}", g[0]);
        assert!((g[1] + 2.0 / 9.0).abs() < 1e-10, "df/dy = {}", g[1]);
    }

    #[test]
    fn linear_grad() {
        let g = linear_gradient(&[1.0, 1.0]);
        assert!((g[0] - 3.0).abs() < 1e-10, "df/dx = {}", g[0]);
        assert!((g[1] - 4.0).abs() < 1e-10, "df/dy = {}", g[1]);
    }

    #[test]
    fn rosenbrock_grad_at_minimum() {
        // At (1, 1), gradient of Rosenbrock is (0, 0).
        let g = rosenbrock_gradient(&[1.0, 1.0]);
        assert!((g[0]).abs() < 1e-10, "df/dx = {}", g[0]);
        assert!((g[1]).abs() < 1e-10, "df/dy = {}", g[1]);
    }

    #[test]
    fn rosenbrock_grad_at_0() {
        // At (0, 0), gradient of Rosenbrock is (-2, -200).
        let g = rosenbrock_gradient(&[0.0, 0.0]);
        assert!((g[0] + 2.0).abs() < 1e-10, "df/dx = {}", g[0]);
        assert!((g[1] - 0.0).abs() < 1e-10, "df/dy = {}", g[1]);
    }

    #[test]
    fn quadratic_grad() {
        let g = quadratic_gradient(&[2.0, 3.0]);
        assert!((g[0] - 7.0).abs() < 1e-10, "df/dx = {}", g[0]);
        assert!((g[1] - 29.0).abs() < 1e-10, "df/dy = {}", g[1]);
    }

    #[test]
    fn trig_grad_at_zero() {
        // d/dx [sin(x) + cos²(x)] at x=0 = cos(0) - 2sin(0)cos(0) = 1
        let g = trig_gradient(&[0.0]);
        assert!((g[0] - 1.0).abs() < 1e-10, "df/dx = {}", g[0]);
    }

    #[test]
    fn let_binding_inlining() {
        // f = x² + 2y  →  df/dx = 2x, df/dy = 2
        let g = with_lets_gradient(&[3.0, 5.0]);
        assert!((g[0] - 6.0).abs() < 1e-10, "df/dx = {}", g[0]);
        assert!((g[1] - 2.0).abs() < 1e-10, "df/dy = {}", g[1]);
    }
}
