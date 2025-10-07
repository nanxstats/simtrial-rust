use simtrial::PiecewiseExponential;

mod common;

use common::{assert_close_slice, load_columns};

#[test]
fn inverse_cdf_matches_r_reference_single_interval() {
    let columns = load_columns("pwexp_single_seed_123_n20.txt");
    let uniforms = &columns[0];
    let expected = &columns[1];

    let dist = PiecewiseExponential::new(&[1.0], &[2.0]).unwrap();
    let actual: Vec<f64> = uniforms
        .iter()
        .map(|&u| dist.inverse_cdf(u).unwrap())
        .collect();

    assert_close_slice(&actual, expected);
}

#[test]
fn inverse_cdf_matches_r_reference_multi_interval() {
    let columns = load_columns("pwexp_multi_seed_456_n30.txt");
    let uniforms = &columns[0];
    let expected = &columns[1];

    let durations = [0.5, 0.5, 1.0];
    let rates = [1.0, 3.0, 10.0];
    let dist = PiecewiseExponential::new(&durations, &rates).unwrap();

    let actual: Vec<f64> = uniforms
        .iter()
        .map(|&u| dist.inverse_cdf(u).unwrap())
        .collect();

    assert_close_slice(&actual, expected);
}
