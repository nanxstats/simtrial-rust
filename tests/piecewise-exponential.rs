use rand::distr::Open01;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use simtrial::{PiecewiseExponential, PiecewiseExponentialError, PiecewiseExponentialSampleError};

mod common;

use common::assert_close_slice;

#[test]
fn samples_are_reproducible_with_fixed_seed() {
    let dist = PiecewiseExponential::new(&[1.0], &[2.0]).unwrap();

    let mut rng_a = StdRng::seed_from_u64(42);
    let mut rng_b = StdRng::seed_from_u64(42);

    let draws_a: Vec<f64> = (0..5).map(|_| dist.sample(&mut rng_a)).collect();
    let draws_b: Vec<f64> = (0..5).map(|_| dist.sample(&mut rng_b)).collect();

    assert_close_slice(&draws_a, &draws_b);
}

#[test]
fn single_interval_matches_manual_inverse_transform() {
    let rate = 0.75;
    let dist = PiecewiseExponential::new(&[1.0], &[rate]).unwrap();

    let mut rng_samples = StdRng::seed_from_u64(123);
    let mut rng_uniforms = StdRng::seed_from_u64(123);

    let draws = dist.sample_n(10, &mut rng_samples);
    let expected: Vec<f64> = (0..10)
        .map(|_| {
            let u: f64 = rng_uniforms.sample(Open01);
            -u.ln() / rate
        })
        .collect();

    assert_close_slice(&draws, &expected);
}

#[test]
fn multi_interval_inverse_transform_matches_manual_reference() {
    let durations = [0.5, 0.5, 1.0];
    let rates = [1.0, 3.0, 10.0];
    let dist = PiecewiseExponential::new(&durations, &rates).unwrap();

    let mut rng_samples = StdRng::seed_from_u64(789);
    let mut rng_uniforms = StdRng::seed_from_u64(789);

    let draws = dist.sample_n(12, &mut rng_samples);
    let expected: Vec<f64> = (0..12)
        .map(|_| {
            let u: f64 = rng_uniforms.sample(Open01);
            let hazard = -u.ln();
            let cum_time = [0.0, durations[0], durations[0] + durations[1]];
            let cum_hazard = [
                0.0,
                durations[0] * rates[0],
                durations[0] * rates[0] + durations[1] * rates[1],
            ];
            let idx = cum_hazard
                .iter()
                .rposition(|&value| value <= hazard)
                .unwrap_or(0);
            cum_time[idx] + (hazard - cum_hazard[idx]) / rates[idx.min(rates.len() - 1)]
        })
        .collect();

    assert_close_slice(&draws, &expected);
}

#[test]
fn infinite_tail_is_supported() {
    let dist = PiecewiseExponential::new(&[1.0, f64::INFINITY], &[0.5, 1.0]).unwrap();
    let mut rng = StdRng::seed_from_u64(2024);
    let draws = dist.sample_n(8, &mut rng);

    assert_eq!(draws.len(), 8);
    assert!(draws.iter().all(|value| value.is_finite()));
}

#[test]
fn sample_n_respects_requested_length() {
    let dist = PiecewiseExponential::new(&[0.5, 1.0], &[1.0, 1.5]).unwrap();
    let mut rng = StdRng::seed_from_u64(1);

    let non_empty = dist.sample_n(3, &mut rng);
    assert_eq!(non_empty.len(), 3);

    let empty = dist.sample_n(0, &mut rng);
    assert!(empty.is_empty());
}

#[test]
fn inverse_cdf_rejects_out_of_range_uniforms() {
    let dist = PiecewiseExponential::new(&[1.0], &[2.0]).unwrap();

    for &value in &[0.0, -0.1, 1.5, f64::NAN] {
        let err = dist.inverse_cdf(value).unwrap_err();
        assert!(matches!(
            err,
            PiecewiseExponentialSampleError::UniformOutOfRange { value: observed }
                if observed.to_bits() == value.to_bits()
        ));
    }
}

#[test]
fn replaying_uniform_stream_matches_rng_draws() {
    let durations = [0.5, 0.5, 1.0];
    let rates = [1.0, 3.0, 10.0];
    let dist = PiecewiseExponential::new(&durations, &rates).unwrap();

    let mut rng_for_uniforms = StdRng::seed_from_u64(456);
    let uniforms: Vec<f64> = (0..10).map(|_| rng_for_uniforms.sample(Open01)).collect();

    let mut rng_for_samples = StdRng::seed_from_u64(456);
    let expected = dist.sample_n(uniforms.len(), &mut rng_for_samples);

    let via_inverse: Vec<f64> = uniforms
        .iter()
        .map(|&u| dist.inverse_cdf(u).unwrap())
        .collect();

    assert_close_slice(&via_inverse, &expected);
}

#[test]
fn invalid_parameters_trigger_informative_errors() {
    assert!(matches!(
        PiecewiseExponential::new(&[], &[]).unwrap_err(),
        PiecewiseExponentialError::EmptyIntervals
    ));

    assert!(matches!(
        PiecewiseExponential::new(&[1.0], &[]).unwrap_err(),
        PiecewiseExponentialError::LengthMismatch { .. }
    ));

    assert!(matches!(
        PiecewiseExponential::new(&[f64::INFINITY, 1.0], &[1.0, 1.0]).unwrap_err(),
        PiecewiseExponentialError::NonFiniteDuration { index: 0 }
    ));

    assert!(matches!(
        PiecewiseExponential::new(&[0.0, 1.0], &[1.0, 1.0]).unwrap_err(),
        PiecewiseExponentialError::NonPositiveDuration { index: 0 }
    ));

    assert!(matches!(
        PiecewiseExponential::new(&[1.0, f64::INFINITY], &[1.0, 0.0]).unwrap_err(),
        PiecewiseExponentialError::NonPositiveRate { index: 1 }
    ));

    assert!(matches!(
        PiecewiseExponential::new(&[1.0, f64::INFINITY], &[1.0, f64::NAN]).unwrap_err(),
        PiecewiseExponentialError::NonFiniteRate { index: 1 }
    ));

    assert!(matches!(
        PiecewiseExponential::new(&[1.0, f64::NAN], &[1.0, 1.0]).unwrap_err(),
        PiecewiseExponentialError::FinalDurationInvalid
    ));

    assert!(matches!(
        PiecewiseExponential::new(&[1.0, -1.0], &[1.0, 1.0]).unwrap_err(),
        PiecewiseExponentialError::NonPositiveFinalDuration
    ));
}
