use rand::Rng;
use rand::distr::Open01;
use std::fmt;

/// Piecewise exponential distribution sampled via the inverse cumulative distribution.
///
/// # Examples
///
/// ```
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
/// use simtrial::PiecewiseExponential;
///
/// let durations = [0.5_f64, f64::INFINITY];
/// let rates = [0.25, 1.0];
/// let distribution = PiecewiseExponential::new(&durations, &rates).unwrap();
///
/// let mut rng = StdRng::seed_from_u64(7);
/// let draw = distribution.sample(&mut rng);
/// assert!(draw >= 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct PiecewiseExponential {
    rates: Vec<f64>,
    cumulative_time: Vec<f64>,
    cumulative_hazard: Vec<f64>,
}

impl PiecewiseExponential {
    /// Build a piecewise exponential distribution definition.
    ///
    /// # Parameters
    ///
    /// * `durations` - Lengths of each interval. All elements must be positive; only the final
    ///   element may be `f64::INFINITY` to represent an open-ended tail.
    /// * `rates` - Hazard rates for the associated intervals. All rates must be strictly positive
    ///   and finite.
    ///
    /// # Errors
    ///
    /// Returns [`PiecewiseExponentialError`] when the inputs violate the constraints described
    /// above.
    ///
    /// # Examples
    ///
    /// ```
    /// use rand::rngs::StdRng;
    /// use rand::{Rng, SeedableRng};
    /// use simtrial::PiecewiseExponential;
    ///
    /// let durations = [0.25, 0.5, f64::INFINITY];
    /// let rates = [1.0, 0.75, 1.5];
    /// let dist = PiecewiseExponential::new(&durations, &rates).unwrap();
    ///
    /// let mut rng = StdRng::seed_from_u64(42);
    /// let samples: Vec<f64> = (0..3).map(|_| dist.sample(&mut rng)).collect();
    /// assert_eq!(samples.len(), 3);
    /// ```
    pub fn new(durations: &[f64], rates: &[f64]) -> Result<Self, PiecewiseExponentialError> {
        let interval_count = durations.len();
        if interval_count == 0 {
            return Err(PiecewiseExponentialError::EmptyIntervals);
        }
        if interval_count != rates.len() {
            return Err(PiecewiseExponentialError::LengthMismatch {
                durations: interval_count,
                rates: rates.len(),
            });
        }

        let last_index = interval_count - 1;
        for (idx, &duration) in durations.iter().enumerate() {
            if duration.is_nan() {
                if idx == last_index {
                    return Err(PiecewiseExponentialError::FinalDurationInvalid);
                }
                return Err(PiecewiseExponentialError::NonFiniteDuration { index: idx });
            }
            if idx < last_index {
                if !duration.is_finite() {
                    return Err(PiecewiseExponentialError::NonFiniteDuration { index: idx });
                }
                if duration <= 0.0 {
                    return Err(PiecewiseExponentialError::NonPositiveDuration { index: idx });
                }
            } else {
                if duration <= 0.0 {
                    return Err(PiecewiseExponentialError::NonPositiveFinalDuration);
                }
                if !duration.is_finite() && !duration.is_infinite() {
                    return Err(PiecewiseExponentialError::FinalDurationInvalid);
                }
                if duration.is_infinite() && duration.is_sign_negative() {
                    return Err(PiecewiseExponentialError::NonPositiveFinalDuration);
                }
            }
        }

        for (idx, &rate) in rates.iter().enumerate() {
            if !rate.is_finite() {
                return Err(PiecewiseExponentialError::NonFiniteRate { index: idx });
            }
            if rate <= 0.0 {
                return Err(PiecewiseExponentialError::NonPositiveRate { index: idx });
            }
        }

        let mut cumulative_time = Vec::with_capacity(interval_count);
        let mut cumulative_hazard = Vec::with_capacity(interval_count);
        cumulative_time.push(0.0);
        cumulative_hazard.push(0.0);

        let mut time_acc = 0.0;
        let mut hazard_acc = 0.0;
        for idx in 0..last_index {
            time_acc += durations[idx];
            hazard_acc += durations[idx] * rates[idx];
            cumulative_time.push(time_acc);
            cumulative_hazard.push(hazard_acc);
        }

        Ok(Self {
            rates: rates.to_vec(),
            cumulative_time,
            cumulative_hazard,
        })
    }

    /// Draw a single sample from the distribution.
    ///
    /// The method accepts any [`rand::Rng`] implementation, so callers can use deterministic
    /// generators such as [`rand::rngs::StdRng`] for reproducibility.
    ///
    /// # Examples
    ///
    /// ```
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    /// use simtrial::PiecewiseExponential;
    ///
    /// let distribution = PiecewiseExponential::new(&[1.0], &[2.0]).unwrap();
    /// let mut rng = StdRng::seed_from_u64(123);
    /// let value = distribution.sample(&mut rng);
    /// assert!(value >= 0.0);
    /// ```
    pub fn sample<R>(&self, rng: &mut R) -> f64
    where
        R: Rng + ?Sized,
    {
        let uniform: f64 = rng.sample(Open01);
        let hazard = -uniform.ln();
        self.sample_from_hazard(hazard)
    }

    /// Transform a single uniform variate into a draw via the inverse cumulative distribution.
    ///
    /// This helper is useful when a caller needs to supply their own stream of uniforms, such as
    /// when replaying numbers recorded from another implementation for verification purposes.
    ///
    /// # Errors
    ///
    /// Returns [`PiecewiseExponentialSampleError::UniformOutOfRange`] when `uniform` is not within
    /// the open interval `(0, 1]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use simtrial::PiecewiseExponential;
    ///
    /// let dist = PiecewiseExponential::new(&[1.0], &[2.0]).unwrap();
    /// let draw = dist.inverse_cdf(0.75).unwrap();
    /// assert!(draw >= 0.0);
    /// ```
    pub fn inverse_cdf(&self, uniform: f64) -> Result<f64, PiecewiseExponentialSampleError> {
        if !(uniform > 0.0 && uniform <= 1.0) {
            return Err(PiecewiseExponentialSampleError::UniformOutOfRange { value: uniform });
        }
        let hazard = -uniform.ln();
        Ok(self.sample_from_hazard(hazard))
    }

    fn sample_from_hazard(&self, hazard: f64) -> f64 {
        let idx = self
            .cumulative_hazard
            .partition_point(|&value| value <= hazard)
            .saturating_sub(1);
        let base_time = self.cumulative_time[idx];
        let offset = (hazard - self.cumulative_hazard[idx]) / self.rates[idx];
        base_time + offset
    }

    /// Draw `n` samples and return them as a `Vec<f64>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    /// use simtrial::PiecewiseExponential;
    ///
    /// let dist = PiecewiseExponential::new(&[0.5, f64::INFINITY], &[1.0, 2.0]).unwrap();
    /// let mut rng = StdRng::seed_from_u64(999);
    /// let draws = dist.sample_n(4, &mut rng);
    /// assert_eq!(draws.len(), 4);
    /// ```
    pub fn sample_n<R>(&self, n: usize, rng: &mut R) -> Vec<f64>
    where
        R: Rng + ?Sized,
    {
        (0..n).map(|_| self.sample(rng)).collect()
    }
}

/// Errors emitted when constructing a [`PiecewiseExponential`] from invalid parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PiecewiseExponentialError {
    /// No intervals were supplied.
    EmptyIntervals,
    /// Durations and rates have mismatched lengths.
    LengthMismatch {
        /// Number of durations supplied.
        durations: usize,
        /// Number of rates supplied.
        rates: usize,
    },
    /// Encountered a non-finite duration outside the final interval.
    NonFiniteDuration {
        /// Index of the offending duration.
        index: usize,
    },
    /// Encountered a non-positive duration outside the final interval.
    NonPositiveDuration {
        /// Index of the offending duration.
        index: usize,
    },
    /// The last interval duration is not strictly positive.
    NonPositiveFinalDuration,
    /// The last interval duration is not finite nor positive infinity.
    FinalDurationInvalid,
    /// Encountered a non-finite rate.
    NonFiniteRate {
        /// Index of the offending rate.
        index: usize,
    },
    /// Encountered a non-positive rate.
    NonPositiveRate {
        /// Index of the offending rate.
        index: usize,
    },
}

impl fmt::Display for PiecewiseExponentialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PiecewiseExponentialError::EmptyIntervals => {
                f.write_str("durations must contain at least one interval")
            }
            PiecewiseExponentialError::LengthMismatch { durations, rates } => write!(
                f,
                "durations and rates must have the same length ({} vs {})",
                durations, rates
            ),
            PiecewiseExponentialError::NonFiniteDuration { index } => {
                write!(f, "duration at index {} must be finite", index)
            }
            PiecewiseExponentialError::NonPositiveDuration { index } => {
                write!(f, "duration at index {} must be positive", index)
            }
            PiecewiseExponentialError::NonPositiveFinalDuration => {
                f.write_str("final duration must be positive")
            }
            PiecewiseExponentialError::FinalDurationInvalid => f.write_str(
                "final duration must be finite or positive infinity (use f64::INFINITY)",
            ),
            PiecewiseExponentialError::NonFiniteRate { index } => {
                write!(f, "rate at index {} must be finite", index)
            }
            PiecewiseExponentialError::NonPositiveRate { index } => {
                write!(f, "rate at index {} must be strictly positive", index)
            }
        }
    }
}

impl std::error::Error for PiecewiseExponentialError {}

/// Errors that may occur while transforming explicit uniforms into samples.
#[derive(Debug, Clone, PartialEq)]
pub enum PiecewiseExponentialSampleError {
    /// The provided uniform variate did not fall inside the valid open interval `(0, 1]`.
    UniformOutOfRange {
        /// The provided uniform variate.
        value: f64,
    },
}

impl fmt::Display for PiecewiseExponentialSampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PiecewiseExponentialSampleError::UniformOutOfRange { value } => write!(
                f,
                "uniform variate {} must lie within the interval (0, 1]",
                value
            ),
        }
    }
}

impl std::error::Error for PiecewiseExponentialSampleError {}
