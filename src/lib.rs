#![warn(missing_docs)]

//! Clinical trial simulation
//!
//! This crate ports core survival distribution sampling routines from the R
//! simtrial package to Rust for fast simulation workflows.

mod piecewise_exponential;

pub use piecewise_exponential::{
    PiecewiseExponential, PiecewiseExponentialError, PiecewiseExponentialSampleError,
};
