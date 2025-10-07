#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

const ABS_TOL: f64 = 1e-14;
const REL_TOL: f64 = 1e-12;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

pub fn load_columns(filename: &str) -> Vec<Vec<f64>> {
    let path = fixtures_dir().join(filename);
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read {}: {}", path.display(), err));
    let mut columns: Vec<Vec<f64>> = Vec::new();
    for (line_idx, line) in contents.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: Vec<f64> = trimmed
            .split_whitespace()
            .map(|field| {
                field.parse::<f64>().unwrap_or_else(|err| {
                    panic!(
                        "failed to parse float at {} (line {}): {}",
                        path.display(),
                        line_idx + 1,
                        err
                    )
                })
            })
            .collect();
        if columns.is_empty() {
            columns.resize_with(row.len(), Vec::new);
        } else {
            assert_eq!(
                columns.len(),
                row.len(),
                "row {} in {} has inconsistent column count",
                line_idx + 1,
                path.display()
            );
        }
        for (col_idx, value) in row.into_iter().enumerate() {
            columns[col_idx].push(value);
        }
    }
    columns
}

pub fn assert_close_slice(actual: &[f64], expected: &[f64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "vector lengths differ: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let tol = ABS_TOL.max(REL_TOL * e.abs());
        assert!(
            diff <= tol,
            "index {idx}: |{a} - {e}| = {diff} exceeds tolerance {tol}"
        );
    }
}
