# Reference data fixtures

The `.txt` files in this directory contain reference outputs from the original
simtrial R implementation. They are used to validate the Rust port of the
piecewise exponential generator.

## Regenerating fixtures

The fixture generator script reproduces the draws using the R implementation
for specific seeds and parameter sets, and writes both the uniform random
numbers and the resulting event times to plain-text files without headers.

```sh
Rscript tests/fixtures/generate_piecewise_exponential.R
```

The Rust test suite consumes these numbers to cross-check the
Rust implementation against the reference algorithm.
