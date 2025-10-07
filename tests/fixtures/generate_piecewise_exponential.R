options(digits = 16, scipen = 999)

suppressPackageStartupMessages(library(simtrial))

output_dir <- "tests/fixtures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

write_matrix <- function(mat, filename) {
  path <- file.path(output_dir, filename)
  write.table(
    mat,
    file = path,
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
  )
}

write_fixture <- function(filename, seed, n, durations, rates) {
  fail_rate <- data.frame(duration = durations, rate = rates)

  set.seed(seed)
  uniforms <- runif(n)

  set.seed(seed)
  r_times <- simtrial::rpwexp(n = n, fail_rate = fail_rate)

  if (!all.equal(length(r_times), n)) {
    stop("Generated R sample length does not match request.")
  }

  data <- cbind(uniforms, r_times)
  write_matrix(data, filename)
}

write_fixture(
  filename = "pwexp_single_seed_123_n20.txt",
  seed = 123,
  n = 20,
  durations = c(1.0),
  rates = c(2.0)
)

write_fixture(
  filename = "pwexp_multi_seed_456_n30.txt",
  seed = 456,
  n = 30,
  durations = c(0.5, 0.5, 1.0),
  rates = c(1.0, 3.0, 10.0)
)
