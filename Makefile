.PHONY: setup pkpd glucose icu ablation all clean

JULIA = julia --project

# Install dependencies
setup:
	$(JULIA) -e 'using Pkg; Pkg.instantiate()'

# Individual experiments
pkpd:
	$(JULIA) experiments/pkpd/run_benchmarks.jl

glucose:
	$(JULIA) experiments/glucose/run_benchmarks.jl

icu:
	$(JULIA) experiments/icu/run_benchmarks.jl

# Ablation study (glucose only)
ablation:
	$(JULIA) experiments/glucose/run_ablation.jl

# Run everything
all: pkpd glucose icu ablation

# Clean generated outputs
clean:
	find experiments -name "results" -type d -exec rm -rf {} + 2>/dev/null || true
