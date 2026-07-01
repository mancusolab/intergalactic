# intergalactic

`intergalactic` implements matrix-free operators for the same-haplotype cis-regulatory interaction kernel.

The package targets variance-component models over stacked phased haplotypes. If `H` contains adjacent maternal and paternal haplotype rows and `C` sums each pair back to one diploid individual, the interaction component is

```text
K_H = C Phi(H) Phi(H)^T C^T
```

where `Phi(h) = h ⊗ h` is applied to each physical haplotype. This is not genotype-level GxG:

```text
Phi(C H) Phi(C H)^T
```

because genotype-level GxG includes cross-haplotype maternal-by-paternal terms. Those terms are not the target for cis-regulatory grammar.

See [docs/covariance-model.md](docs/covariance-model.md) for the covariance model and operator identities.
