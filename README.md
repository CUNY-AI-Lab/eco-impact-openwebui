# registry branch

Data only. `eco_models.json` here is rebuilt every Monday by
[`weekly-registry.yml`](../../blob/main/.github/workflows/weekly-registry.yml)
from CAIL Gateway's public catalog, EcoLogits, and Hugging Face configs, and the
Eco Impact Estimator filter fetches it from this branch.

Don't edit it by hand: fix parameters in `scripts/eco-extra-patterns.json` on
`main` and re-run the workflow.
