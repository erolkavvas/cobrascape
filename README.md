# cobrascape
Constraint-based modeling for allelic variation (COBRA for fitness landSCAPEs)

### Goals
1. Infer dominant evolutionary forces (metabolic selection pressures) from large-scale genomics-phenomics data (e.g., GWAS data).
2. Extend allele-phenotype associations to metabolic-phenotype associations.
3. Population Flux Variability Analysis (popFVA)

![cobrafig](/cobrascape\_fig.png?raw=true)

Installation
	git clone https://github.com/erolkavvas/cobrascape.git

### Required inputs
- `Genetic variant matrix`
  - shape: (strains, alleles)
  - values: presence (1) or absence (0) of allele in strain
- `Phenotype matrix`
  - shape: (strains, phenotypes)
  - values: presence (1) or absence (0) of phenotype in strain
	- Resistant (1) or susceptible (0) in case of _M. tuberculosis_ antimicrobial resistance (AMR) dataset
- `Genome-scale metabolic model`, 
  - a genome-scale metabolic model is a relational knowledge-base consisting of genes, reactions, metabolites that describe a particular microorganism. The phenotypic potential of a microorganism can be realized through constraint-based modeling (optimization) of a genome-scale metabolic model.
  - shape of (S)toichiometric matrix: (metabolites, reactions)
  - see http://bigg.ucsd.edu/ for a list of curated genome-scale models

### Primary scripts
Application to large-scale _M. tuberculosis_ antimicrobial resistance (AMR) dataset, as described in (https://rdcu.be/9rHj)
- `01_gen_ensemble.py`
  - Randomly samples allele-constraint maps and computes their corresponding 
  - Generates `Supplementary Data File 1`
- `02_ass_ensemble.py`
  - Generates descriptive data for the models sampled in  `01_gen_ensemble.py`
### TO DO
- Integrate population functionality with MEDUSA
