# cobrascape

cobrascape is small package for constraint-based reconstruction and analysis (COBRA) of allelic variation, i.e., fitness landscapes. Specifically, cobrascape describes each strain in the genomics dataset as a strain-specific genome-scale model (GEM) parameterized according to the alleles it contains. It builds on the COBRApy package (https://github.com/opencobra/cobrapy) and is extensively utilized in estimating Metabolic Network Classifiers——a flux balance analysis-based machine learning classifier for microbial genome-wide association studies (GWAS) (preprint SOON).

### Primary Features of cobrascape
1. Allele-specific parameterization.
2. Simplified integration of strain variation with GEMs.
2. Population Flux Variability Analysis (popFVA)
3. Parallelized computation

![cobrafig](/cobrascape\_fig.png?raw=true)

Installation
	git clone https://github.com/erolkavvas/cobrascape.git

### Required data inputs for cobrascape
- `Genetic variant matrix`
  - shape: (strains, alleles), values: presence (1) or absence (0) of allele in strain.
- `Genome-scale metabolic model`, 
  - see http://bigg.ucsd.edu/ for a list of curated genome-scale models

see `example.ipynb` for basic use. Example Genetic variant matrix from (https://rdcu.be/9rHj) and genome-scale model from (https://rdcu.be/bG6JO)
