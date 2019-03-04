### gen_ensemble.py: code for generating an ensemble of GEM population models.
import cobrascape.species as cs
from cobra.io import load_json_model
import pandas as pd
import os
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (10000,-1))

DATA_DIR = "cobra_model/"
MODEL_FILE = "iEK1011_drugTesting_media.json"
X_ALLELES_FILE = "x_strain_ID_TB.csv"
Y_PHENOTYPES_FILE = "y_drug_TB.csv"
MODEL_SAMPLES_FILENAME="base_flux_samples.csv" # OPTIONAL. If not provided, script will perform flux sampling.

FVA_frac_opt = 0.1 # Decides the minimum flux required through biomass production
FVA_pfba_fract = 1.5 # Decides allowable flux space based on upper bounding the total sum of fluxes
action_num = 4
ADD_NA_BOUND = False
NUM_SAMPLES = 5000

X_species = pd.read_csv(DATA_DIR+X_ALLELES_FILE, index_col = 0)
Y_phenotypes = pd.read_csv(DATA_DIR+Y_PHENOTYPES_FILE, index_col = 0)

Y_pheno = Y_phenotypes[Y_phenotypes["4-aminosalicylic_acid"].notnull()]
X_species = X_species.reindex(Y_pheno.index).dropna()
X_species_final = X_species.iloc[:]
Y_pheno_final = Y_pheno.loc[X_species_final.index]
print("input: (G)enetic variant matrix= (strains: %d, alleles: %d)" % (X_species_final.shape[0], X_species_final.shape[1]))

COBRA_MODEL = load_json_model(DATA_DIR+MODEL_FILE)
print("input: (S)toichimetric genome-scale model= (genes: %d, reactions: %d, metabolites: %d)" % (len(COBRA_MODEL.genes), 
    len(COBRA_MODEL.reactions), len(COBRA_MODEL.metabolites)))

### Initialize media condition of model
COBRA_MODEL.medium = {
    'EX_asn_L': 1.0,
    'EX_leu_L': 1.0,
    'EX_ile_L': 1.0,
    'EX_his_L': 1.0,
    'EX_asp_L': 1.0,
    'EX_ala_L': 1.0,
    'EX_arg_L': 1.0,
    'EX_glu_L': 1.0,
    'EX_ca2': 1000.0,
    'EX_cit': 1.0,
    'EX_cl': 1000.0,
    'EX_fe3': 1.0,
    'EX_glyc': 1.0,
    'EX_h': 1000,
    'EX_mg2': 1000.0,
    'EX_nh4': 20.0,
    'EX_o2': 20.0,
    'EX_pi': 1000,
    'EX_so4': 1000,
    'EX_etoh': 1,
    'EX_mobd': 1000
}

sol = COBRA_MODEL.optimize()
print("\t... before cleaning (objective_value: %f, EMB flux: %f)" % (sol.objective_value, sol.fluxes["EMB"]))
### Clean base model and apply FVA constriants
COBRA_MODEL = cs.clean_base_model(COBRA_MODEL, open_exchange=True, verbose=False)
sol = COBRA_MODEL.optimize()
print("\t... after cleaning (objective_value: %f, EMB flux: %f)" % (sol.objective_value, sol.fluxes["EMB"]))
COBRA_MODEL, fva_df = cs.init_fva_constraints(COBRA_MODEL,opt_frac=FVA_frac_opt, pfba_fact=FVA_pfba_fract, verbose=False)
sol = COBRA_MODEL.optimize()
print("\t... after fva constraints (objective_value: %f, EMB flux: %f)" % (sol.objective_value, sol.fluxes["EMB"]))
print("\t... filtered GEM= (genes: %d, reactions: %d, metabolites: %d)" % (len(COBRA_MODEL.genes), 
    len(COBRA_MODEL.reactions), len(COBRA_MODEL.metabolites)))

### Create Species object
SPECIES_MODEL = cs.Species("TB_species")
COBRA_MODEL.solver = "glpk"
SPECIES_MODEL.base_cobra_model = COBRA_MODEL
SPECIES_MODEL.load_from_matrix(X_species_final, filter_model_genes=True, allele_gene_sep="_")
for allele in SPECIES_MODEL.alleles:
    allele.cobra_gene = allele.id.split("_")[0]
    
### Get AMR genes of interest
amr_gene_df = pd.read_csv(DATA_DIR+"gene_list_OLD.csv",index_col=0)
amr_gene_list = amr_gene_df.index.tolist()
print(len(amr_gene_list))
gene_list = amr_gene_list # =["Rv1908c", "Rv2245", "Rv1483"]

players, player_reacts, player_metabs = cs.get_gene_players(gene_list, SPECIES_MODEL, verbose=True)
### Update the strains... takes long time... ensure that 
SPECIES_MODEL.update_strains_cobra_model()

ENSEMBLE_DIR = "ens_strains"+str(len(SPECIES_MODEL.strains))+"_alleles"+str(len(players))+"_actions"+str(action_num)
POPFVA_SAMPLES_DIR = ENSEMBLE_DIR+"/popfva_samples/"
print("output dir: %s" % (ENSEMBLE_DIR))
if not os.path.exists(POPFVA_SAMPLES_DIR):
    print('\t... creating sampling directory:'+POPFVA_SAMPLES_DIR)
    os.makedirs(POPFVA_SAMPLES_DIR)
    
MODEL_SAMPLES_FILE = ENSEMBLE_DIR+"/"+MODEL_SAMPLES_FILENAME
if not os.path.exists(MODEL_SAMPLES_FILE):
    from cobra import flux_analysis
    print("\t... generating flux samples for base cobra model...(may take >10 minutes)")
    rxn_flux_samples_ARCH = flux_analysis.sample(COBRA_MODEL, 10000,method='achr', 
                                      thinning=100, processes=6, seed=None)
    print("\t... saving flux samples for base cobra model: ", MODEL_SAMPLES_FILE)
    rxn_flux_samples_ARCH.to_csv(MODEL_SAMPLES_FILE)

ENSEMBLE_BASEMODEL_FILE = ENSEMBLE_DIR+"/base_cobra_model.json"
if not os.path.exists(ENSEMBLE_BASEMODEL_FILE):
    from cobra.io import save_json_model
    print("\t... saving base cobra model: ", ENSEMBLE_BASEMODEL_FILE)
    save_json_model(COBRA_MODEL,ENSEMBLE_BASEMODEL_FILE)
    
base_flux_samples = pd.read_csv(MODEL_SAMPLES_FILE,index_col=0)

### Save genetic variant matrix and AMR phenotypes for each case.
allele_list = [x.id for x in players]
pheno_list = ["ethambutol", "isoniazid", "rifampicin", "4-aminosalicylic_acid", 
             "pyrazinamide", "ethionamide","ofloxacin", "cycloserine"]

ALLELE_PHENO_FILE = ENSEMBLE_DIR+"/allele_pheno_data/"
if not os.path.exists(ALLELE_PHENO_FILE):
    print('\t... creating sampling directory:'+ALLELE_PHENO_FILE)
    os.makedirs(ALLELE_PHENO_FILE)
for pheno_id in pheno_list:
    G_VARIANT_MATRIX_FILE = ALLELE_PHENO_FILE+"/allele_df_"+pheno_id+".csv"
    PHENO_MATRIX_FILE = ALLELE_PHENO_FILE+"/pheno_df_"+pheno_id+".csv"
    X_filtered, Y_filtered = cs.filter_pheno_nan(X_species_final, Y_pheno_final, pheno_id)
    if not os.path.exists(G_VARIANT_MATRIX_FILE):
        X_filtered.loc[:,allele_list].to_csv(G_VARIANT_MATRIX_FILE)
    if not os.path.exists(PHENO_MATRIX_FILE):
        pd.DataFrame(Y_filtered).to_csv(PHENO_MATRIX_FILE) # , header=True

### --- Generate ensemble of random allele-constraint maps and their corresponding popFVA landscapes
pool_obj = cs.sample_species(SPECIES_MODEL,POPFVA_SAMPLES_DIR,players,base_flux_samples, 
                             rxn_set=player_reacts,samples_n=NUM_SAMPLES,fva=True,fva_frac_opt=FVA_frac_opt,
                             action_n=action_num,add_na_bound=ADD_NA_BOUND)