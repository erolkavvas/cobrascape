### assess_samples.py-------Erol Kavvas, SBRG, 2019
from cobrascape.species import load_json_obj, save_json_obj
from tqdm import tqdm
import pandas as pd
import os, sys, resource, warnings, statsmodels
from os import listdir
from os.path import isfile, join
resource.setrlimit(resource.RLIMIT_NOFILE, (10000,-1))
warnings.filterwarnings("ignore")  # sklearn gives hella warnings.
## math / machine learning
import numpy as np
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit #.fit
from statsmodels.tools import add_constant
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
from sklearn.feature_selection import chi2, SelectKBest, f_classif, mutual_info_classif

STRAIN_NUM = 375
ALLELE_NUM = 237
ACTION_NUM = 4
STANDARDIZE = True # Scale popFVA landscapes by Z-score (True) or Minmax (False)

ENSEMBLE_DIR = "ens_strains"+str(STRAIN_NUM)+"_alleles"+str(ALLELE_NUM)+"_actions"+str(ACTION_NUM)
if not os.path.exists(ENSEMBLE_DIR):
    raise ValueError('\t... directory "%s" does not exist' %s (ENSEMBLE_DIR))
else:
    print("dir ensemble: %s" % (ENSEMBLE_DIR))

POPFVA_SAMPLES_DIR = ENSEMBLE_DIR+"/popfva_samples/"
if not os.path.exists(POPFVA_SAMPLES_DIR):
    print('\t... directory "%s" does not exist' %s (POPFVA_SAMPLES_DIR))
    raise ValueError('\t... directory "%s" does not exist' %s (ENSEMBLE_DIR))
else:
    print("dir popfva samples: %s" % (POPFVA_SAMPLES_DIR))

### Create folders to save different types of sample assessments
if STANDARDIZE==True: 
    ENSEMBLE_MAP_ASSESS = ENSEMBLE_DIR+"/popfva_assessment_std/"
    ENSEMBLE_MAP_ANOVA = ENSEMBLE_DIR+"/popfva_anova_std/" ### Save ANOVA F-test enrichments.
else:
    ENSEMBLE_MAP_ASSESS = ENSEMBLE_DIR+"/popfva_assessment/"
    ENSEMBLE_MAP_ANOVA = ENSEMBLE_DIR+"/popfva_anova/" ### Save ANOVA F-test enrichments.

ENSEMBLE_MAP_COMPRESS = ENSEMBLE_DIR+"/popfva_compress/" ### Save numpy array versions of landscapes
for ens_folder in [ENSEMBLE_MAP_ASSESS, ENSEMBLE_MAP_ANOVA, ENSEMBLE_MAP_COMPRESS]:
    if not os.path.exists(ens_folder):
        print('\t... creating %s' % (ens_folder))
        os.makedirs(ens_folder)
    else:
        print("dir assess: %s" % (ens_folder))

### Load in the genetic variant matrix and AMR phenotypes for each case.
pheno_to_data2d_dict = {}
pheno_to_Y_dict = {}
pheno_list = ["ethambutol", "isoniazid", "rifampicin", "4-aminosalicylic_acid", 
             "pyrazinamide", "ethionamide","ofloxacin", "cycloserine"]
ALLELE_PHENO_FILE = ENSEMBLE_DIR+"/allele_pheno_data/"
for pheno_id in pheno_list:
    G_VARIANT_MATRIX_FILE = ALLELE_PHENO_FILE+"/allele_df_"+pheno_id+".csv"
    PHENO_MATRIX_FILE = ALLELE_PHENO_FILE+"/pheno_df_"+pheno_id+".csv"
    pheno_to_data2d_dict.update({pheno_id: pd.read_csv(G_VARIANT_MATRIX_FILE,index_col=0)})
    pheno_to_Y_dict.update({pheno_id: pd.read_csv(PHENO_MATRIX_FILE,index_col=0)[pheno_id]})## series


def compute_ANOVA_test(X1, y1):
    ANOVA_test = pd.DataFrame(list(f_classif(X1, y1)))
    ANOVA_test.columns = X1.columns
    ANOVA_test = ANOVA_test.sort_values([0, 1], axis=1, ascending=False).T
    ANOVA_test.columns = ["F_value", "pVal"]
    # usuing bonferroni correction
    rejected_list,pvalue_corrected_list,_,_ = statsmodels.sandbox.stats.multicomp.multipletests(
        ANOVA_test["pVal"], alpha=0.05, method='bonferroni', is_sorted=False)
    ANOVA_test["corrected_pVal"] = pvalue_corrected_list
    return ANOVA_test


### Save list of samples already assessed, so we can skip these and build a matrix for only the new ones.
# onlyfiles = [f for f in listdir(ENSEMBLE_MAP_ASSESS) if isfile(join(ENSEMBLE_MAP_ASSESS, f))]
# onlyfiles = [f for f in onlyfiles if f != ".DS_Store"]
# samplesBefore = [f for f in onlyfiles if "sample_" in f]
onlyfiles = [f for f in listdir(POPFVA_SAMPLES_DIR) if isfile(join(POPFVA_SAMPLES_DIR, f))]
onlyfiles = [f for f in onlyfiles if f != ".DS_Store"]
int_list = [int(x.split("_")[1]) for x in onlyfiles if len(x.split("_"))>2]
if len(int_list)>0:
    total_samples = max(int_list)+1
else:
    total_samples = 0
    print("\t... nothing in %s" % (POPFVA_SAMPLES_DIR))

### Run loop... should parallelize
ENSEMBLE_DATA_DICT = {}
SAMPLE_ANOVA_DICT = {}
for landscape_sample_num in tqdm(range(total_samples)[:]): #
    sample_id = "sampled_map_"+str(landscape_sample_num)
    ENSEMBLE_DATA_DICT[sample_id] = {}
    SAMPLE_ANOVA_DICT[sample_id] = {}
    
    if not os.path.exists(ENSEMBLE_MAP_ASSESS+"sample_"+str(landscape_sample_num)+"_map_assess.json"):
        variant_dec_file = POPFVA_SAMPLES_DIR+"sample_"+str(landscape_sample_num)+"_varDecision.json"

        if not os.path.exists(variant_dec_file):
            print('file "%s" does not exist' % (variant_dec_file))
        else:
            variant_dec_dict = load_json_obj(variant_dec_file)
            ENSEMBLE_DATA_DICT[sample_id].update(variant_dec_dict)

            fva_landscape_file = POPFVA_SAMPLES_DIR+"sample_"+str(landscape_sample_num)+"_FVA.json"
            fva_landscape_dict = load_json_obj(fva_landscape_file)
            ENSEMBLE_DATA_DICT[sample_id].update({"fva_landscape_file": fva_landscape_file})

            obj_val_list = {}
            for strain_id, strain_fva_dict in fva_landscape_dict.items(): # this has items b/c its already a dict
                obj_val_list[strain_id] = {}
                for rxn, max_min_dict in strain_fva_dict.items():
                    obj_val_list[strain_id].update({rxn+"_max":float(format(max_min_dict["maximum"],'.10f')), 
                                                    rxn+"_min":float(format(max_min_dict["minimum"],'.10f'))})
            fva_landscape_df = pd.DataFrame.from_dict(obj_val_list, orient="index")
            
            ### Save FVA landscape as saved numpy text files
            fva_compress_file = ENSEMBLE_MAP_COMPRESS+"sample_"+str(landscape_sample_num)+"_FVA.txt"
            if not os.path.exists(fva_compress_file):
                np.savetxt(fva_compress_file, fva_landscape_df.values, fmt='%g')

            ### TODO incorporate multiprocessing
            for AMR_drug, G_ALLELE_clustermap_data2d in pheno_to_data2d_dict.items():
                
                G_ALLELE_clustermap_data2d = pheno_to_data2d_dict[AMR_drug]
                Y_pheno_test = pheno_to_Y_dict[AMR_drug]
                Y_pheno_test_reindexed = Y_pheno_test.reindex(G_ALLELE_clustermap_data2d.index)
                fva_landscape_df_reindexed = fva_landscape_df.reindex(G_ALLELE_clustermap_data2d.index)

                ### Type of scaling to occur before fitting regularized logistic regression
                ### see link for discussion on which one to use:
                """https://stats.stackexchange.com/questions/69157/why-do-we-
                need-to-normalize-data-before-principal-component-analysis-pca"""
                if STANDARDIZE==True:
                    landscape_scaler = StandardScaler() # Standardization Z-score
                else:
                    landscape_scaler = MinMaxScaler() # Normalization 0-1 scaling

                G_FVA_clustermap_scaled = landscape_scaler.fit_transform(fva_landscape_df_reindexed)
                G_FVA_clustermap = pd.DataFrame(G_FVA_clustermap_scaled, 
                                                index=fva_landscape_df_reindexed.index, 
                                                columns=fva_landscape_df_reindexed.columns)

                X_standardscaled_SAMPLE = G_FVA_clustermap.reindex(G_ALLELE_clustermap_data2d.index)
                
                # set Y and X variable
                y = Y_pheno_test_reindexed.astype(int)
                y.dropna(axis=0, how='all', inplace=True)
                X = X_standardscaled_SAMPLE
                X = X.reindex(y.index)
                # remove collinearity through PCA
                pca = PCA(n_components=0.9, svd_solver = 'full', whiten=True)
                X_pca = pca.fit_transform(X)
                PCA_weight_dict = pd.DataFrame(pca.components_, 
                                               index=range(len(pca.components_)), 
                                               columns=X.columns).T.to_dict()
                
                ### Perform regularized logistic regression model
                ### NOTE from Hastie, Tibshirani, Friedman (The ridge solutions are not equivariant 
                ### ...under scaling of the inputs, and so one normally standardizes the inputs before solving.)
                lgit = Logit(y, add_constant(X_pca)).fit_regularized(maxiter=1500, disp=False, alpha=0.5)

                ENSEMBLE_DATA_DICT[sample_id].update({
                    "AIC_"+AMR_drug: lgit.aic,
                    "BIC_"+AMR_drug: lgit.bic,
                    "prsquared_"+AMR_drug: lgit.prsquared,
                    "loglikelihood_"+AMR_drug: lgit.llr,
                    "LLR_pval_"+AMR_drug: lgit.llr_pvalue,
                    "p_values_"+AMR_drug: str(lgit.pvalues.to_dict()),
                    "coefs_"+AMR_drug: str(lgit.params.to_dict()),
                    "std_err_"+AMR_drug: str(lgit.bse.to_dict()),
                    "PCA_comp_dict_"+AMR_drug: str(PCA_weight_dict)
                })
                del G_FVA_clustermap

                ### Perform associations between each FVA value and the AMR association... save result to seperate folder.
                X_anova = fva_landscape_df_reindexed
                X_anova = X.reindex(y.index)
                anova_df = compute_ANOVA_test(X_anova, y)
                anova_dict = anova_df.to_dict()
                SAMPLE_ANOVA_DICT[sample_id].update({AMR_drug: anova_dict})
                
            ### Save the Logistic regression fit for the sameple to ENSEMBLE_MAP_ASSESS
            save_json_obj(ENSEMBLE_DATA_DICT[sample_id], ENSEMBLE_MAP_ASSESS+"sample_"+str(landscape_sample_num)+"_map_assess.json")
            ENSEMBLE_DATA_DICT[sample_id] = {}
            ### Save the ANOVA F-testing results for the sameple to LANDSCAPE_MAP_ANOVA
            save_json_obj(SAMPLE_ANOVA_DICT[sample_id], ENSEMBLE_MAP_ANOVA+"sample_"+str(landscape_sample_num)+"_map_anova.json")
            SAMPLE_ANOVA_DICT[sample_id] = {}

print("...assessment of landscapes finished! ...")
