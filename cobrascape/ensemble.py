# coding: utf-8
### ---------------------------------------------------
### --- CobraScape_samples.py  ------------------------
### --- Erol Kavvas, SBRG, 2018  ----------------------
### ---------------------------------------------------
### "CobraScape_samples.py" provides a class object and
### numerous functions for simplifying the statistical
### analysis of the generated model ensemble.
### ---------------------------------------------------
from tqdm import tqdm
import numpy as np
import pandas as pd
from os import listdir, path
import ast

### Cobra
from cobra.io import load_json_model
from cobrascape.species import load_json_obj, save_json_obj, create_action_set
from sklearn.feature_selection import f_classif
from statsmodels.sandbox.stats.multicomp import multipletests

from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from collections import Counter
from sklearn.utils import Bunch

class Samples(object):
    """ Class object of holding information about the ensemble for analysis of 
        the sampled allele-constraint maps and their resulting popFVA landscapes.
    """
    def __init__(self):
        self.assess_df = pd.DataFrame()
        self.constraint_df = pd.DataFrame()
        self.anova_dict = {}
        self.y_pheno_dict = {}
        self.x_allele_dict = {}
        self.base_cobra_model = None
        self.pheno_list = []
        self.hq_models = pd.DataFrame()
        self.model_weights = pd.DataFrame()
        self.signif_pca = pd.DataFrame()
        self.assess_file_loc = None
        
        
    def get_hq_samples(self, ic_id="AIC", pheno_id="isoniazid", thresh_val=7):
        """ Returns Series of high quality samples along with the criteria value
        """
        if ic_id == "cv_score_mean":
            max_ic = self.assess_df[ic_id+"_"+pheno_id].max()
            min_ic = max_ic - thresh_val
            hq_models = self.assess_df[ic_id+"_"+pheno_id][self.assess_df[ic_id+"_"+pheno_id]>min_ic]
        else:
            min_ic = self.assess_df[ic_id+"_"+pheno_id].min()
            max_ic = min_ic + thresh_val
            hq_models = self.assess_df[ic_id+"_"+pheno_id][self.assess_df[ic_id+"_"+pheno_id]<max_ic]
        hq_models.sort_values(inplace=True)
        self.hq_models = hq_models
        return hq_models


    def get_ic_weights(self, pheno_id="isoniazid", ic_id="BIC", sample_all=True):
        """ Returns Series of high quality samples along with the criteria value
        """
        if sample_all==True:
            hqmodels_obj = self.get_hq_samples(ic_id=ic_id, pheno_id=pheno_id, thresh_val=2000) # in all samples
        else:
            hqmodels_obj = self.get_hq_samples(ic_id=ic_id, pheno_id=pheno_id, thresh_val=40) # only the top
        hqmodels_obj = hqmodels_obj-min(hqmodels_obj)
        hqmodels_obj_sum = sum([np.exp(x/-2) for x in hqmodels_obj])
        hqmodels_obj = hqmodels_obj.map(lambda x: np.exp(x/-2)/hqmodels_obj_sum)
        self.model_weights = hqmodels_obj
        return hqmodels_obj
    
    
    def popFVA_enriched(self, ic_id="AIC", pheno_id="isoniazid", thresh_val=7, FDR_filter=False):
        """ Mann-whitney U test for whether a particular FVA objective has higher associations
            in high quality models (low AIC) than the other models (non-low AIC)
        """
        if self.hq_models.empty == True:
            self.get_hq_samples(ic_id, pheno_id, thresh_val)
            
        pheno_id = self.hq_models.name.replace("cv_score_mean_","").replace("AIC_","").replace("BIC_","")
        hq_sample_ids = set(self.hq_models.index)
        lq_sample_ids = set(self.assess_df.index) - hq_sample_ids
        s_df = -np.log(self.anova_dict[pheno_id])
        s_df.fillna(1, inplace=True)
        
        feat_to_test_dict = {}
        for feat in s_df.columns:
            if len(s_df[feat].unique())!=1:
                X = s_df.loc[list(hq_sample_ids), feat]
                Y = s_df.loc[list(lq_sample_ids), feat]
                if X.shape[0]!=0:
                    U_stat, pval = stats.mannwhitneyu(X, Y, use_continuity=True, alternative='greater') 
                else:
                    U_stat, pval = 1, 1
                feat_to_test_dict.update({feat: {"stat": U_stat, "pvalue": pval}})
                
        feat_to_test_df = pd.DataFrame.from_dict(feat_to_test_dict, orient="index")
        feat_to_test_df.sort_values("pvalue", ascending=True, inplace=True)
        if FDR_filter == True:
            feat_to_test_df = FDR(feat_to_test_df,fdr_rate=.1)
        return feat_to_test_df
    
    
    def get_logreg_params(self, pval_thresh=5e-2, hq_mods=pd.DataFrame()):
        """ Gets significant PCA components
        """
        signif_pca_comps, signif_pca_react_loads = {}, {}
        if self.hq_models.empty == True and hq_mods.empty==True:
            print("Give list of models or run -get_hq_samples")
            return None
        elif hq_mods.empty!=True:
            self.hq_models = hq_mods
            
        pheno_id = "_".join(self.hq_models.name.split("_")[1:])
        for sampled_map_num, ic_val in tqdm(self.hq_models.items()):
            landscape_sample_num = sampled_map_num.split("_")[-1]
            sample_id = "sample_"+landscape_sample_num+"_map_assess.json"
            landscape_assess_sample_file = self.assess_file_loc+sample_id
            if path.exists(landscape_assess_sample_file):
                landscape_assess_sample = load_json_obj(landscape_assess_sample_file)
                pval_dict = ast.literal_eval(landscape_assess_sample["p_values_"+pheno_id].replace("nan", "1.0"))
                coef_dict = ast.literal_eval(landscape_assess_sample["coefs_"+pheno_id])
                comp_dict = ast.literal_eval(landscape_assess_sample["PCA_comp_dict_"+pheno_id])
                comp_dict = {"x"+str(k+1):v for k, v in comp_dict.items() }
                signif_pca_comps[sampled_map_num] = {}
                for pca_comp, p_val in pval_dict.items():
                    if p_val < pval_thresh and pca_comp!="const": # and abs(coef_dict[pca_comp])>0.5:
                        signif_pca_comps[sampled_map_num].update({pca_comp: {
                            "p_val": p_val, 
                            "coef": coef_dict[pca_comp],
                            "pca_load":comp_dict[pca_comp]}})
        self.signif_pca = signif_pca_comps
        return signif_pca_comps
                        
        
    def get_sample_pca(self, sample_id, pca_thresh=0.0, pca_comp_id=None):
        """ Returns a dataframe of (popFVA features, pca components) for a particular sample.
            pca_thresh: the cutoff value for a popFVA feature in a pca component.
            pca_comp_id: decides which component the dataframe will be sorted by.
        """
        if len(self.signif_pca.keys())==0:
            self.get_logreg_params()

        sample_logreg_df = pd.DataFrame.from_dict(self.signif_pca[sample_id]).T
        sample_logreg_df = sample_logreg_df.sort_values(["p_val"])
        if pca_comp_id==None:
            top_pca_comp = sample_logreg_df.index[0]
        else:
            top_pca_comp = pca_comp_id

        pca_comp_df = pd.DataFrame()
        for pca_comp, pca_row in sample_logreg_df.iterrows():
            comp_df = pd.DataFrame.from_dict(pca_row["pca_load"], orient="index") # orient=pca_row["pca_load"].keys()
            comp_df.columns = [pca_comp]
            pca_comp_df = pd.concat([pca_comp_df, comp_df], axis=1)

        pca_comp_df = pca_comp_df.T
        pca_comp_df["coef"] = pca_comp_df.index.map(lambda x: self.signif_pca[sample_id][x]["coef"])
        pca_comp_df["p_val"] = pca_comp_df.index.map(lambda x: self.signif_pca[sample_id][x]["p_val"])
        pca_comp_df.sort_values(["p_val"], inplace=True)
        pca_comp_df = pca_comp_df.T
        pca_comp_df.drop(["coef", "p_val"], axis=0, inplace=True)
        pca_comp_filt = pca_comp_df[abs(pca_comp_df)>pca_thresh]
        pca_comp_filt.dropna(how="all", inplace=True)
        pca_comp_filt.sort_values([top_pca_comp], inplace=True)
        return pca_comp_filt
        
        
    def get_alleles_LOR(self, allele_list, pheno_id="isoniazid", addval=0.5):
        """Takes in a list of alleles and returns the log odds ratio of each allele occurance with a phenotype
        """
        drug_allele_df = filter_0_alleles(self.x_allele_dict[pheno_id].copy())
        LOR_list, num_R_list, num_strains_list, perc_R_list = [], [], [], []
        for x_allele in allele_list:
            strains_with_allele = drug_allele_df[drug_allele_df[x_allele]==1].index.tolist()
            allele_resist_percent = round(resist_percentage(self.y_pheno_dict[pheno_id], strains_with_allele), 4)
            LOR, num_R = log_odds_ratio(x_allele, drug_allele_df, self.y_pheno_dict[pheno_id], addval=addval)
            LOR_list.append(LOR)
            num_R_list.append(num_R)
            num_strains_list.append(len(strains_with_allele))
            perc_R_list.append(allele_resist_percent)
        return LOR_list,num_R_list,num_strains_list,perc_R_list


    def load_ensemble_data(self, STRAIN_NUM=375,ALLELE_NUM=237,ACTION_NUM=4, 
                           pheno_list = ["ethambutol", "isoniazid", "rifampicin", "4-aminosalicylic_acid", 
                                         "pyrazinamide", "ethionamide","ofloxacin", "cycloserine"], 
                           STANDARDIZE=False, test_set=True):
        """ Loads in the data describing a particular ensemble
        """
        ENSEMBLE_DIR = "ens_strains"+str(STRAIN_NUM)+"_alleles"+str(ALLELE_NUM)+"_actions"+str(ACTION_NUM)
        if not path.exists(ENSEMBLE_DIR):
            raise ValueError('\t... directory "%s" does not exist' %s (ENSEMBLE_DIR))
        else:
            print("dir ensemble: %s" % (ENSEMBLE_DIR))

        POPFVA_SAMPLES_DIR = ENSEMBLE_DIR+"/popfva_samples/"
        if not path.exists(POPFVA_SAMPLES_DIR):
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
            ENSEMBLE_MAP_ANOVA = ENSEMBLE_DIR+"/popfva_anova/"

        self.assess_file_loc = ENSEMBLE_MAP_ASSESS
        ENSEMBLE_MAP_COMPRESS = ENSEMBLE_DIR+"/popfva_compress/" ### Save numpy array versions of landscapes

        ENSEMBLE_BASEMODEL_FILE = ENSEMBLE_DIR+"/base_cobra_model.json"
        COBRA_MODEL = load_json_model(ENSEMBLE_BASEMODEL_FILE)
        self.base_cobra_model = COBRA_MODEL

        ### -------------- LOAD 1 -----------------
        print("(1) load COBRA_MODEL, s, pheno_to_data2d_dict, pheno_to_Y_dict ...")
        ### Load in the genetic variant matrix and AMR phenotypes for each case.
        pheno_to_data2d_dict = {}
        pheno_to_Y_dict = {}
        ALLELE_PHENO_FILE = ENSEMBLE_DIR+"/allele_pheno_data/"
        for pheno_id in pheno_list:
            G_VARIANT_MATRIX_FILE = ALLELE_PHENO_FILE+"/allele_df_"+pheno_id+".csv"
            PHENO_MATRIX_FILE = ALLELE_PHENO_FILE+"/pheno_df_"+pheno_id+".csv"
            pheno_to_data2d_dict.update({pheno_id: pd.read_csv(G_VARIANT_MATRIX_FILE,index_col=0)})
            pheno_to_Y_dict.update({pheno_id: pd.read_csv(PHENO_MATRIX_FILE,index_col=0)[pheno_id]})## to make Series
        
        self.x_allele_dict = pheno_to_data2d_dict
        self.y_pheno_dict = pheno_to_Y_dict
        self.pheno_list = pheno_list

        ### -------------- LOAD 2 -----------------
        print("(2) load SAMPLES_ASSESS_DF ...")
        onlyfiles = [f for f in listdir(ENSEMBLE_MAP_ASSESS) if path.isfile(path.join(ENSEMBLE_MAP_ASSESS, f))]
        onlyfiles = [f for f in onlyfiles if f != ".DS_Store"]
        if test_set==True:
            samplesAfter = [f for f in onlyfiles if "sample_" in f][:20]# only get 20 sample so files are small
        else:
            samplesAfter = [f for f in onlyfiles if "sample_" in f] 

        wanted_keys = []
        for pheno_id in pheno_list:
            wanted_keys.extend(["AIC_"+pheno_id, "BIC_"+pheno_id, "prsquared_"+pheno_id, 
                                "loglikelihood_"+pheno_id, "LLR_pval_"+pheno_id, "cv_score_mean_"+pheno_id])

        SAMPLES_ASSESS_DF = {}
        for landscape_sample_name in tqdm(samplesAfter):
            landscape_sample_num = landscape_sample_name.split("_")[1]
            sample_id = "sampled_map_"+str(landscape_sample_num)
            landscape_assess_sample_file = ENSEMBLE_MAP_ASSESS+landscape_sample_name

            if path.exists(landscape_assess_sample_file):
                landscape_assess_sample = load_json_obj(landscape_assess_sample_file)
                SAMPLES_ASSESS_DF[sample_id] = {}
                SAMPLES_ASSESS_DF[sample_id].update(dict((k, landscape_assess_sample[k]) for k in wanted_keys if k in landscape_assess_sample))

        # transform to pandas dataframe
        SAMPLES_ASSESS_DF = pd.DataFrame.from_dict(SAMPLES_ASSESS_DF,orient="index")
        print("\t... SAMPLES_ASSESS_DF shape: (samples: %d, assess_cols: %d)" % (SAMPLES_ASSESS_DF.shape[0], SAMPLES_ASSESS_DF.shape[1]))
        self.assess_df = SAMPLES_ASSESS_DF

        ### -------------- LOAD 3 -----------------
        print("(3) load SAMPLES_ANOVA_DICT ...")
        SAMPLES_ANOVA_DF = {}
        for pheno_id in pheno_list:
            SAMPLES_ANOVA_DF[pheno_id] = {}

        for landscape_sample_name in tqdm(samplesAfter[:]):
            landscape_sample_num = landscape_sample_name.split("_")[1]
            sample_id = "sample_"+landscape_sample_num+"_map_anova.json"
            landscape_anova_sample_file = ENSEMBLE_MAP_ANOVA+sample_id

            if path.exists(landscape_anova_sample_file):
                landscape_anova_sample = load_json_obj(landscape_anova_sample_file)
                
                for pheno_id in pheno_list:
                    SAMPLES_ANOVA_DF[pheno_id]["sampled_map_"+landscape_sample_num] = {}
                    SAMPLES_ANOVA_DF[pheno_id]["sampled_map_"+landscape_sample_num].update(landscape_anova_sample[pheno_id]["pVal"])

        print("\t... generating SAMPLES_ANOVA_DICT")
        SAMPLES_ANOVA_DICT = {}
        for pheno_id in tqdm(pheno_list):
            SAMPLES_ANOVA_DICT.update({pheno_id: pd.DataFrame.from_dict(SAMPLES_ANOVA_DF[pheno_id],orient="index")})
        
        self.anova_dict = SAMPLES_ANOVA_DICT

        ### -------------- LOAD 3 -----------------
        print("(4) load SAMPLES_AC_DF ...")
        allele_col_ids = [x for x in pheno_to_data2d_dict[pheno_list[0]].columns]

        SAMPLES_AC_DF = {}
        for landscape_sample_name in tqdm(samplesAfter):
            landscape_sample_num = landscape_sample_name.split("_")[1]
            sample_id = "sampled_map_"+str(landscape_sample_num)
            landscape_assess_sample_file = ENSEMBLE_MAP_ASSESS+landscape_sample_name

            if path.exists(landscape_assess_sample_file):
                landscape_assess_sample = load_json_obj(landscape_assess_sample_file)
                SAMPLES_AC_DF[sample_id] = {}
                SAMPLES_AC_DF[sample_id].update(dict((k, landscape_assess_sample[k]) for k in allele_col_ids if k in landscape_assess_sample))

        SAMPLES_AC_DF = pd.DataFrame.from_dict(SAMPLES_AC_DF,orient="index")
        print("\t... SAMPLES_AC_DF shape: (samples: %d, assess_cols: %d)" % (SAMPLES_AC_DF.shape[0], SAMPLES_AC_DF.shape[1]))
        self.constraint_df = SAMPLES_AC_DF
        


def compute_ANOVA_test(X1,y1,correction_test=False,correct_alpha=0.05):
    """ returns ANOVA_test (X1 columns vs f_value, p-value, etc.)
    """
    ANOVA_test = pd.DataFrame(list(f_classif(X1, y1)))
    ANOVA_test.columns = X1.columns
    ANOVA_test = ANOVA_test.sort_values([0, 1], axis=1, ascending=False).T
    ANOVA_test.columns = ["F_value", "pvalue"]
    ANOVA_test["value_counts"] = ANOVA_test.index.map(lambda x: Counter(X1[x].values).most_common())
    if correction_test!=False:
        rejected_list, pvalue_corrected_list, alphaC, alphacBonf = multipletests(
            ANOVA_test["pvalue"], alpha=correct_alpha, method='bonferroni', is_sorted=False)
        ANOVA_test_corrected = ANOVA_test[rejected_list].copy()
        ANOVA_test_corrected["corrected_pVal"] = pvalue_corrected_list[rejected_list]
        return ANOVA_test_corrected
    else:
        return ANOVA_test

        
def FDR(p_values,fdr_rate=.01):
    """False discovery rate boiii
    """
    sorted_vals = p_values.sort_values('pvalue')
    m = len(p_values)
    ranks = range(1,m+1)
    crit_vals = np.true_divide(ranks,m)*fdr_rate
    sig = (sorted_vals.pvalue < crit_vals)
    if len(np.argwhere(sig)) == 0:
        return pd.DataFrame(columns=['log_OR','pvalue','precision','recall','TP'])
    else:
        thresh = np.argwhere(sig)[-1][0]
        final_vals = sorted_vals[:thresh+1]
        return final_vals.sort_values('pvalue',ascending=True)

    
def get_rxn_alleles(rxn, mod, ac_df):
    """plot_alleles = get_rxn_alleles("DCPT", COBRA_MODEL, SAMPLES_AC_DF)
    """
    rxn_gem_obj = mod.reactions.get_by_id(rxn)
    rxn_gem_gene_list = [x.id for x in list(rxn_gem_obj.genes)]
    rxn_alleles = []
    for g_all in ac_df.columns:
        g_ = g_all.split("_")[0]
        if g_ in rxn_gem_gene_list:
            rxn_alleles.append(g_all)
    return rxn_alleles

    
def get_gene_alleles(gene_id, ac_df):
    """plot_alleles = get_gene_alleles("Rv1908c", SAMPLES_AC_DF)
    """
    g_alleles = []
    for g_all in ac_df.columns:
        g_ = g_all.split("_")[0]
        if g_ == gene_id:
            g_alleles.append(g_all)
    return g_alleles


def resist_percentage(resistance_data, list_of_strains):
    return resistance_data.loc[list_of_strains].sum()/float(len(resistance_data.loc[list_of_strains].index))

    
def log_odds_ratio(allele_, allele_df, pheno_df, addval=0.5):
    """Return the log odds ratio of the allele penetrance with the AMR phenotype.
    """
    allele_df["pheno"] = pheno_df
    presence_R = float(len(allele_df[(allele_df[allele_]==1)&(allele_df["pheno"]==1)].index))
    presence_S = float(len(allele_df[(allele_df[allele_]==1)&(allele_df["pheno"]==0)].index))
    absence_R = float(len(allele_df[(allele_df[allele_]==0)&(allele_df["pheno"]==1)].index))
    absence_S = float(len(allele_df[(allele_df[allele_]==0)&(allele_df["pheno"]==0)].index))
    num_R = presence_R
    if presence_R==0 or presence_S==0 or absence_R==0 or absence_S==0:
        presence_R+=addval
        presence_S+=addval
        absence_R+=addval
        absence_S+=addval
        
    odds_ratio = (presence_R/presence_S)/(absence_R/absence_S)
    LOR = np.log(odds_ratio)
    return LOR, num_R


def filter_0_alleles(allele_df):
    """Drop alleles that do not appear in any of the strains.
    """
    drop_cols = []
    for col in allele_df.columns:
        if allele_df[col].sum()<5:
            drop_cols.append(col)

    allele_df.drop(drop_cols, inplace=True, axis=1)
    return allele_df


def get_action_constraint_mapping(action_number):
    """action_constraint_mapping = get_action_constraint_mapping(action_number)
    """
    actions = create_action_set(number_of_actions=action_number, add_no_change=True)
    lb_list = ["lb_"+str(x) for x in range(0, action_number/2, 1)]
    ub_list = ["ub_"+str(x) for x in range(0, action_number/2, 1)]
    action_list = lb_list + ["no_change"] + ub_list
    action_ord_list = range(-action_number/2, action_number/2+1)
    action_constraint_mapping = dict(zip(tuple(action_list), tuple(action_ord_list)))
    return action_constraint_mapping


def get_LOR_colors(LOR_list, min_max=(-2, 2)):
    """Use Log Odds Ratios list to create color map for allele columns
    """
    cmap = cm.coolwarm
    if min_max!=False:
        norm = Normalize(vmin=min_max[0], vmax=min_max[1])
    else:
        bnd_Val = max(abs(min(LOR_list)), abs(max(LOR_list)))
        print("min(LOR_list), max(LOR_list): ",min(LOR_list), max(LOR_list))
        norm = Normalize(vmin=-bnd_Val, vmax=bnd_Val)
        
    allele_color_list = [cmap(norm(x)) for x in LOR_list]
    return allele_color_list


# allele_color_list = get_LOR_colors(LOR_list)
### 2 functions below help out with analyzing interactions amongst constraints
def rxn_to_constraints_samples_ids(player_list, action_list, samps, reacter, base_mod):
    """ I should remove reacter
    Parameters
    ----------
    player_list = ["Rv1908c", "Rv1484", ...]
    """
    allele_rxns_constraint_dict = {}
    for all_player in player_list:
        allele_rxns_constraint_dict[all_player] = {}
        react_ids = [x.id for x in base_mod.genes.get_by_id(all_player).reactions]
        for react in react_ids:
            allele_rxns_constraint_dict[all_player][react] = {}
            max_flux, min_flux = max(samps[react]), min(samps[react])
            mean_flux = np.mean(samps[react])
    
            action_to_constraints_dict = {}
            # for reactions that can't have any change, keep their bounds at a single value.
            if max_flux == min_flux: 
                for a in action_list:
                    action_to_constraints_dict.update({a: max_flux})
            else:
                left_bound_distance = mean_flux - min_flux

                gradient_steps = len(action_list)/2
                min_to_mean_grad = np.arange(min_flux, mean_flux, (mean_flux-min_flux)/gradient_steps)
                max_to_mean_grad = np.arange(mean_flux, max_flux, (max_flux-mean_flux)/gradient_steps)

                for a in action_list:
                    if a == "no_change":
                        action_to_constraints_dict.update({a: 0})
                    else:
                        dec_or_inc = a.split("_")[0]
                        grad_dist = int(a.split("_")[1])
                        # It doesn't matter if mean_flux is less than or greater than 0.
                        if dec_or_inc == "lb": # Change upper_bound
                            action_to_constraints_dict.update({a: min_to_mean_grad[grad_dist]})
                        elif dec_or_inc == "ub": # Change lower_bound
                            action_to_constraints_dict.update({a: max_to_mean_grad[grad_dist]})
            allele_rxns_constraint_dict[all_player][react].update(action_to_constraints_dict)
        
    return allele_rxns_constraint_dict


def get_ac_interactions(COBRA_MODEL, gene_rxn_action_dict, x_gene_rxn, y_gene_rxn):
    """Generates a 2 by 2 heatmaps of allele constraint interactions for min/max optimizations
        of corresponding allele-encoded metabolic reactions.
    input: 
        COBRA_MODEL, x_gene_rxn, y_gene_rxn
        gene_rxn_action_dict: 
    output:
        f: figure of 2 by 2 heatmaps
        payoff_df_dict: dictionary of heatmap values for each optimized objective
    -- example: f, payoff_dict = get_ac_interactions(COBRA_MODEL, ['Rv1908c','CAT'], ['Rv3280','ACCC'])
    """
    f, (ax1, ax2) = plt.subplots(2, 2, figsize=(8,6),sharex=True, sharey=True)
    interact_landscape_list = []
    interact_landscape_dict = {}
    COBRA_MODEL_COPY = COBRA_MODEL.copy()
    payoff_df_dict = {}
    for obj_id, ax in [(x_gene_rxn[1], ax1), (y_gene_rxn[1], ax2)]:
        for obj_dir, ax_col in [("max", ax[0]),("min", ax[1])]:
            for x_action, x_constraint in gene_rxn_action_dict[x_gene_rxn[0]][x_gene_rxn[1]].items():
                for y_action, y_constraint in gene_rxn_action_dict[y_gene_rxn[0]][y_gene_rxn[1]].items():
                    with COBRA_MODEL_COPY:   
                        COBRA_MODEL_COPY.objective = obj_id
                        COBRA_MODEL_COPY.objective_direction = obj_dir
                        strain_react_x = COBRA_MODEL_COPY.reactions.get_by_id(x_gene_rxn[1])
                        if x_action.split("_")[0] == "lb":    
                            COBRA_MODEL_COPY.reactions.get_by_id(x_gene_rxn[1]).lower_bound = x_constraint     
                        elif x_action.split("_")[0] == "ub":
                            COBRA_MODEL_COPY.reactions.get_by_id(x_gene_rxn[1]).upper_bound = x_constraint

                        strain_react_y = COBRA_MODEL_COPY.reactions.get_by_id(y_gene_rxn[1])
                        if y_action.split("_")[0] == "lb":
                            COBRA_MODEL_COPY.reactions.get_by_id(y_gene_rxn[1]).lower_bound = y_constraint 
                        elif y_action.split("_")[0] == "ub":
                            COBRA_MODEL_COPY.reactions.get_by_id(y_gene_rxn[1]).upper_bound = y_constraint

                        OPT_VAL = COBRA_MODEL_COPY.optimize().f
                        interact_landscape_list.append((x_action,y_action,OPT_VAL))
                        if x_action not in interact_landscape_dict.keys():
                            interact_landscape_dict[x_action] = {}
                            interact_landscape_dict[x_action].update({y_action: OPT_VAL})
                        else:
                            interact_landscape_dict[x_action].update({y_action: OPT_VAL})

            # print x_action, x_constraint
            payoff_df = pd.DataFrame(interact_landscape_dict).T
            g = sns.heatmap(payoff_df, ax=ax_col)
            g.set_xlabel(": ".join(y_gene_rxn))
            g.set_ylabel(": ".join(x_gene_rxn))
            g.set_title(obj_id+" "+obj_dir)
            payoff_df_dict.update({obj_id+" "+obj_dir: payoff_df})
    return f, payoff_df_dict


def fva_AMR_clustermap_show(X_AMR_alleles, Y_AMR_pheno, figSIZE=(4, 8), clusterCOL=False,clusterROW=False, save_file=None):
    
    triple_color_palette = [sns.color_palette("RdBu_r", 7, desat=1)[3], sns.color_palette("RdBu_r", 7, desat=1)[-1],
            sns.color_palette("RdBu_r", 7, desat=1)[0]]
    double_color_palette = [sns.color_palette("RdBu_r", 7, desat=1)[3],sns.color_palette("RdBu_r", 7, desat=1)[-1]]
    ### Inputs: INH_species, INH_alleles, INH_phenotype
    specific_color_palette = [sns.color_palette("RdBu_r", 7, desat=1)[0], sns.color_palette("RdBu_r",7, desat=1)[-1]]
    # specific_color_palette = [sns.color_palette("RdBu_r", 3, desat=1)[0], sns.color_palette("RdBu_r", 3, desat=1)[2]]
    X_plot_df = X_AMR_alleles #INH_species.loc[:, INH_alleles]
    Y_plot_df = Y_AMR_pheno.reindex(X_plot_df.index) # INH_phenotype
    
    if True in Y_plot_df[Y_plot_df.columns].isna().any().values:
        Y_plot_df.fillna(2, inplace=True)

    colorsForDF_list = []
    for y_pheno in Y_plot_df.columns:
        labels = Y_plot_df[y_pheno].values
        if len(Y_plot_df[y_pheno].unique())>2:
            lut = {0.0: (0.9690888119953865, 0.9664744329104191, 0.9649365628604382),
             1.0: (0.7284890426758939, 0.15501730103806222, 0.19738562091503264), 
             2.0: (0.7614763552479814, 0.8685121107266436, 0.924567474048443)}
            # lut = dict(zip(set(labels), triple_color_palette))
        else:
            lut = {0.0: (0.9690888119953865, 0.9664744329104191, 0.9649365628604382),
             1.0: (0.7284890426758939, 0.15501730103806222, 0.19738562091503264)}
            # lut = dict(zip(set(labels), double_color_palette))
        # lut = dict(zip(set(labels), specific_color_palette)) # sns.hls_palette(len(set(labels)), l=0.5, s=0.8))
        row_colors_iter = pd.DataFrame(labels)[0].map(lut)
        colorsForDF_list.append(row_colors_iter)
    ### Drop FVA columns that have no differences (i.e., all the same values)
    drop_cols = []
    for col in X_plot_df.columns:
        if len(X_plot_df[col].unique())==1:
            drop_cols.append(col)
    X_plot_df.drop(drop_cols, axis=1, inplace=True)

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    g=sns.clustermap(abs(X_plot_df), 
                     # method='average', metric='euclidean',
                     method='ward', metric='euclidean', # metric='correlation',
                     standard_scale=True, # z_score=True,
                     row_cluster=clusterROW, col_cluster=clusterCOL,
                     row_colors=colorsForDF_list,cmap=cmap,figsize=figSIZE);
    
    if save_file!=None:
        g.ax_heatmap.set_title(save_file.split("/")[-1])
        g.savefig(save_file+".png", dpi=150);
        g.savefig(save_file+".svg");
    return g


def filter_amr_fva(fva_landscape_df, G_ALLELE_clustermap_data2d, Y_pheno_test):
    """Notice the use of MinMaxScaler below...
    """
    Y_pheno_test_reindexed = Y_pheno_test.reindex(G_ALLELE_clustermap_data2d.index)
    fva_landscape_df_reindexed = fva_landscape_df.reindex(G_ALLELE_clustermap_data2d.index)
    ### --- 
    landscape_scaler = MinMaxScaler() # StandardScaler()
    fva_landscape_df_reindexed.fillna(0, inplace=True)
    G_FVA_clustermap_scaled = landscape_scaler.fit_transform(fva_landscape_df_reindexed)
    G_FVA_clustermap = pd.DataFrame(G_FVA_clustermap_scaled, 
                                    index=fva_landscape_df_reindexed.index, 
                                    columns=fva_landscape_df_reindexed.columns)
    X_standardscaled_SAMPLE = G_FVA_clustermap.reindex(G_ALLELE_clustermap_data2d.index)
    y = Y_pheno_test_reindexed.astype(int)
    y.dropna(axis=0, how='all', inplace=True)
    X = X_standardscaled_SAMPLE
    X = X.reindex(y.index)
    return X, y
    

def load_landscape_sample(fva_landscape_file):
    """Load the popFVA landscape for a particular model sample
    """
    fva_landscape_dict = load_json_obj(fva_landscape_file)
    obj_val_list = {}
    for strain_id, strain_fva_dict in fva_landscape_dict.items():
        obj_val_list[strain_id] = {}
        for rxn, max_min_dict in strain_fva_dict.items():
            obj_val_list[strain_id].update({rxn+"_max":float(format(max_min_dict["maximum"],'.10f')), 
                                            rxn+"_min":float(format(max_min_dict["minimum"],'.10f'))})
    fva_landscape_df = pd.DataFrame.from_dict(obj_val_list, orient="index")
    return fva_landscape_df


def get_sample_constraints(variant_dec_file):
    """Load the allele-constraint map for a 
    particular model sample"""
    if os.path.exists(variant_dec_file):
        variant_dec_dict = load_json_obj(variant_dec_file)
    else:
        print("variant_dec_dict does not exist: ", variant_dec_file)
        variant_dec_dict = {}
    return variant_dec_dict
