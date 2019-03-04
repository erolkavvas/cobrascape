### Plotting utilities for cobrascape
import cobrascape.ensemble as ens
from random import shuffle
import numpy as np
import seaborn as sns
import warnings ### Freaking SKlearn bro gives hella warnings.
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from matplotlib.colors import Normalize


def popfva_manhatten_plot(popfva_enrich_dic, s_obj, pheno_id="isoniazid", fdr_line=True,
                          s_size=100, labelsizes=20, f_scale=1.0, figSIZE=(9,4)):
    """Plots a manhatten plot for both alleles and popFVA associations
        - popfva_enrich_dict = {pheno_id: dataframe of popfva enrichments}
        - s_obj = cobrascape Sample object
        - f_scale = fontsize
    """
    AMR_to_gene = {"ethambutol": ["Rv3806c", "Rv3795"], "pyrazinamide": ["Rv2043c"],
                   "isoniazid": ["Rv1908c", "Rv1484"], 
                   ## "isoniazid": ["Rv1908c","Rv1484","Rv2243","Rv2247","Rv2245","Rv3139","Rv1483","Rv0129c"], 
                   "4-aminosalicylic_acid": ["Rv2447c", "Rv2764c"]}
    # sns.set()
    plt.style.use(["seaborn-white"])
    sns.set_style("ticks")
    flatui = ["#e74c3c", "#3498db", "#9b59b6",   "#ff7f00",  "lightgray"]
    current_palette = sns.color_palette(flatui)
    amr_to_color = dict(zip(["isoniazid", "pyrazinamide", "ethambutol", 
                             "4-aminosalicylic_acid", "unknown"], current_palette.as_hex()))
    
    rc_par = {"axes.labelsize": labelsizes, "xtick.labelsize":labelsizes, 
              "ytick.labelsize":labelsizes,"axes.titlesize":labelsizes}
    with sns.plotting_context("notebook", font_scale=f_scale, rc=rc_par):
        fig, (ax_gwas, ax_popfva) = plt.subplots(1, 2, figsize=figSIZE)
        X_alleles = s_obj.x_allele_dict[pheno_id]
        Y_pheno = s_obj.y_pheno_dict[pheno_id]
        ### ---- popFVA manhatten plot -----
        sample_inference_df = s_obj.anova_dict[pheno_id].copy()
        sample_inference_df.fillna(1,inplace=True)
        sample_inference_df["AIC"]=sample_inference_df.index.map(lambda x: s_obj.assess_df.loc[x,"AIC_"+pheno_id])
        sample_inference_df["BIC"]=sample_inference_df.index.map(lambda x: s_obj.assess_df.loc[x,"BIC_"+pheno_id])

        react_to_gene, react_to_AMR = {}, {}
        feat_to_test_df = popfva_enrich_dic[pheno_id]
        for y in feat_to_test_df.index:
            react = y.replace("_max", "").replace("_min", "")
            rxn_genes = [str(x.id) for x in s_obj.base_cobra_model.reactions.get_by_id(react).genes]
            react_to_gene.update({y: rxn_genes})
            for rg in rxn_genes:
                for dr, dr_g in AMR_to_gene.items():
                    if rg in dr_g:
                        react_to_AMR.update({y: dr})

        popfva_genes = list(set([str(x.split("_")[0]) for x in X_alleles.columns]))
        feat_to_test_df["drug_type"]= feat_to_test_df.index.map(lambda x: react_to_AMR[x] if x in react_to_AMR.keys() else "unknown")
        feat_to_test_df["log10pval"]= -np.log(feat_to_test_df["pvalue"])
        x = [i for i in range(len(feat_to_test_df.index))]
        shuffle(x)
        feat_to_test_df["rand_index"] = x
        new_color_order = [amr_to_color[x] for x in feat_to_test_df["drug_type"].unique()]
        ax_popfva = sns.scatterplot(x="rand_index", y="log10pval", data=feat_to_test_df, hue="drug_type",
                         palette=new_color_order, ax=ax_popfva, s=s_size)
        ax_popfva.set_title("popFVA enrichments: "+pheno_id)
        ax_popfva.set_xlabel("popFVA features of "+str(len(popfva_genes))+" AMR genes")
        ax_popfva.legend(loc='upper right', bbox_to_anchor=(1.8, 0.9))

        ### ---- Allele manhatten plot ---- 
        anova_df = ens.compute_ANOVA_test(X_alleles,Y_pheno)
        anova_df = ens.FDR(anova_df, fdr_rate=1) ### Add horizontal lines for FDR
        anova_genes = list(set([str(x.split("_")[0]) for x in anova_df.index]))
        allele_to_AMR = {}
        for drg, drg_genes in AMR_to_gene.items():
            for allele in anova_df.index:
                if allele.split("_")[0] in drg_genes:
                    allele_to_AMR.update({allele: drg})

        anova_df["drug_type"]=anova_df.index.map(lambda x: allele_to_AMR[x] if x in allele_to_AMR.keys() else "unknown")
        anova_df["log10pval"]= -np.log(anova_df["pvalue"])
        x = [i for i in range(len(anova_df.index))]
        shuffle(x)
        anova_df["rand_index"] = x
        new_color_order = [amr_to_color[x] for x in anova_df["drug_type"].unique()]

        ax_gwas = sns.scatterplot(x="rand_index", y="log10pval", data=anova_df, hue="drug_type",
                                  ax=ax_gwas, palette=new_color_order, s=s_size, legend=False)
        ax_gwas.set_title("classical GWAS: "+pheno_id)
        ax_gwas.set_xlabel("alleles of "+str(len(popfva_genes))+" AMR genes")
    return fig

