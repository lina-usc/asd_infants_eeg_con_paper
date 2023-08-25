from pathlib import Path

from IPython.core.display import display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import mne_connectivity
from tqdm.notebook import tqdm
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

import acareeg


def abline(slope, intercept, ax, **kwargs):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, **kwargs)


def get_ind_both_time_point(tmp):
    return np.in1d(tmp.subject, tmp[["con", "age", "subject"]].pivot_table(values="con", columns="age", index="subject").dropna().index)


def add_bin_dist(df, key, nb_bins=50):
    bins = np.percentile(df[key], np.linspace(start=0, stop=100, num=nb_bins, endpoint=True))
    bins[-1] += 0.00001
    dist_bins = (bins[1:] + bins[:-1])/2.0
    df.insert(len(df.columns), key + "_bin", dist_bins[np.digitize(df[key], bins)-1])   
    return df


def load_connectivity_data(con_path, band="broadband", con_name="ciplv", lambda2=1e-4, subjects_dir = ".",):
    # Merging individual connectivity files
    use_common_template = True
    common_template_age = 12

    if use_common_template:
        pattern = f"*{con_name}_{common_template_age}m-template.csv"
    else:
        pattern = "*.csv"

    dfs = []
    for path in con_path.glob(pattern):
        subject, dataset, age = path.name[:-4].split("_")[:3]
        df = pd.read_csv(path, index_col=0)

        # Averaging across bootstrapping iterations
        df.groupby(list(df.columns.drop("con"))).mean().reset_index()
        df["subject_no"] = int(subject)
        df["dataset"] = dataset
        df["age"] = int(age)
        df["con_name"] = con_name
        df = df[(df.band == band) & (df.con_name == con_name)]
        dfs.append(df)    
    dat = pd.concat(dfs)
    dat = dat.groupby(["region1", "region2", "con_name", "band", "age", "subject_no", "dataset"]).mean().reset_index()
    del dfs

    # Adding region position info
    mne.datasets.fetch_infant_template(f'{common_template_age}mo', subjects_dir=subjects_dir)

    if use_common_template:
        pos_df = pd.concat([acareeg.infantmodels.region_centers_of_masse(age, include_vol_src=False, subjects_dir=subjects_dir) 
                            for age in [common_template_age]])
        dat["template"] = [f"ANTS{common_template_age}-0Months3T" for _ in dat.age.values]
    else:
        pos_df = pd.concat([acareeg.infantmodels.region_centers_of_masse(age, include_vol_src=False, subjects_dir=subjects_dir) 
                            for age in [6, 12, 18]])
        dat["template"] = [f"ANTS{age}-0Months3T" for age in dat.age.values]

    dat = dat.merge(pos_df, left_on=["template", "region1"], right_on=["template", "region"], suffixes=("", "_1"))
    dat = dat.merge(pos_df, left_on=["template", "region2"], right_on=["template", "region"], 
                    suffixes=("_1", "_2")).drop(columns=["region_1", "region_2"])


    # Rejecting the entorhinal regions because they are not present in all templates
    dat = dat[dat.region1 != "entorhinal-lh"]
    dat = dat[dat.region1 != "entorhinal-rh"]
    dat = dat[dat.region2 != "entorhinal-lh"]
    dat = dat[dat.region2 != "entorhinal-rh"]

    # Adding between-region distances
    dat["dx"] = np.abs(dat.x_1 - dat.x_2)
    dat["dy"] = np.abs(dat.y_1 - dat.y_2)
    dat["dz"] = np.abs(dat.z_1 - dat.z_2)
    
    dat["dist"] = np.sqrt(dat.dx**2 + dat.dy**2 + dat.dz**2)

    for col in ['x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2', 'dx', 'dy', 'dz']:
        del dat[col]        
    
    add_bin_dist(dat, key="dist")

    dat["region12"] = ["{}-{}".format(r1, r2) for r1, r2 in zip(dat["region1"], dat["region2"])]
    dat["dataset_age"] = ["{}-{}".format(d, a) for d, a in zip(dat["dataset"], dat["age"])]
    dat["subject"] = ["{}-{}".format(subject_no, dataset) 
                         for subject_no, dataset in zip(dat.subject_no, dat.dataset)]
    return dat


def get_demo_data(force_download=False):
    #mastersheet = pd.read_excel("/home/christian/Downloads/Mastersheet_2020_03_13.xlsx", index_col=0, header=0)
    mastersheet = acareeg.eegip.get_mastersheet(force_download=force_download)

    # Keeping the latest ADOS score available
    ados_cols = ['adoscss_24m', 'ados_sacss_24m', 'ados_rrbcss_24m', 'adoscss_36m', 'ados_sacss_36m', 'ados_rrbcss_36m']
    mastersheet[mastersheet[ados_cols] > 100] = np.nan
    mastersheet[['adoscss_earliest', 'ados_sacss_earliest', 'ados_rrbcss_earliest']] = [row[['adoscss_24m', 'ados_sacss_24m', 'ados_rrbcss_24m']].values 
                                                                                         if not np.isnan(row['adoscss_24m']) 
                                                                                         else row[['adoscss_36m', 'ados_sacss_36m', 'ados_rrbcss_36m']].values
                                                                                         for ind, row in mastersheet.iterrows()]    
    mastersheet[['adoscss_latest', 'ados_sacss_latest', 'ados_rrbcss_latest']] = [row[['adoscss_24m', 'ados_sacss_24m', 'ados_rrbcss_24m']].values 
                                                                                 if np.isnan(row['adoscss_36m']) 
                                                                                 else row[['adoscss_36m', 'ados_sacss_36m', 'ados_rrbcss_36m']].values
                                                                                 for ind, row in mastersheet.iterrows()]    

    mastersheet = mastersheet.reset_index()[["Site", "ID", "sex", "outcome", "Risk Group", 
                                             'adoscss_earliest', 'ados_sacss_earliest', 'ados_rrbcss_earliest',
                                             'adoscss_latest', 'ados_sacss_latest', 'ados_rrbcss_latest',
                                             'adoscss_24m', 'ados_sacss_24m', 'ados_rrbcss_24m',
                                             'adoscss_36m', 'ados_sacss_36m', 'ados_rrbcss_36m']]

    mastersheet.rename(columns={"Site": "dataset", "ID": "subject_no", "Risk Group": "risk"}, inplace=True)
    mastersheet = mastersheet.dropna(how="all")
    mastersheet.loc[mastersheet.risk == "High Risk", "risk"] = "HRA"
    mastersheet.loc[mastersheet.risk == "High Risk_12mo", "risk"] = "HRA"
    mastersheet.loc[mastersheet.risk == "Low Risk", "risk"] = "Control"
    mastersheet.loc[mastersheet.risk == "Low Risk_12mo", "risk"] = "Control"
    mastersheet.dataset = mastersheet.dataset.str.lower()
    mastersheet["subject_no"] = [int(subject[-3:]) if isinstance(subject, str) else subject for subject in mastersheet["subject_no"]]

    mastersheet.loc[(mastersheet.outcome == "asd") & (mastersheet.risk == "HRA"), "group"] = "HRA-ASD"
    mastersheet.loc[(mastersheet.outcome == "no-asd") & (mastersheet.risk == "HRA"), "group"] = "HRA-noASD"
    mastersheet.loc[(mastersheet.outcome == "no-asd") & (mastersheet.risk == "Control"), "group"]  = "Control"
    mastersheet.loc[(mastersheet.outcome == 777) & (mastersheet.risk == "Control"), "group"]  = "Control"

    mastersheet["group"] = pd.Categorical(mastersheet.group,
                                          categories=["Control",
                                                      "HRA-noASD",
                                                      "HRA-ASD"],
                                          ordered=True)
    return mastersheet


def get_thresholds(df, func, k=1.5):
    data = {"site": [], "age": [], "lower": [],
            "upper": [], "Q1": [], "Q2": [], "Q3": []}
    for (site, age), group in df.groupby(["site", "age"]):
        log_con = func(group["con"])
        if len(log_con) == 0:
            continue
        Q1, Q2, Q3 = np.nanpercentile(log_con, [25, 50, 75])

        data["lower"].append(Q1 - k*(Q3-Q1))
        data["upper"].append(Q3 + k*(Q3-Q1))
        data["Q1"].append(Q1)
        data["Q2"].append(Q2)
        data["Q3"].append(Q3)
        data["site"].append(site)
        data["age"].append(age)

    return pd.DataFrame(data).set_index(["site", "age"])


def label_outliers(dat, func=None, func_name=None):

    if func is None:
        func = logit
        func_name = "logit"

    tmp = dat.groupby(["subject", "sex", "age", "group", "site"],
                      observed=True).mean().reset_index()
    threshold_df = get_thresholds(tmp, func)

    rejected = []
    for (site, age), tmp2 in tmp.groupby(["site", "age"]):
        log_con = func(tmp2["con"])
        if len(log_con) == 0:
            continue

        thresh = threshold_df.loc[site, age]
        rejected.append(tmp2[(log_con > thresh.upper) |
                             (log_con < thresh.lower)])

    rejected = pd.concat(rejected)

    subject_age = ["{}-{}".format(subject, age)
                   for subject, age in zip(dat.subject, dat.age)]
    rejected_subject_age = ["{}-{}".format(subject, age)
                            for subject, age
                            in zip(rejected.subject, rejected.age)]
    dat["outliers_" + func_name] = np.in1d(subject_age, rejected_subject_age)

    return rejected


def plot_outliers(dat, func=None, func_name=None):

    if func is None:
        func = logit
        func_name = "logit"

    tmp = dat.groupby(["subject", "sex", "age", "group", "site"],
                      observed=True).mean().reset_index()
    threshold_df = get_thresholds(tmp, func)

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5), sharex=True, sharey=True)

    min_ = 100
    max_ = -100
    for age, ax in zip([6, 12, 18], axes):
        log_con = func(tmp["con"][tmp.age.values == age])
        min_ = min(min_, np.nanmin(log_con))
        max_ = max(max_, np.nanmax(log_con))

    bins = np.linspace(min_-0.01*(max_-min_),
                       max_+0.01*(max_-min_),
                       30)

    for ((site, age), tmp2), ax in zip(tmp.groupby(["site", "age"]),
                                       axes.ravel()):
        log_con = func(tmp2["con"])
        if len(log_con) == 0:
            continue

        thresh = threshold_df.loc[site, age]
        x = ax.hist(log_con[tmp2.group == "Control"], bins=bins,
                    color="green", label="TLA")
        y = ax.hist(log_con[tmp2.group == "HRA-noASD"], bins=bins,
                    bottom=x[0], color="orange", label="ELA-noASD")
        ax.hist(log_con[tmp2.group == "HRA-ASD"], bins=bins,
                bottom=x[0] + y[0], color="red", label="ELA-ASD")
        ax.axvline(x=thresh.upper, color="r", linestyle="--")
        ax.axvline(x=thresh.lower, color="r", linestyle="--")

        ax.axvline(x=thresh.Q1, color="k", linestyle="--")
        ax.axvline(x=thresh.Q2, color="k")
        ax.axvline(x=thresh.Q3, color="k", linestyle="--")

        if site == "London":
            ax.set_title("{} months".format(age))

    axes[0, 0].set_ylabel("Nb. subjects - London")
    axes[1, 0].set_ylabel("Nb. subjects - Seattle")

    if func_name == "identity":
        axes[1, 0].set_xlabel("logit(CON)")
        axes[1, 1].set_xlabel("logit(CON)")
    else:
        axes[1, 0].set_xlabel(func_name + "(CON)")
        axes[1, 1].set_xlabel(func_name + "(CON)")

    ax.legend()

    fig.tight_layout(h_pad=2, w_pad=1)
    fig.savefig("images/" + func_name + "_connectivity_outliers.png", dpi=300)


def logit(x):
    return np.log(x/(1-x))
