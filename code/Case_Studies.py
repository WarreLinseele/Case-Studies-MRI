"""
How does trait neuroticism (NEO-FFI N subscale) relate to brain responses
during dynamic face perception (anger, contempt, joy, pride, neutral) in PIOP1?

H01  Higher neuroticism → increased amygdala & dmPFC activity for all
     emotionally loaded expressions  [emotion > neutral]
H02  Higher neuroticism → increased amygdala & dmPFC for NEGATIVE expressions
     [anger + contempt > neutral]
H03  Higher neuroticism → activity changes (any direction) for POSITIVE
     expressions  [joy + pride > neutral]  — exploratory, two-tailed
H04  Neuroticism–amygdala/dmPFC association is stronger for negative than
     positive expressions  [negative > positive]
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn import plotting
from nilearn.maskers import NiftiMasker
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.glm import threshold_stats_img
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


###
# Configuration
###

# Root of dataset
root = r"\wsl.localhost\Ubuntu\home\warrelinseele\ds002785-download"

# fMRIPrep derivatives directory
fMRIPrep_derivatives = os.path.join(root, "derivatives", "fmriprep")

# Place for output files
output = os.path.join(root, "derivatives", "neuroticism_analysis")

# Acquisition parameters
tr              = 0.75  # seconds, MB3 multiband, face-perception task
n_volumes       = 330   # volumes per run  (247.5 s / 0.75 s)
slice_time_ref  = 0.5   # reference slice = midpoint of TR
                         # Note: fMRIPrep PIOP1 did NOT apply slice-timing correction (stated in paper), so this mainly affects the HRF shift — leaving at 0.5 is safe.

# GLM parameters
hrf_model       = "spm"     # SPM canonical double-gamma HRF
drift_model     = "cosine"  # cosine basis set for high-pass filtering
high_pass       = 1 / 128   # Hz  ≈ 128 s cutoff  (SPM / nilearn default)
noise_model     = "ar1"     # AR(1) temporal autocorrelation correction
smoothing_FWHM  = 6.0       # mm FWHM Gaussian kernel (standard for group comparisons in small FOV)

# Quality control
max_mean_fd     = 0.5   # mm — exclude subject if mean FD > threshold
max_spike_frac  = 0.20  # exclude if > 20 % volumes have FD > 0.5 mm

# Statistics
roi_fdr_alpha  = 0.05  # FDR-BH threshold for ROI tests
wb_z_thresh    = 3.29  # z ≈ p < 0.001 uncorrected for whole-brain
wb_cluster_k   = 20    # minimum cluster size (voxels)


###
# Subject discovery and path utilities
###

# Sorted list of subject IDs with face-task fMRIPrep output
def discover_subjects(fmriprep_dir: str) -> list:
    pattern = os.path.join(
        fmriprep_dir, "sub-*", "func",
        "*task-faces*acq-mb3*space-MNI152NLin2009cAsym*desc-preproc_bold.nii.gz"
    )
    hits = sorted(glob.glob(pattern))
    ids  = sorted({os.path.basename(p).split("_")[0] for p in hits})
    print(f"[discover] Found {len(ids)} subjects with MB3 face-task BOLD.")
    return ids


# Assembling all file paths for one subject
def subject_paths(sub_id: str) -> dict:
    func_deriv  = os.path.join(fMRIPrep_derivatives, sub_id, "func")
    func_raw    = os.path.join(root, sub_id, "func")
    base        = f"{sub_id}_task-faces_acq-mb3"
    space       = "space-MNI152NLin2009cAsym"

    return {
        "bold":      os.path.join(func_deriv, f"{base}_{space}_desc-preproc_bold.nii.gz"),
        "mask":      os.path.join(func_deriv, f"{base}_{space}_desc-brain_mask.nii.gz"),
        # fMRIPrep ≥ 1.4 uses _desc-confounds_timeseries.tsv
        # older fMRIPrep uses _desc-confounds_regressors.tsv
        "confounds": (os.path.join(func_deriv, f"{base}_desc-confounds_timeseries.tsv")
                      if os.path.exists(os.path.join(func_deriv,
                          f"{base}_desc-confounds_timeseries.tsv"))
                      else os.path.join(func_deriv, f"{base}_desc-confounds_regressors.tsv")),
        "events":    os.path.join(func_raw, f"{base}_events.tsv"),
    }


def paths_ok(paths: dict) -> bool:
    check = True
    for k, v in paths.items():
        if not os.path.exists(v):
            print(f"  [MISSING] {k}: {v}")
            check = False
    return check


###
# Motion quality control
###

# Computing mean framewise displacement and spike fraction
def motion_stats(confounds_file: str) -> tuple:
    cf = pd.read_csv(confounds_file, sep="\t")
    col = "framewise_displacement"
    if col not in cf.columns:
        return np.nan, np.nan
    fd     = cf[col].fillna(0).values
    mfd    = float(np.mean(fd))
    spikes = float(np.mean(fd > 0.5))
    return mfd, spikes


# Selecting a 24-parameter motion model and ACompCor and global signal
"""
 # High parameter since neurotic subjects tend to move more often (residual motion can correlate with neuroticism scores)
 # The Friston 24-parameter model removes spin-history (quadratic) effects that are common with head motion
 # The top-5 aCompCor components capture WM/CSF physiological noise more completely than a single mean signal
  # Columns selected( if present in file):
   # - 6 rigid - body realignment parameters
   # - their first derivatives (suffix_derivative1) 
   # - quadratic terms (suffix_power2)
   # - derivative quadratic terms(_derivative1_power2) → 24 total motion
   # - top 5 aCompCor components (a_comp_cor_00 … _04)
   # - global_signal
"""
def select_confounds(confounds_file: str) -> pd.DataFrame:
    cf = pd.read_csv(confounds_file, sep="\t")
    wanted = [
        "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
        "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
        "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
        "trans_x_power2", "trans_y_power2", "trans_z_power2",
        "rot_x_power2", "rot_y_power2",  "rot_z_power2",
        "trans_x_derivative1_power2", "trans_y_derivative1_power2", "trans_z_derivative1_power2",
        "rot_x_derivative1_power2",   "rot_y_derivative1_power2", "rot_z_derivative1_power2",
        "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03", "a_comp_cor_04",
        "global_signal",
    ]
    cols   = [c for c in wanted if c in cf.columns]
    result = cf[cols].fillna(0)
    print(f"  Confounds selected: {len(cols)} columns")
    return result


###
# Contrast definitions
###

contrasts = {
    # H01: all 4 emotional expressions > neutral
    "emotion_vs_neutral":   "anger + contempt + pride + joy - 4*neutral",
    # H02: negative > neutral
    "negative_vs_neutral":  "anger + contempt - 2*neutral",
    # H03: positive > neutral
    "positive_vs_neutral":  "pride + joy - 2*neutral",
    # H04: negative > positive
    "negative_vs_positive": "anger + contempt - pride - joy",
}


###
# First-level GLM
###

# Fit first-level GLM and compute 4 contrast beta maps for one subject
def run_first_level(sub_id: str, paths: dict, output_dir: str, overwrite: bool = False) -> dict:

    fl_dir = os.path.join(output_dir, "first_level")
    os.makedirs(fl_dir, exist_ok=True)
    beta_paths = {
        name: os.path.join(fl_dir, f"{sub_id}_{name}_beta.nii.gz")
        for name in contrasts
    }

    # Return cached maps if already computed
    if not overwrite and all(os.path.exists(p) for p in beta_paths.values()):
        print(f"  [{sub_id}] First-level cached — loading.")
        return {name: nib.load(p) for name, p in beta_paths.items()}
    if not paths_ok(paths):
        print(f"  [{sub_id}] SKIP — missing files.")
        return {}

    # Load events (keep only columns nilearn needs; extra columns are warned about)
    events = pd.read_csv(paths["events"], sep="\t")[["onset", "duration", "trial_type"]]

    # Verify conditions match expectations
    found = set(events["trial_type"].unique())
    expected = {"anger", "contempt", "joy", "pride", "neutral"}
    if not expected.issubset(found):
        print(f"  [{sub_id}] WARNING: conditions found = {found}")

    confounds = select_confounds(paths["confounds"])

    glm = FirstLevelModel(
        t_r            = tr,
        hrf_model      = hrf_model,
        drift_model    = drift_model,
        high_pass      = high_pass,
        slice_time_ref = slice_time_ref,
        noise_model    = noise_model,
        mask_img       = paths["mask"],
        smoothing_fwhm = smoothing_FWHM,
        standardize    = False,
        verbose        = 0,
    )
    glm.fit(paths["bold"], events=events, confounds=confounds)

    beta_imgs = {}
    for name, formula in contrasts.items():
        beta_img = glm.compute_contrast(formula, output_type="effect_size")
        z_img    = glm.compute_contrast(formula, output_type="z_score")
        nib.save(beta_img, beta_paths[name])
        nib.save(z_img, beta_paths[name].replace("_beta.", "_zstat."))
        beta_imgs[name] = beta_img

    print(f"  [{sub_id}] First-level done.")
    return beta_imgs


###
# Phenotype loading
###

# Load participants.tsv from the BIDS root directory.
def load_participants(bids_root: str) -> pd.DataFrame:
    tsv = os.path.join(bids_root, "participants.tsv")
    if not os.path.exists(tsv):
        raise FileNotFoundError(f"participants.tsv not found at: {tsv}")

    df = pd.read_csv(tsv, sep="\t", na_values="n/a")
    df = df.rename(columns={"participant_id": "subject"})
    df = df.set_index("subject")

    # Standardise neuroticism column name
    neuro_candidates = [c for c in df.columns if "neuro" in c.lower() or c.upper() in ("NEO_N", "NEO-N")]
    if not neuro_candidates:
        raise KeyError(f"Cannot find neuroticism column. Columns: {list(df.columns)}")
    df = df.rename(columns={neuro_candidates[0]: "neuroticism"})
    df["neuroticism_z"] = (df["neuroticism"] - df["neuroticism"].mean()) \
                           / df["neuroticism"].std()

    # Encode sex as binary dummy
    if "sex" in df.columns:
        df["sex_code"] = (df["sex"].str.lower() == "male").astype(float)

    print(f"[phenotype] Loaded {len(df)} participants.")
    print(df["neuroticism"].describe().to_string())
    return df


# Compute per-subject mean FD and merge into phenotype dataframe
"""
 # Mean FD as second-level covariate since anxious subjects tend to move more
 # Prevents motion artefacts inflating neuro-BOLD correlations
"""
def add_mean_fd(pheno: pd.DataFrame, subject_ids: list) -> pd.DataFrame:
    records = []
    for sid in subject_ids:
        paths = subject_paths(sid)
        mfd, _ = motion_stats(paths["confounds"]) if os.path.exists(paths["confounds"]) \
                 else (np.nan, np.nan)
        records.append({"subject": sid, "mean_fd": mfd})

    fd_df  = pd.DataFrame(records).set_index("subject")
    result = pheno.join(fd_df, how="left")
    return result


###
# ROI masks (amygdala + dmPFC)
###

# Build binary ROI masks using Harvard–Oxford atlases
"""
 # dmPFC as 'Paracingulate Gyrus' + 'Cingulate Gyrus, anterior division'
 # approximate dorsomedial PFC / dACC, consistent with the PPT's 'dmPFC' ROI specification.
"""
def build_roi_masks() -> dict:
    ho_sub  = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    ho_cort = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")

    def mask_from_labels(atlas, keywords):
        labels = atlas.labels
        indices = [i for i, l in enumerate(labels)
                   if any(kw.lower() in l.lower() for kw in keywords)]
        print(f"  ROI labels matched: {[labels[i] for i in indices]}")
        data = np.asarray(atlas.maps.dataobj)
        # Harvard-Oxford label indices are 1-based in the image
        binary = np.isin(data, [i + 1 for i in indices]).astype(np.int8)
        return nib.Nifti1Image(binary, atlas.maps.affine)

    return {
        "amygdala": mask_from_labels(ho_sub,  ["amygdala"]),
        "dmPFC":    mask_from_labels(ho_cort, ["paracingulate", "cingulate gyrus, anterior"]),
    }


# Compute mean beta estimate across all voxels in each ROI mask
def extract_roi_betas(sub_id: str, beta_imgs: dict,
                      roi_masks: dict) -> dict:
    row = {"subject": sub_id}
    for roi_name, mask_img in roi_masks.items():
        masker = NiftiMasker(mask_img=mask_img, standardize=False)
        for contrast_name, beta_img in beta_imgs.items():
            vals = masker.fit_transform(beta_img)
            row[f"{contrast_name}_{roi_name}"] = float(np.nanmean(vals))
    return row


###
# Second-level ROI analysis
###

# For every ROI x contrast combination: beta_roi ~ neuroticism_z + age + sex_code + mean_fd
# with FDR-BH correctio across all tests (one correction per ROI to keep tests within the same brain region grouped; adjust as needed)
def roi_regression(roi_df: pd.DataFrame, pheno: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    os.makedirs(os.path.join(output_dir, "roi"), exist_ok=True)

    # Merge ROI values with phenotype
    covar_cols = [c for c in ["neuroticism_z", "age", "sex_code", "mean_fd"]
                  if c in pheno.columns]
    df = roi_df.join(pheno[covar_cols], how="inner").dropna()
    print(f"\n[second-level ROI] N = {len(df)} subjects after dropna")

    roi_cols = [c for c in roi_df.columns if c != "subject"]
    results  = []

    for col in roi_cols:
        # backtick-quote the column name for formula-API compatibility
        formula = f"`{col}` ~ " + " + ".join(covar_cols)
        try:
            model = smf.ols(formula, data=df).fit()
            results.append({
                "roi_contrast": col,
                "n":            int(model.nobs),
                "beta_neuro":   float(model.params["neuroticism_z"]),
                "t_value":      float(model.tvalues["neuroticism_z"]),
                "p_unc":        float(model.pvalues["neuroticism_z"]),
                "r_squared":    float(model.rsquared),
            })
        except Exception as exc:
            print(f" [WARN] Regression failed for {col}: {exc}")
            results.append({
                "roi_contrast": col, "n": len(df),
                "beta_neuro": np.nan, "t_value": np.nan,
                "p_unc": np.nan, "r_squared": np.nan,
            })

    res = pd.DataFrame(results)

    # FDR-BH correction
    valid_mask = res["p_unc"].notna()
    _, p_fdr, _, _ = multipletests(
        res.loc[valid_mask, "p_unc"].values,
        alpha=roi_fdr_alpha, method="fdr_bh"
    )
    res.loc[valid_mask, "p_fdr"] = p_fdr
    res["sig_fdr"] = res["p_fdr"] < roi_fdr_alpha

    out = os.path.join(output_dir, "roi", "roi_regression_results.csv")
    res.to_csv(out, index=False)
    print("\n[ROI results]")
    print(res.to_string(index=False))
    print(f"Saved: {out}")
    return res


###
# Whole-brain voxelwise regression
###

# Voxelwise second-level OLS regression: voxel_beta ~ neuroticism_z + age + sex_code + mean_fd
def whole_brain_regression(beta_paths_by_contrast: dict, pheno: pd.DataFrame, output_dir: str):
    os.makedirs(os.path.join(output_dir, "second_level"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"),      exist_ok=True)

    covar_cols = [c for c in ["neuroticism_z", "age", "sex_code", "mean_fd"]
                  if c in pheno.columns]

    for contrast, paths_list in beta_paths_by_contrast.items():
        # Filter to subjects that exist on disk and have full phenotype data
        valid = [(p, os.path.basename(p).split("_")[0])
                 for p in paths_list if os.path.exists(p)]
        subs  = [s for _, s in valid]

        # Only keep subjects present in phenotype with no NaN in covariates
        pheno_sub = pheno.reindex(subs)[covar_cols].dropna()
        keep_subs = list(pheno_sub.index)
        imgs      = [p for p, s in valid if s in keep_subs]
        pheno_sub = pheno_sub.loc[keep_subs]

        if len(imgs) < 10:
            print(f"  [SKIP] {contrast}: only {len(imgs)} valid subjects.")
            continue

        print(f"\n[whole-brain] {contrast}: N = {len(imgs)}")

        # Design matrix: intercept + covariates (neuroticism_z first for contrast)
        dm = pheno_sub[covar_cols].copy()
        dm.insert(0, "intercept", 1.0)
        dm = dm.reset_index(drop=True)

        second_level = SecondLevelModel(n_jobs=-1)
        second_level.fit(imgs, design_matrix=dm)

        z_img = second_level.compute_contrast(
            second_level_contrast="neuroticism_z",
            output_type="z_score"
        )

        # Threshold: p < 0.001 uncorrected + minimum cluster size
        thresh_img, thresh_z = threshold_stats_img(
            z_img, alpha=0.001, height_control="fpr",
            cluster_threshold=wb_cluster_k
        )
        # FDR threshold
        fdr_img, fdr_z = threshold_stats_img(
            z_img, alpha=0.05, height_control="fdr"
        )

        base = os.path.join(output_dir, "second_level", contrast)
        nib.save(z_img,     f"{base}_z.nii.gz")
        nib.save(thresh_img, f"{base}_z_p001_k{wb_cluster_k}.nii.gz")
        nib.save(fdr_img,    f"{base}_z_fdr05.nii.gz")
        print(f"  Saved z-maps for {contrast}.")

        # Figure: glass brain of thresholded map
        fig_path = os.path.join(output_dir, "figures", f"{contrast}_wholebrain_neuroticism.png")
        display = plotting.plot_glass_brain(
            thresh_img,
            title=(f"Neuroticism ~ {contrast}\n"
                   f"z > {thresh_z:.2f}, k ≥ {wb_cluster_k} voxels"),
            colorbar=True,
            plot_abs=False,
        )
        display.savefig(fig_path, dpi=150)
        display.close()
        print(f"  Figure: {fig_path}")


###
# Visualization
###

# Scatter: ROI mean beta vs. neuroticism score with regression line
def scatter_roi(roi_df: pd.DataFrame, pheno: pd.DataFrame,
                contrast: str, roi: str, output_dir: str):
    col = f"{contrast}_{roi}"
    if col not in roi_df.columns:
        return
    df = roi_df[[col]].join(pheno[["neuroticism"]], how="inner").dropna()
    if len(df) < 5:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df["neuroticism"], df[col], s=25, alpha=0.6, color="steelblue",
               edgecolors="white", linewidths=0.4)
    slope, intercept = np.polyfit(df["neuroticism"].to_numpy(dtype=float), df[col].to_numpy(dtype=float), 1)
    xs   = np.linspace(df["neuroticism"].min(), df["neuroticism"].max(), 100)
    ax.plot(xs, slope * xs + intercept, color="crimson", lw=2)
    ax.set_xlabel("NEO-FFI Neuroticism", fontsize=11)
    ax.set_ylabel(f"Mean β  [{roi}]", fontsize=11)
    ax.set_title(f"{contrast}\n× Neuroticism ({roi})", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    out = os.path.join(output_dir, "figures", f"scatter_{contrast}_{roi}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Scatter saved: {out}")


###
# MAIN
###

def main(test_subject: str = None, overwrite: bool = False):

    # Create output directories
    for d in ["first_level", "second_level", "roi", "figures"]:
        os.makedirs(os.path.join(output, d), exist_ok=True)

    # 1. Subject discovery
    if test_subject:
        all_subs = [test_subject]
        print(f"[TEST MODE] Running single subject: {test_subject}")
    else:
        all_subs = discover_subjects(fMRIPrep_derivatives)

    # 2. Load phenotype
    pheno = load_participants(root)
    pheno = add_mean_fd(pheno, all_subs)

    # Restrict to subjects with both imaging and phenotype
    subs_with_pheno = [s for s in all_subs if s in pheno.index]
    print(f"[overlap] {len(subs_with_pheno)} subjects have imaging + phenotype.")

    # 3. Motion QC exclusion
    included, excluded = [], []
    for sub in subs_with_pheno:
        mfd = pheno.loc[sub, "mean_fd"] if "mean_fd" in pheno.columns else np.nan
        paths = subject_paths(sub)
        _, spk = motion_stats(paths["confounds"]) \
                 if os.path.exists(paths["confounds"]) else (np.nan, np.nan)
        if np.isnan(mfd) or mfd > max_mean_fd or spk > max_spike_frac:
            excluded.append(sub)
        else:
            included.append(sub)

    print(f"[QC] Included: {len(included)},  Excluded: {len(excluded)}")

    # 4. First-level GLM loop
    roi_masks  = build_roi_masks()
    roi_rows   = []
    all_betas  = {}    # {contrast: [path, ...]}

    for cname in contrasts:
        all_betas[cname] = []

    for sub in included:
        print(f"\n=== {sub} ===")
        paths     = subject_paths(sub)
        beta_imgs = run_first_level(sub, paths, output, overwrite=overwrite)
        if not beta_imgs:
            continue

        # Collect beta paths for second-level
        for cname in contrasts:
            bpath = os.path.join(output, "first_level", f"{sub}_{cname}_beta.nii.gz")
            all_betas[cname].append(bpath)

        # Extract ROI mean betas
        row = extract_roi_betas(sub, beta_imgs, roi_masks)
        roi_rows.append(row)

    # 5. ROI second-level
    if len(roi_rows) == 0:
        print("[ERROR] No subjects completed first-level. Exiting.")
        return

    roi_df = pd.DataFrame(roi_rows).set_index("subject")
    roi_df.to_csv(os.path.join(output, "roi", "roi_mean_betas.csv"))

    roi_regression(roi_df, pheno, output)

    # Scatter plots for primary ROI results
    for contrast in ["negative_vs_neutral", "emotion_vs_neutral",
                     "positive_vs_neutral", "negative_vs_positive"]:
        for roi in ["amygdala", "dmPFC"]:
            scatter_roi(roi_df, pheno, contrast, roi, output)

    # ── 6. Whole-brain second-level ────────────────────────────────────────
    if not test_subject:   # skip whole-brain in test-subject mode (too slow)
        whole_brain_regression(all_betas, pheno, output)

    print(f"\n{'='*60}")
    print(f"Pipeline complete.  Outputs in: {output}")
    print(f"{'='*60}")


###
# ENTRY
###

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AOMIC PIOP1 neuroticism–face-perception pipeline")
    parser.add_argument("--test-subject", type=str, default=None, help="Run on a single subject only (e.g. sub-0001). Skips whole-brain.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute first-level maps even if output files already exist.")
    args = parser.parse_args()
    main(test_subject=args.test_subject, overwrite=args.overwrite)
