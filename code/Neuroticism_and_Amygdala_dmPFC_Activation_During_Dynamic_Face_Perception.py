"""
How does trait neuroticism (NEO-FFI N subscale) relate to brain responses
during dynamic face perception (anger, contempt, joy, pride, neutral) in PIOP1?

H01: Higher neuroticism → increased amygdala & dmPFC activity for all emotionally loaded expressions (emotion > neutral)
H02: Higher neuroticism → increased amygdala & dmPFC for negative expressions (anger + contempt > neutral)
H03  Higher neuroticism → activity changes (any direction) for POSITIVE expressions (joy + pride > neutral)
H04  Neuroticism–amygdala/dmPFC association is stronger for negative than positive expressions (negative > positive)
"""

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
from nilearn.glm.second_level import SecondLevelModel, make_second_level_design_matrix
from nilearn import plotting
from nilearn.maskers import NiftiMasker
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.datasets import load_mni152_brain_mask
from nilearn.glm import threshold_stats_img
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


###
# Configuration
###

# Root of dataset
root = r"\\wsl.localhost\Ubuntu\home\ibe_maes\data\ds002785-fmriprep"

# fMRIPrep derivatives directory
fMRIPrep_derivatives = os.path.join(root, "derivatives", "fmriprep")

# Place for output files
output = os.path.join(root, "derivatives", "neuroticism_analysis")

# Acquisition parameters
tr             = 0.75  # seconds
n_volumes      = 330   # volumes per run (247.5 s / 0.75 s)
slice_time_ref = 0.5   # reference slice = midpoint of TR (0.5 × TR = 0.375 s), fMRIPrep PIOP1 did NOT apply slice-timing correction.

# GLM parameters
hrf_model      = "spm"     # SPM canonical double-gamma HRF
drift_model    = "cosine"  # cosine basis for high-pass filtering
high_pass      = 1 / 128   # SPM / nilearn default
noise_model    = "ar1"     # AR(1) temporal autocorrelation correction
smoothing_FWHM = 6.0       # mm FWHM Gaussian kernel (standard for group comparisons in small FOV)

# contrasts
contrasts = {"emotion_vs_neutral": "(anger + contempt + pride + joy) / 4 - neutral", #H01
             "negative_vs_neutral": "(anger + contempt) / 2 - neutral", #H02
             "positive_vs_neutral": "(pride + joy) / 2 - neutral", #H03
             "negative_vs_positive": "(anger + contempt - pride - joy) / 2"} #H04

# Quality control
max_mean_fd    = 0.5   # mm, exclude subject if mean FD > threshold
max_spike_frac = 0.20  # exclude if > 20 % volumes have FD > 0.5 mm

# Statistics
roi_fdr_alpha = 0.05  # FDR-BH threshold for ROI tests
wb_cluster_k  = 20    # minimum cluster size (voxels)

# Cremers et al. (2010) / Deng et al. (2019): dmPFC neuroticism cluster at MNI (-6, 50, 35) and ~(0, 44-52, 28);
# midpoint ≈ (0, 50, 30).
dmpfc_seed   = (0, 50, 30)
dmpfc_radius = 10  # mm

###
# Subject discovery and path utilities
###

# Sorted list of subjects
def discover_subjects(fmriprep_dir: str) -> list:
    pattern = os.path.join(fmriprep_dir, "sub-*", "func", "*task-faces*acq-mb3*space-MNI152NLin2009cAsym*desc-preproc_bold.nii.gz")
    hits = sorted(glob.glob(pattern))
    ids = sorted({os.path.basename(i).split("_")[0] for i in hits})
    print(f"Found {len(ids)} subjects with face-task BOLD.")
    return ids


# File paths for one subject
def subject_paths(sub_id: str) -> dict:
    func_deriv = os.path.join(fMRIPrep_derivatives, sub_id, "func")
    func_raw = os.path.join(root, sub_id, "func")
    base = f"{sub_id}_task-faces_acq-mb3"
    space = "space-MNI152NLin2009cAsym"

    return {
        "bold": os.path.join(func_deriv, f"{base}_{space}_desc-preproc_bold.nii.gz"),
        "mask": os.path.join(func_deriv, f"{base}_{space}_desc-brain_mask.nii.gz"),
        "confounds": (os.path.join(func_deriv, f"{base}_desc-confounds_timeseries.tsv")
                      if os.path.exists(os.path.join(func_deriv,
                          f"{base}_desc-confounds_timeseries.tsv")) # fMRIPrep ≥ 1.4 uses _desc-confounds_timeseries.tsv
                      else os.path.join(func_deriv, f"{base}_desc-confounds_regressors.tsv")), # older fMRIPrep uses _desc-confounds_regressors.tsv
        "events": os.path.join(func_raw, f"{base}_events.tsv")}


def paths_ok(paths: dict) -> bool:
    check = True
    for k, v in paths.items():
        if not os.path.exists(v):
            print(f"[MISSING] {k}: {v}")
            check = False
    return check


###
# Motion quality control
###

# Mean framewise displacement and spike fraction
def motion_stats(confounds_file: str) -> tuple:
    cf = pd.read_csv(confounds_file, sep="\t")
    col = "framewise_displacement"
    if col not in cf.columns:
        return np.nan, np.nan

    fd = cf[col].values

    # Non-steady-state volume indices to exclude from FD summary
    nss_cols = [c for c in cf.columns if c.startswith("non_steady_state_outlier")]
    if nss_cols:
        nss_mask = cf[nss_cols].any(axis=1).values  # True where NSS flagged
    else:
        nss_mask = np.zeros(len(fd), dtype=bool)

    # Exclude volume 0 (FD undefined) and NSS volumes
    valid_mask = ~nss_mask
    valid_mask[0] = False   # volume 0 always excluded

    fd_valid = fd[valid_mask]
    fd_valid = fd_valid[~np.isnan(fd_valid)]

    mfd = float(np.mean(fd_valid)) if len(fd_valid) > 0 else np.nan
    spikes = float(np.mean(fd_valid > 0.5)) if len(fd_valid) > 0 else np.nan
    return mfd, spikes


# 24-parameter motion model, ACompCor, and global signal
"""
 # High parameter since neurotic subjects tend to move more often (residual motion can correlate with neuroticism scores)
 # The Friston 24-parameter model removes spin-history (quadratic) effects that are common with head motion
 # The top-5 aCompCor components capture WM/CSF physiological noise more completely than a single mean signal
  # Columns selected (if present in file so check):
   # - 6 rigid body realignment parameters
   # - their first derivatives (suffix_derivative1) 
   # - quadratic terms (suffix_power2)
   # - derivative quadratic terms (_derivative1_power2) → 24 total motion
   # - top 5 aCompCor components (a_comp_cor_00 … _04)
    # - if this aCompCor is not available, use global signal instead
"""
def select_confounds(confounds_file: str) -> pd.DataFrame:
    cf = pd.read_csv(confounds_file, sep="\t")

    motion_params = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
                     "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
                     "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
                     "trans_x_power2", "trans_y_power2", "trans_z_power2",
                     "rot_x_power2", "rot_y_power2", "rot_z_power2",
                     "trans_x_derivative1_power2", "trans_y_derivative1_power2", "trans_z_derivative1_power2",
                     "rot_x_derivative1_power2", "rot_y_derivative1_power2", "rot_z_derivative1_power2"]

    acompcor_params = ["a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03", "a_comp_cor_04"]
    acompcor_available = [c for c in acompcor_params if c in cf.columns]
    if not acompcor_available:
        print("[confounds] WARNING: No aCompCor columns found. "
              "Consider adding global_signal as fallback.")

    # Non-steady-state outliers
    nss_cols = [c for c in cf.columns if c.startswith("non_steady_state_outlier")]

    # Volume-level spike regressors
    spike_cols = [c for c in cf.columns if c.startswith("motion_outlier")]

    wanted = motion_params + acompcor_params + nss_cols + spike_cols
    cols = [c for c in wanted if c in cf.columns]

    result = cf[cols].fillna(0)

    # Warn if any non-motion column has unexpected NaNs
    sus = [c for c in cols if cf[c].isna().sum() > 1]
    if sus:
        print(f"[confounds] WARNING: columns with >1 NaN (filled with 0): {sus}")

    print(f"[confounds] Selected {len(cols)} columns "
          f"({len(nss_cols)} NSS, {len(spike_cols)} spikes).")
    return result


###
# First-level GLM
###

# First-level GLM and 4 contrast beta maps for a subject
def run_first_level(sub_id: str, paths: dict, output_dir: str, overwrite: bool = False) -> dict:
    fl_dir = os.path.join(output_dir, "first_level")
    os.makedirs(fl_dir, exist_ok=True)
    beta_paths = {name: os.path.join(fl_dir, f"{sub_id}_{name}_cope.nii.gz") for name in contrasts}

    # If already computed
    if not overwrite and all(os.path.exists(p) for p in beta_paths.values()):
        print(f"[{sub_id}] First-level cached")
        return {name: nib.load(p) for name, p in beta_paths.items()}
    if not paths_ok(paths):
        print(f"[{sub_id}] SKIP")
        return {}

    # Load events
    events = pd.read_csv(paths["events"], sep="\t")[["onset", "duration", "trial_type"]]

    # Check if conditions match
    found = set(events["trial_type"].unique())
    expected = {"anger", "contempt", "joy", "pride", "neutral"}
    if not expected.issubset(found):
        print(f"[{sub_id}] WARNING: conditions found = {found}")

    confounds = select_confounds(paths["confounds"])

    glm = FirstLevelModel(t_r             = tr,
                          hrf_model       = hrf_model,
                          drift_model     = drift_model,
                          high_pass       = high_pass,
                          slice_time_ref  = slice_time_ref,
                          noise_model     = noise_model,
                          mask_img        = paths["mask"],
                          smoothing_fwhm  = smoothing_FWHM,
                          standardize     = False,
                          verbose         = 0)

    glm.fit(paths["bold"], events=events, confounds=confounds)

    beta_imgs = {}
    for name, formula in contrasts.items():
        beta_img = glm.compute_contrast(formula, output_type="effect_size")
        z_img = glm.compute_contrast(formula, output_type="z_score")
        nib.save(beta_img, beta_paths[name])
        z_path = os.path.join(fl_dir, f"{sub_id}_{name}_zstat.nii.gz")
        nib.save(z_img, z_path)
        beta_imgs[name] = beta_img

    print(f"[{sub_id}] First-level done.")
    return beta_imgs


###
# Phenotype loading
###

# participants.tsv from root directory
def load_participants(bids_root: str) -> pd.DataFrame:
    tsv = os.path.join(bids_root, "participants.tsv")
    if not os.path.exists(tsv):
        raise FileNotFoundError(f"participants.tsv not found at: {tsv}")

    df = pd.read_csv(tsv, sep="\t", na_values="n/a")
    df = df.rename(columns={"participant_id": "subject"})
    df = df.set_index("subject")

    # Standardize neuroticism name
    neuro_candidates = [c for c in df.columns if "neuro" in c.lower() or c.upper() in ("NEO_N", "NEO-N")]
    if not neuro_candidates:
        raise KeyError(f"Cannot find neuroticism column. Columns: {list(df.columns)}")
    df = df.rename(columns={neuro_candidates[0]: "neuroticism"})
    df["neuroticism_z"] = (df["neuroticism"] - df["neuroticism"].mean()) \
                           / df["neuroticism"].std()

    # Sex as binary variable
    if "sex" in df.columns:
        df["sex_code"] = (df["sex"].str.lower() == "male").astype(float)

    print(f"[phenotype] Loaded {len(df)} participants.")
    print(df["neuroticism"].describe().to_string())
    return df


# Per-subject mean FD into phenotype dataframe
 # Mean FD as second-level covariate since anxious subjects tend to move more; prevents motion artifacts inflating neuro-BOLD correlations
def add_mean_fd(pheno: pd.DataFrame, subject_ids: list) -> pd.DataFrame:
    records = []
    for sid in subject_ids:
        paths = subject_paths(sid)
        mfd, _ = motion_stats(paths["confounds"]) if os.path.exists(paths["confounds"]) else (np.nan, np.nan)
        records.append({"subject": sid, "mean_fd": mfd})

    fd_df  = pd.DataFrame(records).set_index("subject")
    result = pheno.join(fd_df, how="left")
    return result


###
# ROI masks (amygdala + dmPFC)
###

# amygdala (L/R) from Harvard-Oxford subcortical atlas;
# dmPFC as a 10 mm sphere at MNI (0, 50, 30).
def build_roi_masks() -> dict:
    ho_sub = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")

    def mask_from_exact_label(atlas, exact_keyword):
        labels  = atlas.labels
        indices = [i for i, l in enumerate(labels)
                   if exact_keyword.lower() in l.lower()]
        print(f"ROI labels matched [{exact_keyword}]: {[labels[i] for i in indices]}")
        data   = np.asarray(atlas.maps.dataobj)
        binary = np.isin(data, [i + 1 for i in indices]).astype(np.int8)
        return nib.Nifti1Image(binary, atlas.maps.affine)

    def make_sphere_mask(center_mni, radius_mm, reference_img):
        aff   = reference_img.affine
        shape = reference_img.shape[:3]
        # Voxel indices as MNI coordinates
        i, j, k    = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        vox_coords = np.array([i.ravel(), j.ravel(), k.ravel()])
        mni_coords = (aff[:3, :3] @ vox_coords + aff[:3, 3:4]).T
        # Euclidean distance from sphere center
        dist   = np.linalg.norm(mni_coords - np.array(center_mni), axis=1)
        sphere = (dist <= radius_mm).reshape(shape).astype(np.int8)
        return nib.Nifti1Image(sphere, aff)

    def print_centroid(img, label):
        data = np.asarray(img.dataobj)
        vox  = np.array(np.where(data > 0)).mean(axis=1)
        aff  = img.affine
        mni  = aff[:3, :3] @ vox + aff[:3, 3]
        print(f"[ROI centroid] {label}: MNI ({mni[0]:.1f}, {mni[1]:.1f}, {mni[2]:.1f})")

    amyg_L = mask_from_exact_label(ho_sub, "left amygdala")
    amyg_R = mask_from_exact_label(ho_sub, "right amygdala")

    # Amygdala image as reference space (2 mm MNI) for the sphere
    dmpfc  = make_sphere_mask(dmpfc_seed, dmpfc_radius, amyg_L)
    print(f"dmPFC sphere: centre {dmpfc_seed}, r = {dmpfc_radius} mm")

    masks = {"amygdala_L": amyg_L,
             "amygdala_R": amyg_R,
             "dmPFC":      dmpfc}

    for name, img in masks.items():
        print_centroid(img, name)

    return masks


# Mean beta estimate across all voxels in each ROI mask
def extract_roi_betas(sub_id: str, beta_imgs: dict, fitted_maskers: dict) -> list[dict]:
    rows = []
    for roi_name, masker in fitted_maskers.items():
        for contrast_name, beta_img in beta_imgs.items():
            vals = masker.transform(beta_img)
            rows.append({"subject": sub_id, "contrast": contrast_name, "roi": roi_name, "mean_beta": float(np.nanmean(vals))})
    return rows


###
# Second-level ROI analysis
###

# For every ROI x contrast combination: beta_roi ~ neuroticism_z + age + sex_code + mean_fd
# with FDR-BH corrections across all tests
def roi_regression(roi_df: pd.DataFrame, pheno: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    os.makedirs(os.path.join(output_dir, "roi"), exist_ok=True)

    covar_cols = [c for c in ["neuroticism_z", "age", "sex_code", "mean_fd"] if c in pheno.columns]
    results = []

    for contrast in contrasts:
        for roi in ["amygdala_L", "amygdala_R", "dmPFC"]:
            sub_df = roi_df[(roi_df["contrast"] == contrast) & (roi_df["roi"] == roi)].copy()
            df = sub_df.merge(pheno[covar_cols], left_on="subject", right_index=True, how="inner").dropna()

            if len(df) == 0:
                continue

            model = smf.ols("mean_beta ~ " + " + ".join(covar_cols), data=df).fit()

            results.append({"contrast": contrast, "roi": roi, "n": int(model.nobs), "beta_neuro": float(model.params["neuroticism_z"]),
                            "t_value": float(model.tvalues["neuroticism_z"]), "p_unc": float(model.pvalues["neuroticism_z"]),
                            "r_squared": float(model.rsquared)})

    res = pd.DataFrame(results)

    # One-tailed p for directional hypotheses, two-tailed for H03
    directional_contrasts = ["emotion_vs_neutral", "negative_vs_neutral", "negative_vs_positive"]

    res["p_for_fdr"] = np.nan
    for idx, row in res.iterrows():
        if pd.isna(row["p_unc"]) or pd.isna(row["beta_neuro"]):
            continue

        if row["contrast"] in directional_contrasts:
            if row["beta_neuro"] > 0:
                res.at[idx, "p_for_fdr"] = row["p_unc"] / 2.0
            else:
                res.at[idx, "p_for_fdr"] = 1.0 - row["p_unc"] / 2.0
        else:
            res.at[idx, "p_for_fdr"] = row["p_unc"]

    valid = res["p_for_fdr"].notna()
    res["p_fdr"] = np.nan
    if valid.sum() > 0:
        _, p_fdr_all, _, _ = multipletests(res.loc[valid, "p_for_fdr"].values, alpha=roi_fdr_alpha, method="fdr_bh")
        res.loc[valid, "p_fdr"] = p_fdr_all
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

    group_mask = load_mni152_brain_mask(resolution=2)

    covar_cols = [c for c in ["neuroticism_z", "age", "sex_code", "mean_fd"]
                  if c in pheno.columns]

    for contrast, paths_list in beta_paths_by_contrast.items():
        # Filter out subjects without full phenotype data
        valid = [(p, os.path.basename(p).split("_")[0])
                 for p in paths_list if os.path.exists(p)]
        subs = [s for _, s in valid]

        # Keep subjects with no NaN in covariates
        pheno_sub = pheno.reindex(subs)[covar_cols].dropna()
        keep_subs = list(pheno_sub.index)
        path_for_sub = {s: p for p, s in valid}
        imgs = [path_for_sub[s] for s in keep_subs]
        confounds_2nd = pheno_sub.loc[keep_subs, covar_cols].copy()
        confounds_2nd.insert(0, "subject_label", keep_subs)  # required column name
        dm = make_second_level_design_matrix(subjects_label=keep_subs, confounds=confounds_2nd)

        if len(imgs) < 10:
            print(f"[SKIP] {contrast}: only {len(imgs)} valid subjects.")
            continue

        second_level = SecondLevelModel(mask_img=group_mask, n_jobs=-1)
        second_level.fit(imgs, design_matrix=dm)

        z_img = second_level.compute_contrast(second_level_contrast="neuroticism_z", output_type="z_score")

        # Primary: FDR-corrected at q < 0.05
        try:
            fdr_img, fdr_z = threshold_stats_img(z_img, alpha=0.05, height_control="fdr")
        except Exception as e:
            print(f"[{contrast}] FDR thresholding failed: {e}, skipping FDR map.")
            fdr_img, fdr_z = None, None

        # Supplementary: uncorrected p < 0.001 + cluster extent
        try:
            unc_img, unc_z = threshold_stats_img(z_img, alpha=0.001, height_control="fpr", cluster_threshold=wb_cluster_k)
        except Exception as e:
            print(f"[{contrast}] Uncorrected thresholding failed: {e}, skipping unc map.")
            unc_img, unc_z = None, None

        base = os.path.join(output_dir, "second_level", contrast)
        nib.save(z_img, f"{base}_z.nii.gz")
        if fdr_img is not None:
            nib.save(fdr_img, f"{base}_z_fdr05.nii.gz")
        if unc_img is not None:
            nib.save(unc_img, f"{base}_z_p001_k{wb_cluster_k}.nii.gz")

        fdr_empty = fdr_img is None or not np.any(np.asarray(fdr_img.dataobj) != 0)
        unc_empty = unc_img is None or not np.any(np.asarray(unc_img.dataobj) != 0)

        ## Visualizations
        # Paths
        fdr_glass_path = os.path.join(output_dir, "figures", f"{contrast}_glass_neuroticism_FDR.png")
        unthr_glass_path = os.path.join(output_dir, "figures", f"{contrast}_glass_neuroticism_unthresholded.png")
        unc_glass_path = os.path.join(output_dir, "figures", f"{contrast}_glass_neuroticism_unc001.png")

        def _format_cbar(disp):
            if hasattr(disp, "_cbar") and disp._cbar is not None:
                disp._cbar.set_ticks([-5, -2.5, 0, 2.5, 5])
                disp._cbar.ax.tick_params(labelsize=9, colors="black")
                disp._cbar.outline.set_edgecolor("black")

        def _save_glass(img, out_path, threshold=None):
            try:
                fig = plt.figure(figsize=(14, 4), facecolor="white")
                disp = plotting.plot_glass_brain(img,
                                                 figure=fig,
                                                 display_mode="lyrz",
                                                 colorbar=True,
                                                 plot_abs=False,
                                                 cmap="cold_hot",
                                                 vmax=5.0,
                                                 symmetric_cbar=True,
                                                 threshold=threshold,
                                                 title=None,
                                                 annotate=False,
                                                 alpha=0.9,
                                                 black_bg=False)
                if hasattr(disp, "_cbar") and disp._cbar is not None:
                    disp._cbar.set_ticks([-5, -2.5, 0, 2.5, 5])
                    disp._cbar.set_ticklabels(["-5", "-2.5", "0", "2.5", "5"])
                    disp._cbar.ax.tick_params(labelsize=9, colors="black")
                    disp._cbar.outline.set_edgecolor("black")

                for ax in fig.axes:
                    ax.set_facecolor("white")

                _format_cbar(disp)
                disp.savefig(out_path, dpi=200, bbox_inches="tight")
                disp.close()
                plt.close(fig)

            except Exception as e:
                print(f"[WARNING] Could not save figure {out_path}: {e}")
            finally:
                plt.close("all")

        def _save_glass_untresholded(img, out_path, threshold=None):
            try:
                fig = plt.figure(figsize=(14, 4), facecolor="white")
                disp = plotting.plot_glass_brain(img,
                                                 figure=fig,
                                                 display_mode="lyrz",
                                                 colorbar=True,
                                                 plot_abs=False,
                                                 cmap="cold_hot",
                                                 vmax=5.0,
                                                 symmetric_cbar=True,
                                                 threshold=threshold,
                                                 title=None,
                                                 annotate=False,
                                                 alpha=0.9,
                                                 black_bg=True)
                if hasattr(disp, "_cbar") and disp._cbar is not None:
                    disp._cbar.set_ticks([-5, -2.5, 0, 2.5, 5])
                    disp._cbar.set_ticklabels(["-5", "-2.5", "0", "2.5", "5"])
                    disp._cbar.ax.tick_params(labelsize=9, colors="black")
                    disp._cbar.outline.set_edgecolor("black")

                for ax in fig.axes:
                    ax.set_facecolor("white")

                _format_cbar(disp)
                disp.savefig(out_path, dpi=200, bbox_inches="tight")
                disp.close()
                plt.close(fig)

            except Exception as e:
                print(f"[WARNING] Could not save figure {out_path}: {e}")
            finally:
                plt.close("all")

        # FDR q < 0.05
        if fdr_empty:
            print(f"[{contrast}] FDR map empty")
            _save_glass(fdr_img, fdr_glass_path, threshold=1e-12)
        else:
            _save_glass(fdr_img, fdr_glass_path, threshold=1e-12)

        # Unthresholded z-map
        _save_glass_untresholded(z_img, unthr_glass_path, threshold=None)

        # Uncorrected p < 0.001 + cluster extent
        if unc_empty:
            print(f"[{contrast}] Uncorrected map empty")
        else:
            _save_glass(unc_img, unc_glass_path, threshold=1e-12)


# Scatter: ROI mean beta vs. neuroticism score with regression line
def scatter_roi(roi_df: pd.DataFrame, pheno: pd.DataFrame, contrast: str, roi: str, output_dir: str):
    sub_df = roi_df[(roi_df["contrast"] == contrast) & (roi_df["roi"] == roi)].copy()
    if sub_df.empty:
        return

    covar_cols = [c for c in ["age", "sex_code", "mean_fd"] if c in pheno.columns]
    df = sub_df.merge(pheno[["neuroticism", "neuroticism_z"] + covar_cols], left_on="subject", right_index=True, how="inner").dropna()
    if len(df) < 5:
        return

    if covar_cols:
        nuisance_formula = " + ".join(covar_cols)
        res_y = smf.ols(f"mean_beta ~ {nuisance_formula}", data=df).fit().resid
        res_x = smf.ols(f"neuroticism ~ {nuisance_formula}", data=df).fit().resid
    else:
        res_y = df["mean_beta"]
        res_x = df["neuroticism"]

    partial_df = pd.DataFrame({"x": res_x.values, "y": res_y.values})
    fit = smf.ols("y ~ x", data=partial_df).fit()

    xs = np.linspace(res_x.min(), res_x.max(), 100)
    xs_df = pd.DataFrame({"x": xs})
    pred = fit.get_prediction(xs_df).summary_frame(alpha=0.05)

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="white")
    ax.set_facecolor("white")

    ax.scatter(res_x, res_y, s=30, alpha=0.50, color="#6f8fbf", edgecolors="none", zorder=3)
    ax.plot(xs, pred["mean"], color="#b54b4b", lw=2, zorder=4)
    ax.fill_between(xs, pred["mean_ci_lower"], pred["mean_ci_upper"], color="#b54b4b", alpha=0.10, zorder=2)

    roi_plot = {"amygdala_L": "Left amygdala",
                "amygdala_R": "Right amygdala",
                "dmPFC": "dmPFC"}
    ax.set_xlabel("Neuroticism", fontsize=10, color="black")
    ax.set_ylabel(f"Mean COPE [{roi_plot.get(roi, roi)}]", fontsize=10, color="black")

    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.margins(x=0.04, y=0.08)

    ax.tick_params(axis="both", labelsize=9, colors="#444444", width=0.7, length=2.5)

    fig.tight_layout()

    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    out = os.path.join(output_dir, "figures", f"scatter_{contrast}_{roi}.png")
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter saved: {out}")

###
# MAIN
###

def main(test_subject: str = None, overwrite: bool = False):

    # Output directories
    for d in ["first_level", "second_level", "roi", "figures"]:
        os.makedirs(os.path.join(output, d), exist_ok=True)

    # 1. Subject discovery
    if test_subject:
        all_subs = [test_subject]
        print(f"Running single subject: {test_subject}")
    else:
        all_subs = discover_subjects(fMRIPrep_derivatives)

    # 2. Load phenotype
    pheno = load_participants(root)
    pheno = add_mean_fd(pheno, all_subs)

    # Only subjects with both imaging and phenotype
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

    # Standardize neuroticism
    neuro_mean = pheno.loc[included, "neuroticism"].mean()
    neuro_std  = pheno.loc[included, "neuroticism"].std()
    pheno["neuroticism_z"] = (pheno["neuroticism"] - neuro_mean) / neuro_std
    print(f"[phenotype] Neuroticism z-score re-standardised in N={len(included)} included subjects.")

    # 4. First-level GLM loop
    roi_masks = build_roi_masks()

    # Maskers for ROI beta extraction
    fitted_maskers = {}
    for roi_name, mask_img in roi_masks.items():
        m = NiftiMasker(mask_img=mask_img, standardize=False)
        m.fit()
        fitted_maskers[roi_name] = m

    roi_rows  = []
    all_betas = {}

    for cname in contrasts:
        all_betas[cname] = []

    # Load already-extracted ROI betas so we can skip those subjects
    roi_cache_path = os.path.join(output, "roi", "roi_mean_betas.csv")
    cached_subs = set()
    if os.path.exists(roi_cache_path):
        cached_df = pd.read_csv(roi_cache_path)
        cache_mtime = os.path.getmtime(roi_cache_path)
        # Drop subjects whose cope files have been modified since the cache was written
        fl_dir = os.path.join(output, "first_level")
        stale = set()
        for sub in cached_df["subject"].unique():
            cope_files = [os.path.join(fl_dir, f"{sub}_{cname}_cope.nii.gz") for cname in contrasts]
            if any(os.path.exists(p) and os.path.getmtime(p) > cache_mtime for p in cope_files):
                stale.add(sub)
        if stale:
            print(f"[cache] {len(stale)} subject(s) have updated cope files, will reprocess: {sorted(stale)}")
            cached_df = cached_df[~cached_df["subject"].isin(stale)]
        roi_rows = cached_df.to_dict("records")
        cached_subs = set(cached_df["subject"].unique())
        print(f"[cache] Loaded ROI betas for {len(cached_subs)} subjects from cache.")

    for sub in included:
        # Collect beta paths for whole-brain regardless (files already on disk)
        fl_dir = os.path.join(output, "first_level")
        beta_paths_sub = {cname: os.path.join(fl_dir, f"{sub}_{cname}_cope.nii.gz") for cname in contrasts}
        if all(os.path.exists(p) for p in beta_paths_sub.values()):
            for cname, bpath in beta_paths_sub.items():
                all_betas[cname].append(bpath)

        if sub in cached_subs and not overwrite:
            print(f"[{sub}] ROI betas cached, skipping image load.")
            continue

        print(f"\n=== {sub} ===")
        paths = subject_paths(sub)
        beta_imgs = run_first_level(sub, paths, output, overwrite=overwrite)
        if not beta_imgs:
            continue

        # ROI mean betas
        rows = extract_roi_betas(sub, beta_imgs, fitted_maskers)
        roi_rows.extend(rows)

    # 5. ROI second-level
    if len(roi_rows) == 0:
        print("[ERROR] No subjects completed first-level")
        return

    roi_df = pd.DataFrame(roi_rows)
    roi_df.to_csv(os.path.join(output, "roi", "roi_mean_betas.csv"))

    roi_regression(roi_df, pheno, output)

    # Scatter plots for primary ROI results
    for contrast in ["negative_vs_neutral", "emotion_vs_neutral", "positive_vs_neutral", "negative_vs_positive"]:
        for roi in ["amygdala_L", "amygdala_R", "dmPFC"]:
            scatter_roi(roi_df, pheno, contrast, roi, output)

    # 6. Whole-brain second-level
    if not test_subject: # skip whole-brain in test-subject mode (too slow)
        whole_brain_regression(all_betas, pheno, output)

    print(f"\n{'='*60}")
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
