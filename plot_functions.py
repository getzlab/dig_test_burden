import pandas as pd
import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from adjustText import adjust_text

import os

def plot_volcano(pvals, logfc, labels, marker_labels=None, marker_legend_title=None, pval_bounds=None, ymax_vol=None, ymax_qq=None, alp=0.1, logfc_thr=0.5, logfc_xlabel=None, nlab='sig'):
    """
    Generates a volcano and a Q-Q plot for given p-values, logfold changes and labels.
    
    Parameters
    ----------
    pvals : numpy.ndarray
        non-adjusted p-values
    logfc : numpy.ndarray
        log2(fold change)
    labels : numpy.ndarray
        array of str, gene names
    marker_labels : numpy.ndarray
        array of str, group names encoded by marker types, number of groups must be less than 7
    marker_legend_title: str
        title on the legend displaying marker label groups
    pval_bounds : numpy.ndarray
        bound to be displayed around p-values, dimensions: pvals.shape[0] x 2
    ymax_vol : float
        vertical limit of volcano plot
    ymax_qq : float
        vertical limit of QQ plot
    alp : float (0 < alp < 1)
        false-discovery threshold
    logfc_thr : float > 0
        threshold of logfold change
    logfc_xlabel : str
        label displayed on the volcano plot's x axis
    nlab: str or int
        if 'sig' the only labels corresponding to significant pvals are plotted
        if int then all labels associated with the nlab smallest pvalues are plotted
    
    Returns
    -------
    None
    """

    # size of markers in scatter plot
    s = 10
    # width of markers
    lw = 1
    # significant
    col_sig = [1, 0, 0]
    # non-significant
    col_nsig = [0, 0, 1]
    # marker color on legend
    col_legend = [0] * 3
    # list of markers (at most 6 groups are possible in mutsig)
    marker_choices = ['s', '+', 'x', 'o', '^', '*']
    
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2)
    gs = GridSpec(1, 2, wspace=0.15)
    
    ind_sorted = np.argsort(pvals)
    pvals = pvals[ind_sorted].copy()
    nvals = pvals.shape[0]
    qvals = sp.stats.false_discovery_control(pvals)
    logq = -np.log10(qvals)
    labels = labels[ind_sorted].copy()
    logfc = logfc[ind_sorted].copy()
    if marker_labels is not None:
        marker_labels = marker_labels[ind_sorted].copy()
        marker_label_choices = np.unique(marker_labels)
        if marker_label_choices.shape[0] > len(marker_choices):
            raise ValueError('Too many groups in \"marker_labels\"! Please extend \"marker_choices\"!')
    
    ind_sig = qvals < alp
    ind_lfc = np.abs(logfc) > logfc_thr
    ind_kept = np.logical_and(ind_sig, ind_lfc)
    col = np.array([col_nsig] * nvals)
    col[ind_kept] = col_sig
    
    # volcano
    ax = fig.add_subplot(gs[:, 0])
    if ymax_vol is None:
        ind_capped = [False] * pvals.shape[0]
    else:
        ind_capped = logq > ymax_vol
        logq[ind_capped] = ymax_vol
    
    if marker_labels is None:
        ax.scatter(logfc, logq, s=s, c=col, edgecolors=None, alpha=1, zorder=11)
    else:
        for i, ml in enumerate(marker_label_choices):
            ind_marker = marker_labels == ml
            ax.scatter(logfc[ind_marker], logq[ind_marker], s=s, c=col[ind_marker], edgecolors=None, alpha=1, zorder=11, marker=marker_choices[i], linewidth=lw)

    if pval_bounds is not None:
        pval_bounds = pval_bounds[ind_sorted, :].copy()
        qval_bounds = np.c_[sp.stats.false_discovery_control(pval_bounds[:, 0]), sp.stats.false_discovery_control(pval_bounds[:, 1])]
        logq_bounds = -np.log10(qval_bounds).copy()
        for i in range(logfc.shape[0]):
            if not ind_capped[i]:
                ax.plot(np.array([logfc[i]] * 2), logq_bounds[i, :], color=col[i], alpha=0.15, zorder=10)
        
    xlim = ax.get_xlim()
    if isinstance(ymax_vol, float) or isinstance(ymax_vol, int):
        ylim = [0, ymax_vol * 1.05]
        ax.plot(xlim, [ymax_vol] * 2, linestyle='dashed', color='gray', zorder=0)
    else:
        ylim = [0, np.max(logq) * 1.05]
    ax.plot(xlim, [1, 1], 'gray', zorder=0)
    ax.plot([0, 0], ylim, 'gray', zorder=0)
    ax.plot([-logfc_thr] * 2, ylim, linestyle='dashed', color='gray', zorder=0)
    ax.plot([logfc_thr] * 2, ylim, linestyle='dashed', color='gray', zorder=0)
    if logfc_xlabel is None:
        ax.set_xlabel(r'$\log_{2}(\mathrm{fold change})$')
    else:
        ax.set_xlabel(logfc_xlabel)
    ax.set_ylabel(r'$-\log_{10}(\mathrm{FDR})$')
    ax.set_xlim(xlim)
    ax.set_ylim([0, ylim[1]])

    if nlab is not None:
        if isinstance(nlab, str):
            if nlab=='sig':
                X, Y, L = logfc[ind_kept], logq[ind_kept], labels[ind_kept]
        elif isinstance(nlab, int):
            X, Y, L = logfc[:nlab], logq[:nlab], labels[:nlab]
        txt = [plt.text(x, y, lab, zorder=12) for x, y, lab in zip(X, Y, L)]
        adjust_text(txt, arrowprops = { 'color' : 'k', "arrowstyle" : "-" })

    # QQ
    ax = fig.add_subplot(gs[:, 1])
    x = -np.log10(np.arange(1, nvals + 1) / (nvals + 1))
    y = -np.log10(pvals)
    if ymax_qq is None:
        ind_capped = [False] * pvals.shape[0]
    else:
        ind_capped = y > ymax_qq
    y_plot = y.copy()
    y_plot[ind_capped] = ymax_qq
    if marker_labels is None:
        ax.scatter(x, y_plot, marker='o', c=col, s=s, edgecolors=None, alpha=1, zorder=11)
    else:
        for i, ml in enumerate(marker_label_choices):
            ind_marker = marker_labels == ml
            ax.scatter(x[ind_marker], y_plot[ind_marker], s=s, c=col[ind_marker], edgecolors=None, alpha=1, zorder=11, marker=marker_choices[i], linewidth=lw, label=ml)
    if pval_bounds is not None:
        for i in range(pval_bounds.shape[0]):
            if not ind_capped[i]:
                ax.plot(np.array([x[i]] * 2), -np.log10(pval_bounds[i, :]), color=col[i], alpha=0.15, zorder=10)
    xlim = ax.get_xlim()
    if isinstance(ymax_qq, float) or isinstance(ymax_qq, int):
        ylim = [0, ymax_qq * 1.05]
        ax.plot([0, xlim[1]], [ymax_qq] * 2, linestyle='--', color='gray', zorder=0)
    else:
        ylim = [0, np.max(y_plot) * 1.05]    
    # add legend
    if marker_labels is not None:
        ax.legend()
        leg = ax.get_legend()
        for i in range(marker_label_choices.shape[0]):
            leg.legendHandles[i].set_color(col_legend)
        if marker_legend_title is not None:
            leg.set_title(marker_legend_title)
        
    ax.plot([0, xlim[1]], [0, xlim[1]], 'k--', zorder=1)
    ax.set_xlabel(r'$-\log_{10}(p_\mathrm{expected})$')
    ax.set_ylabel(r'$-\log_{10}(p_\mathrm{observed})$')
    ax.set_xlim([0, xlim[1]])
    ax.set_ylim([0, ylim[1]])
    
    plt.show()