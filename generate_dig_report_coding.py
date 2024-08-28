import argparse
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import scipy as sp
from statsmodels.stats.multitest import fdrcorrection
import json

# threshold for log2 fold change (observed vs expected)
logfc_thr = 1
# minimum number of rows (genes) to display in the table
n_rows_min = 50
# horizontal buffer for the scatter plots
hor_buffer = 0.05
# buffer for the number of rows in the table
n_rows_buffer = 0.5
# maximum value along the vertical axis for the volcano and Q-Q plots
ymax = 16

# dictionaries for the two dropdowns
burden_type = {
    'Total': 'BURDEN',
    'Sample-wise': 'BURDEN_SAMPLE',
}
mut_type = {
    'Indel': 'INDEL',
    'Nonsynonymous SNV': 'NONSYN',
    'Missense SNV': 'MIS',
    'Nonsense SNV': 'NONS',
    'Truncating SNV': 'TRUNC',
    'Splice site SNV': 'SPL',
    'Synonymous SNV': 'SYN',
}


def nb_pvalue_greater_midp(k, alpha, p):
    """ Calculate an UPPER TAIL p-value for a negative binomial distribution
        with a midp correction
    """
    return 0.5 * sp.stats.nbinom.pmf(k, alpha, p) + sp.special.betainc(k+1, alpha, 1-p)


def nb_pvalue_lower(k, alpha, p):
    """ Calculate the upper bound for the p-value of a negative binomial distribution.
    """
    return sp.special.betainc(k+1, alpha, 1-p)


def nb_pvalue_upper(k, alpha, p):
    """ Calculate the upper bound for the p-value of a negative binomial distribution.
    """
    ind_0 = k == 0
    pvals = np.zeros_like(alpha)
    pvals[ind_0] = sp.stats.nbinom.pmf(k[ind_0], alpha[ind_0], p[ind_0])
    pvals[~ind_0] = sp.special.betainc(k[~ind_0], alpha[~ind_0], 1-p[~ind_0])
    return pvals


def reformat_numbers(x, format='{:.3E}'):
    """
    Reformat numbers in an array to a specific format
    """
    return [format.format(n) for n in x]


def generate_dig_report(path_to_dig_results, dir_output, prefix_output=None, alp=0.1):
    # Output from DIGDriver
    df = pd.read_csv(path_to_dig_results, sep='\t')
    # Computing lower and upper bounds for the p-values
    muts_ts = list(mut_type.values())
    muts_ts.remove('INDEL')
    for m in muts_ts:
        # total burden
        df['PVAL_' + m + '_BURDEN_recalc'] = nb_pvalue_greater_midp(
            df['OBS_' + m],
            df.ALPHA,
            1 / (df.THETA * df['Pi_' + m] + 1)
        )
        df['PVAL_' + m + '_BURDEN_lower'] = nb_pvalue_lower(
            df['OBS_' + m],
            df.ALPHA,
            1 / (df.THETA * df['Pi_' + m] + 1)
        )
        df['PVAL_' + m + '_BURDEN_upper'] = nb_pvalue_upper(
            df['OBS_' + m],
            df.ALPHA,
            1 / (df.THETA * df['Pi_' + m] + 1)
        )
        # sample-wise burden
        df['PVAL_' + m + '_BURDEN_SAMPLE_recalc'] = nb_pvalue_greater_midp(
            df['N_SAMP_' + m],
            df.ALPHA,
            1 / (df.THETA * df['Pi_' + m] + 1)
        )
        df['PVAL_' + m + '_BURDEN_SAMPLE_lower'] = nb_pvalue_lower(
            df['N_SAMP_' + m],
            df.ALPHA,
            1 / (df.THETA * df['Pi_' + m] + 1)
        )
        df['PVAL_' + m + '_BURDEN_SAMPLE_upper'] = nb_pvalue_upper(
            df['N_SAMP_' + m],
            df.ALPHA,
            1 / (df.THETA * df['Pi_' + m] + 1)
        )
    # total indel burden
    df['PVAL_INDEL_BURDEN_recalc'] = nb_pvalue_greater_midp(
        df.OBS_INDEL,
        df.ALPHA,
        1 / (df.THETA_INDEL * df.Pi_INDEL + 1)
    )
    df['PVAL_INDEL_BURDEN_lower'] = nb_pvalue_lower(
        df.OBS_INDEL,
        df.ALPHA,
        1 / (df.THETA_INDEL * df.Pi_INDEL + 1)
    )
    df['PVAL_INDEL_BURDEN_upper'] = nb_pvalue_upper(
        df.OBS_INDEL,
        df.ALPHA,
        1 / (df.THETA_INDEL * df.Pi_INDEL + 1)
    )

    def generate_plot_data(mut, bur):
        """
        Given a mutation type and a burden type, generate the data for the volcano plot, Q-Q plot, and table plot
        :param mut: str, mutation type
        :param bur: str, burden type
        """
        col_obs = 'OBS' if (bur == 'BURDEN') else 'N_SAMP'
        cols_lfc = [e + '_' + mut for e in [col_obs, 'EXP']]
        col_pval = 'PVAL_' + mut + '_' + bur

        # subsetting to only those genes for which the expected nuber of mutations is greater than 0
        ind_keep = df[cols_lfc[1]] > 0
        df_kept = df.loc[ind_keep].copy()

        df_kept['LOGFC_' + mut + '_' + bur] = np.log2(df_kept[cols_lfc[0]] / df_kept[cols_lfc[1]] + 1)
        df_kept['FDR_' + mut + '_' + bur] = fdrcorrection(df_kept[col_pval])[1]
        df_kept['FDR_' + mut + '_' + bur + '_recalc'] = fdrcorrection(df_kept[col_pval + '_recalc'])[1]
        df_kept['FDR_' + mut + '_' + bur + '_lower'] = fdrcorrection(df_kept[col_pval + '_lower'])[1]
        df_kept['FDR_' + mut + '_' + bur + '_upper'] = fdrcorrection(df_kept[col_pval + '_upper'])[1]
        df_kept['dNdS_OBS'] = df_kept[col_obs + '_NONSYN'] / df_kept['OBS_SYN']
        df_kept['dNdS_EXP'] = df_kept['EXP_NONSYN'] / df_kept['EXP_SYN']
        df_kept = df_kept.sort_values(by='PVAL_' + mut + '_' + bur, ignore_index=True)
        df_kept['RANK'] = df_kept.index + 1

        labels = df_kept.GENE.to_numpy()
        logfc = df_kept['LOGFC_' + mut + '_' + bur].to_numpy()
        pvals = df_kept['PVAL_' + mut + '_' + bur + '_recalc'].to_numpy()
        qvals = df_kept['FDR_' + mut + '_' + bur + '_recalc'].to_numpy()
        pval_bounds = df_kept[['PVAL_' + mut + '_' + bur + '_lower', 'PVAL_' + mut + '_' + bur + '_upper']].to_numpy()
        logq = -np.log10(qvals)
        logq_bounds = -np.log10(df_kept[['FDR_' + mut + '_' + bur + '_lower', 'FDR_' + mut + '_' + bur + '_upper']].to_numpy())

        # Determine significant points
        ind_sig = qvals < alp
        ind_lfc = np.abs(logfc) > logfc_thr
        ind_kept = np.logical_and(ind_sig, ind_lfc)

        # Dataframe for the plot
        cols_kept = [
            'RANK', 'GENE', 'CHROM', 'GENE_LENGTH',
            'PVAL_' + mut + '_' + bur,
            'FDR_' + mut + '_' + bur,
            col_obs + '_' + mut,
            'EXP_' + mut,
            'MU', 'SIGMA', 'dNdS_OBS', 'dNdS_EXP', 'FLAG']
        n_rows = max(n_rows_min, int(np.sum(ind_sig) * (1 + n_rows_buffer)))
        df_plot = df_kept.iloc[:n_rows][cols_kept].copy()
        df_plot.rename(columns={
            'GENE_LENGTH': 'LENGTH',
            'PVAL_' + mut + '_' + bur: 'PVAL',
            'FDR_' + mut + '_' + bur: 'FDR',
            col_obs + '_' + mut: 'OBS',
            'EXP_' + mut: 'EXP'
        }, inplace=True)

        for col in ['PVAL', 'FDR']:
            df_plot[col] = reformat_numbers(df_plot[col].to_numpy())
        for col in ['MU', 'SIGMA']:
            df_plot[col] = reformat_numbers(df_plot[col].to_numpy(), format='{:.2f}')
        for col in ['dNdS_EXP', 'EXP']:
            df_plot[col] = reformat_numbers(df_plot[col].to_numpy(), format='{:.3f}')

        df_plot['OBS'] = df_plot['OBS'].astype(int)
        is_inf = np.logical_or(np.isinf(df_plot.dNdS_OBS.to_numpy()), np.isnan(df_plot.dNdS_OBS.to_numpy()))
        df_plot['dNdS_OBS'] = df_plot['dNdS_OBS'].astype(str)
        df_plot.loc[is_inf, 'dNdS_OBS'] = 'NA'
        df_plot['FLAG'] = df_plot['FLAG'].astype(str).str.upper()
        is_flagged = df_plot.FLAG == 'TRUE'
        df_plot.loc[is_flagged, 'GENE'] = df_plot['GENE'][is_flagged] + '*'

        # Generate table figure
        headerColor = 'grey'
        rowEvenColor = 'lightgrey'
        rowOddColor = 'white'
        cols_specific = ['PVAL', 'FDR', 'OBS', 'EXP']
        # Making the significant rows bold
        df_plot = df_plot.astype(str)
        for i in range(df_plot.shape[0]):
            if ind_kept[i]:
                df_plot.loc[i, :] = '<b>' + df_plot.loc[i, :].astype(str) + '</b>'

        table_fig = go.Figure(data=[go.Table(
            header=dict(values=['<b><i>' + col + '</i></b>' if col in cols_specific else '<b>' + col + '</b>' for col in
                                df_plot.columns],
                        line_color='darkslategray',
                        fill_color=headerColor,
                        align=['left', 'center'],
                        font=dict(color='white', size=12)),
            cells=dict(values=[df_plot[col].tolist() for col in df_plot.columns],
                       line_color='darkslategray',
                       fill_color=[[rowOddColor if i % 2 == 0 else rowEvenColor for i in range(df_plot.shape[0])]],
                       align=['left', 'center'],
                       font=dict(color='darkslategray', size=11)))
        ])
        table_fig.update_layout(
            annotations=[
                dict(
                    text="*FLAG=TRUE: At least one kilobase-scale region overlapped by gene is <50% uniquely mappable or in the top 99.99th percentile of mutation rate.",
                    x=0,
                    y=-0.15,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    align="left",
                    valign="top",
                    font=dict(size=12)
                )
            ]
        )

        return df_kept, pvals, pval_bounds, logfc, logq, logq_bounds, labels, ind_kept, table_fig

    # generate the data and table for the default values
    df_kept, pvals, pval_bounds, logfc, logq, logq_bounds, labels, ind_kept, table_fig = generate_plot_data('MIS', 'BURDEN_SAMPLE')

    # Update the HTML template to include the additional text below the table
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DIG for Coding Genes</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            .container {{
                display: flex;
            }}
            .plot {{
                margin: 10px;
            }}
            #dnds-plot {{
                width: 33%;
            }}
            #volcano-plot {{
                width: 50%;
            }}
            #qq-plot {{
                width: 50%;
            }}
            #table-plot {{
                width: 100%;
            }}
            .figures-container {{
                display: flex;
                justify-content: space-between;
            }}
            .figure-column {{
                width: 33%;
            }}
        </style>
    </head>
    <body>
        <h1>DIG Driver Results for Coding Genes</h1>
    
        <label for="mut-type">Select Mutation Type:</label>
        <select id="mut-type" onchange="updatePlot()">
            {mut_options}
        </select>
    
        <label for="burden-type">Select Burden Type:</label>
        <select id="burden-type" onchange="updatePlot()">
            {burden_options}
        </select>
    
        <h2 id="plot-title"></h2>
    
        <div class="container">
            <div id="volcano-plot" class="plot"></div>
            <div id="qq-plot" class="plot"></div>
        </div>
        <div id="table-plot" class="plot"></div>
        <div class="figures-container">
            <div class="figure-column">{fig_mu_html}</div>
            <div class="figure-column">{fig_sigma_html}</div>
            <div id="dnds-plot" class="plot"></div>
        </div>
    
        <script>
            var plotData = {plot_data};
    
            function updatePlot() {{
                var mutTypeKey = document.getElementById("mut-type").value;
                var burdenTypeKey = document.getElementById("burden-type").value;
    
                var data = plotData[mutTypeKey + '_' + burdenTypeKey];
    
                // Update Volcano Plot
                var volcanoData = data.volcano;
                Plotly.react('volcano-plot', volcanoData);
    
                // Update Q-Q Plot
                var qqData = data.qq;
                Plotly.react('qq-plot', qqData);
                
                // Update dNdS Plot
                var dndsData = data.dnds;
                Plotly.react('dnds-plot', dndsData);
    
                // Update Table Plot
                var tableData = data.table;
                Plotly.react('table-plot', tableData);
    
                // Update Header
                var headerText = burdenTypeKey + ' Mutation Burden of ' + mutTypeKey + 's';
                document.getElementById("plot-title").textContent = headerText;
            }}
    
            // Initial plot
            updatePlot();
        </script>
    </body>
    </html>
    """

    # generate the dropdown options
    mut_options = "\n".join([f'<option value="{key}">{key}</option>' for key in mut_type.keys()])
    burden_options = "\n".join([f'<option value="{key}">{key}</option>' for key in burden_type.keys()])

    # prepare plot data for all combinations of mut_type and burden_type dropdown options
    plot_data = {}
    for mut_key, mut_val in mut_type.items():
        for bur_key, bur_val in burden_type.items():
            if not (mut_key == 'Indel' and bur_key == 'Sample-wise'):
                df_kept, pvals, pval_bounds, logfc, logq, logq_bounds, labels, ind_kept, table_fig = generate_plot_data(mut_val, bur_val)

                # Volcano Plot
                logq_upper = logq_bounds[:, 0] - logq
                logq_lower = logq - logq_bounds[:, 1]
                ind_capped = logq > ymax

                volcano_fig = go.Figure()
                volcano_fig.add_trace(
                    go.Scatter(
                        x=logfc[~ind_kept].tolist(),
                        y=logq[~ind_kept].tolist(),
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=np.round(logq_upper[~ind_kept], 3).tolist(),
                            arrayminus=np.round(logq_lower[~ind_kept], 3).tolist(),
                            thickness=0.3,
                            width=2
                        ),
                        mode='markers',
                        marker=dict(color='black', opacity=0.5),
                        text=labels[~ind_kept].tolist(),
                        name='Non-significant',
                        showlegend=False,
                        xhoverformat='.3f',
                        yhoverformat='.3f',
                    )
                )

                ind_ncapped = np.logical_and(ind_kept, ~ind_capped)

                volcano_fig.add_trace(
                    go.Scatter(
                        x=logfc[ind_ncapped].tolist(),
                        y=logq[ind_ncapped].tolist(),
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=np.round(logq_upper[ind_ncapped], 3).tolist(),
                            arrayminus=np.round(logq_lower[ind_ncapped], 3).tolist(),
                            thickness=0.3,
                            width=2
                        ),
                        mode='markers',
                        marker=dict(color='red', opacity=0.7),
                        text=labels[ind_ncapped].tolist(),
                        name='Significant',
                        showlegend=False,
                        xhoverformat='.3f',
                        yhoverformat='.3f'
                    )
                )

                labels_capped = labels[ind_capped].tolist()
                logfc_capped = logfc[ind_capped].tolist()
                logq_capped = logq[ind_capped].tolist()
                logq_upper_capped = logq_upper[ind_capped].tolist()
                logq_lower_capped = logq_lower[ind_capped].tolist()
                labels_capped = [(f"({logfc_capped[i]:.3f}, {logq_capped[i]:.3f} +{logq_upper_capped[i]:.3f} / -{logq_lower_capped[i]:.3f})<br>{l}") for i, l in enumerate(labels_capped)]

                volcano_fig.add_trace(
                    go.Scatter(
                        x=logfc[ind_capped].tolist(),
                        y=[ymax] * sum(ind_capped),
                        mode='markers',
                        marker=dict(color='red', opacity=0.7),
                        text=labels_capped,
                        name='Significant',
                        showlegend=False,
                        hoverinfo='name+text',
                    )
                )

                volcano_fig.add_trace(
                    go.Scatter(x=[1, 1], y=[0, np.max(logq) * (1 + hor_buffer)], mode='lines',
                               line=dict(dash='dash', color='gray', width=0.75), showlegend=False)
                )
                volcano_fig.add_trace(
                    go.Scatter(x=[0, np.max(logfc) * (1 + hor_buffer)], y=[-np.log10(alp), -np.log10(alp)],
                               mode='lines',
                               line=dict(dash='dash', color='gray', width=2.5), showlegend=False)
                )
                volcano_fig.add_trace(
                    go.Scatter(x=[0, np.max(logfc) * (1 + hor_buffer)], y=[ymax] * 2,
                               mode='lines',
                               line=dict(dash='dash', color='gray', width=0.75), showlegend=False)
                )

                volcano_fig.update_layout(
                    title='Observed/Expected counts vs. False Discovery Rate:',
                    xaxis_title='Log2(Observed/Expected + 1)',
                    yaxis_title='-Log10(FDR)',
                    xaxis=dict(range=[0, np.max(logfc) * (1 + hor_buffer)]),
                    yaxis=dict(range=[0, min(np.max(logq), ymax) * (1 + hor_buffer)]),
                    template='plotly_white'
                )

                # Q-Q Plot
                x = -np.log10(np.arange(1, len(pvals) + 1) / (len(pvals) + 1))
                y = -np.log10(pvals)
                y_upper = -np.log10(pval_bounds[:, 0]) - y
                y_lower = y + np.log10(pval_bounds[:, 1])
                ind_capped = y > ymax

                qq_fig = go.Figure()

                qq_fig.add_trace(
                    go.Scatter(
                        x=x[~ind_kept].tolist(),
                        y=y[~ind_kept].tolist(),
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=np.round(y_upper[~ind_kept], 3).tolist(),
                            arrayminus=np.round(y_lower[~ind_kept], 3).tolist(),
                            thickness=0.3,
                            width=2
                        ),
                        mode='markers',
                        marker=dict(color='black', opacity=0.5),
                        text=labels[~ind_kept].tolist(),
                        name='Non-significant',
                        showlegend=False,
                        xhoverformat='.3f',
                        yhoverformat='.3f'
                    )
                )

                ind_ncapped = np.logical_and(ind_kept, ~ind_capped)

                qq_fig.add_trace(
                    go.Scatter(
                        x=x[ind_ncapped].tolist(),
                        y=y[ind_ncapped].tolist(),
                        error_y=dict(
                            type='data',
                            symmetric=False,
                            array=np.round(y_upper[ind_ncapped], 3).tolist(),
                            arrayminus=np.round(y_lower[ind_ncapped], 3).tolist(),
                            thickness=0.3,
                            width=2
                        ),
                        mode='markers',
                        marker=dict(color='red', opacity=0.7),
                        text=labels[ind_ncapped].tolist(),
                        name='Significant',
                        showlegend=False,
                        xhoverformat='.3f',
                        yhoverformat='.3f'
                    )
                )

                labels_capped = labels[ind_capped].tolist()
                x_capped = x[ind_capped].tolist()
                y_capped = y[ind_capped].tolist()
                y_upper_capped = y_upper[ind_capped].tolist()
                y_lower_capped = y_lower[ind_capped].tolist()
                labels_capped = [(f"({x_capped[i]:.3f}, {y_capped[i]:.3f} +{y_upper_capped[i]:.3f} / -{y_lower_capped[i]:.3f})<br>{l}") for i, l in enumerate(labels_capped)]

                qq_fig.add_trace(
                    go.Scatter(
                        x=x_capped,
                        y=[ymax] * sum(ind_capped),
                        mode='markers',
                        marker=dict(color='red', opacity=0.7),
                        text=labels_capped,
                        name='Significant',
                        showlegend=False,
                        hoverinfo='name+text',
                    )
                )

                qq_fig.add_trace(
                    go.Scatter(
                        x=[0, np.max(x) * (1 + hor_buffer)],
                        y=[0, np.max(x) * (1 + hor_buffer)],
                        mode='lines',
                        line=dict(dash='dash', color='gray', width=2.5),
                        showlegend=False
                    )
                )

                qq_fig.add_trace(
                    go.Scatter(x=[0, np.max(x) * (1 + hor_buffer)], y=[ymax] * 2,
                               mode='lines',
                               line=dict(dash='dash', color='gray', width=0.75), showlegend=False)
                )

                qq_fig.update_layout(
                    title='QQ-Plot of P-values:',
                    xaxis_title='Expected -Log10(P-value)',
                    yaxis_title='Observed -Log10(P-value)',
                    xaxis=dict(range=[0, np.max(x) * (1 + hor_buffer)]),
                    yaxis=dict(range=[0, min(np.max(y), ymax) * (1 + hor_buffer)]),
                    template='plotly_white'
                )
                # dNdS Plot
                ind_isna = np.logical_or(df_kept['dNdS_EXP'].isna(), df_kept['dNdS_OBS'].isna())
                dnds_obs = df_kept['dNdS_OBS'][~ind_isna].to_numpy()
                dnds_exp = df_kept['dNdS_EXP'][~ind_isna].to_numpy()
                dnds_labels = df_kept['GENE'][~ind_isna].to_numpy()
                xmax = np.max(dnds_exp) * (1 + hor_buffer)
                ind_psel = dnds_obs > dnds_exp

                dnds_fig = go.Figure()
                dnds_fig.add_trace(
                    go.Scatter(
                        x=dnds_exp[~ind_psel].tolist(),
                        y=dnds_obs[~ind_psel].tolist(),
                        mode='markers',
                        marker=dict(color='black', opacity=0.5),
                        text=dnds_labels[~ind_psel].tolist(),
                        name='Lower than expected',
                        showlegend=False
                    )
                )
                dnds_fig.add_trace(
                    go.Scatter(
                        x=dnds_exp[ind_psel].tolist(),
                        y=dnds_obs[ind_psel].tolist(),
                        mode='markers',
                        marker=dict(color='red', opacity=0.7),
                        text=dnds_labels[ind_psel].tolist(),
                        name='Higher than expected',
                        showlegend=False
                    )
                )
                dnds_fig.add_trace(
                    go.Scatter(x=[0, xmax], y=[0, xmax], mode='lines',
                               line=dict(dash='dash', color='gray'), showlegend=False))
                dnds_fig.update_layout(
                    title='Observed vs Expected dNdS ratios:',
                    xaxis_title='Expected dNdS',
                    yaxis_title='Observed dNdS',
                    xaxis=dict(range=[0, xmax]),
                    yaxis=dict(range=[0, xmax]),
                    template='plotly_white'
                )

                # Save figures as separate data
                plot_data[f"{mut_key}_{bur_key}"] = {
                    'volcano': volcano_fig.to_dict(),
                    'qq': qq_fig.to_dict(),
                    'dnds': dnds_fig.to_dict(),
                    'table': table_fig.to_dict()
                }

    # convert plot data to JSON-like structure
    plot_data_json = json.dumps(plot_data)

    # generate histograms for MU and SIGMA
    fig_mu = go.Figure(data=[go.Histogram(
        x=df_kept['MU'],
        opacity=0.8,
        marker_color='gray'
    )])
    fig_mu.update_layout(
        title='Mean of GP model:',
        xaxis_title='MU (mutations per kilobase)',
        yaxis_title='Number of genes',
        template='plotly_white',
        yaxis_type='log'
    )

    fig_sigma = go.Figure(data=[go.Histogram(
        x=df_kept['SIGMA'],
        opacity=0.8,
        marker_color='gray'
    )])
    fig_sigma.update_layout(
        title='Standard deviation of GP model:',
        xaxis_title='SIGMA (mutations per kilobase)',
        yaxis_title='Number of genes',
        template='plotly_white',
        yaxis_type='log'
    )

    # save static figures as HTML divs
    fig_mu_html = fig_mu.to_html(full_html=False, include_plotlyjs='cdn')
    fig_sigma_html = fig_sigma.to_html(full_html=False, include_plotlyjs='cdn')

    # combine everything into the final HTML
    html_content = html_content.format(
        mut_options=mut_options,
        burden_options=burden_options,
        plot_data=plot_data_json,
        fig_mu_html=fig_mu_html,
        fig_sigma_html=fig_sigma_html
    )

    # save to an HTML file
    with open(dir_output + 'DIG_report_coding' + ('' if (prefix_output is None) else '_' + prefix_output) + '.html',
              'w') as f:
        f.write(html_content)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate DIG report for coding genes.")
    parser.add_argument("path_to_dig_results", type=str, help="Path to the DIG results file.")
    parser.add_argument("dir_output", type=str, help="Output directory.")
    parser.add_argument("--prefix_output", type=str, default=None, help="Prefix for the output file.")
    parser.add_argument("--alp", type=float, default=0.1, help="Significance level (default: 0.1).")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    # Call the function with parsed arguments
    generate_dig_report(args.path_to_dig_results, args.dir_output, args.prefix_output, args.alp)
