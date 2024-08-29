import argparse
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

# properties of significant points
col_sig = 'red'
opac_sig = 0.7
# properties of non-significant points
col_nonsig = 'black'
opac_nonsig = 0.5
# properties of thinner lines
thk_thin = 0.75
col_thin = 'gray'
typ_thin = 'dash'
# properties of thicker lines
thk_thick = 2.5
col_thick = 'gray'
typ_thick = 'dash'
# properties of error bars
wid_err = 2
thk_err = 0.3
# properties of the bar plot
col_bar = 'gray'
opac_bar = 0.8

burden_type = {
    'Total': 'BURDEN',
    'Sample-wise': 'SAMPLE_BURDEN',
}
mut_type = {
    'Indel': 'INDEL',
    'SNV': 'SNV',
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


def generate_dig_report(path_to_dig_results, dir_output, name_interval_set, prefix_output=None, alp=0.1):
    # Output from DIGDriver
    df = pd.read_csv(path_to_dig_results, sep='\t')
    # Computing lower and upper bounds for the p-values
    pfxs_obs = ['SNV', 'INDEL', 'SAMPLES']
    pfxs_pval = ['SNV', 'INDEL', 'SAMPLE']
    pfxs_pi = ['SUM', 'INDEL', 'SUM']
    pfxs_at = ['', '_INDEL', '']

    for i in range(len(pfxs_obs)):
        df['PVAL_' + pfxs_pval[i] + '_BURDEN_recalc'] = nb_pvalue_greater_midp(
            df['OBS_' + pfxs_obs[i]],
            df['ALPHA' + pfxs_at[i]],
            1 / (df['THETA' + pfxs_at[i]] * df['Pi_' + pfxs_pi[i]] + 1)
        )
        df['PVAL_' + pfxs_pval[i] + '_BURDEN_lower'] = nb_pvalue_lower(
            df['OBS_' + pfxs_obs[i]],
            df['ALPHA' + pfxs_at[i]],
            1 / (df['THETA' + pfxs_at[i]] * df['Pi_' + pfxs_pi[i]] + 1)
        )
        df['PVAL_' + pfxs_pval[i] + '_BURDEN_upper'] = nb_pvalue_upper(
            df['OBS_' + pfxs_obs[i]],
            df['ALPHA' + pfxs_at[i]],
            1 / (df['THETA' + pfxs_at[i]] * df['Pi_' + pfxs_pi[i]] + 1)
        )

    def generate_plot_data(mut, bur):
        """
        Given a mutation type and a burden type, generate the data for the volcano plot, Q-Q plot, and table plot
        :param mut: str, mutation type
        :param bur: str, burden type
        """
        col_obs = 'OBS_' + mut if (bur == 'BURDEN') else 'OBS_SAMPLES'
        col_exp = 'EXP_' + mut
        col_pval = 'PVAL_' + (mut + '_' if bur == 'BURDEN' else '') + bur

        # Extract gene name and Ensembl ID
        df['GENE'] = df.ELT.str.split('::', expand=True)[2]
        df['ENSEMBL_ID'] = df.ELT.str.split('::', expand=True)[3]

        # subsetting to only those genes for which the expected nuber of mutations is greater than 0
        ind_keep = df[col_exp] > 0
        df_kept = df.loc[ind_keep].copy()

        df_kept['LOGFC_' + mut + '_' + bur] = np.log2(df_kept[col_obs] / df_kept[col_exp] + 1)
        df_kept['FDR_' + mut + '_' + bur] = fdrcorrection(df_kept[col_pval])[1]
        df_kept['FDR_' + mut + '_' + bur + '_recalc'] = fdrcorrection(df_kept[col_pval + '_recalc'])[1]
        df_kept['FDR_' + mut + '_' + bur + '_lower'] = fdrcorrection(df_kept[col_pval + '_lower'])[1]
        df_kept['FDR_' + mut + '_' + bur + '_upper'] = fdrcorrection(df_kept[col_pval + '_upper'])[1]
        df_kept = df_kept.sort_values(by=col_pval, ignore_index=True)
        df_kept['RANK'] = df_kept.index + 1

        labels = df_kept.GENE.to_numpy()
        logfc = df_kept['LOGFC_' + mut + '_' + bur].to_numpy()
        pvals = df_kept[col_pval + '_recalc'].to_numpy()
        pval_bounds = df_kept[[col_pval + '_lower', col_pval + '_upper']].to_numpy()
        qvals = df_kept['FDR_' + mut + '_' + bur + '_recalc'].to_numpy()
        logq = -np.log10(qvals)
        logq_bounds = -np.log10(
        df_kept[['FDR_' + mut + '_' + bur + '_lower', 'FDR_' + mut + '_' + bur + '_upper']].to_numpy())

        # Determine significant points
        ind_sig = qvals < alp
        ind_lfc = np.abs(logfc) > logfc_thr
        ind_kept = np.logical_and(ind_sig, ind_lfc)

        # Dataframe for the plot
        cols_kept = [
            'RANK', 'GENE', 'ELT_SIZE',
            col_pval,
            'FDR_' + mut + '_' + bur,
            col_obs,
            col_exp,
            'MU', 'SIGMA',
            'FLAG']
        n_rows = max(n_rows_min, int(np.sum(ind_sig) * (1 + n_rows_buffer)))
        df_plot = df_kept.iloc[:n_rows][cols_kept].copy()
        df_plot.rename(columns={
            'ELT_SIZE': 'SIZE',
            col_pval: 'PVAL',
            'FDR_' + mut + '_' + bur: 'FDR',
            col_obs: 'OBS',
            col_exp: 'EXP'
        }, inplace=True)

        for col in ['PVAL', 'FDR']:
            df_plot[col] = reformat_numbers(df_plot[col].to_numpy())
        for col in ['MU', 'SIGMA']:
            df_plot[col] = reformat_numbers(df_plot[col].to_numpy(), format='{:.2f}')
        for col in ['EXP']:
            df_plot[col] = reformat_numbers(df_plot[col].to_numpy(), format='{:.3f}')

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
    df_kept, pvals, pval_bounds, logfc, logq, logq_bounds, labels, ind_kept, table_fig = generate_plot_data('SNV', 'BURDEN')

    # Update the HTML template to include the additional text below the table
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DIG for {name_interval_set}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            .container {{
                display: flex;
            }}
            .plot {{
                margin: 10px;
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
                width: 50%;
            }}
        </style>
    </head>
    <body>
        <h1>DIG Driver Results for {name_interval_set}</h1>

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
                _, pvals, pval_bounds, logfc, logq, logq_bounds, labels, ind_kept, table_fig = generate_plot_data(mut_val, bur_val)

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
                            thickness=thk_err,
                            width=wid_err
                        ),
                        mode='markers',
                        marker=dict(color=col_nonsig, opacity=opac_nonsig),
                        text=labels[~ind_kept].tolist(),
                        name='Non-significant',
                        showlegend=False,
                        xhoverformat='.3f',
                        yhoverformat='.3f'
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
                            thickness=thk_err,
                            width=wid_err
                        ),
                        mode='markers',
                        marker=dict(color=col_sig, opacity=opac_sig),
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
                labels_capped = [(
                                     f"({logfc_capped[i]:.3f}, {logq_capped[i]:.3f} +{logq_upper_capped[i]:.3f} / -{logq_lower_capped[i]:.3f})<br>{l}")
                                 for i, l in enumerate(labels_capped)]

                volcano_fig.add_trace(
                    go.Scatter(
                        x=logfc[ind_capped].tolist(),
                        y=[ymax] * sum(ind_capped),
                        mode='markers',
                        marker=dict(color=col_sig, opacity=opac_sig),
                        text=labels_capped,
                        name='Significant',
                        showlegend=False,
                        hoverinfo='name+text',
                    )
                )

                ylim_upper = min(np.max(logq_upper + logq), ymax) * (1 + hor_buffer)
                volcano_fig.add_trace(
                    go.Scatter(x=[1, 1], y=[0, ylim_upper], mode='lines',
                               line=dict(dash=typ_thin, color=col_thin, width=thk_thin), showlegend=False)
                )
                volcano_fig.add_trace(
                    go.Scatter(x=[0, np.max(logfc) * (1 + hor_buffer)], y=[-np.log10(alp), -np.log10(alp)],
                               mode='lines',
                               line=dict(dash=typ_thick, color=col_thick, width=thk_thick), showlegend=False)
                )
                volcano_fig.add_trace(
                    go.Scatter(x=[0, np.max(logfc) * (1 + hor_buffer)], y=[ymax] * 2,
                               mode='lines',
                               line=dict(dash=typ_thin, color=col_thin, width=thk_thin), showlegend=False)
                )

                volcano_fig.update_layout(
                    title='Observed/Expected counts vs. False Discovery Rate:',
                    xaxis_title='Log2(Observed/Expected + 1)',
                    yaxis_title='-Log10(FDR)',
                    xaxis=dict(range=[0, np.max(logfc) * (1 + hor_buffer)]),
                    yaxis=dict(range=[0, ylim_upper]),
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
                            thickness=thk_err,
                            width=wid_err
                        ),
                        mode='markers',
                        marker=dict(color=col_nonsig, opacity=opac_nonsig),
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
                            thickness=thk_err,
                            width=wid_err
                        ),
                        mode='markers',
                        marker=dict(color=col_sig, opacity=opac_sig),
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
                labels_capped = [(
                                     f"({x_capped[i]:.3f}, {y_capped[i]:.3f} +{y_upper_capped[i]:.3f} / -{y_lower_capped[i]:.3f})<br>{l}")
                                 for i, l in enumerate(labels_capped)]

                qq_fig.add_trace(
                    go.Scatter(
                        x=x_capped,
                        y=[ymax] * sum(ind_capped),
                        mode='markers',
                        marker=dict(color=col_sig, opacity=opac_sig),
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
                        line=dict(dash=typ_thick, color=col_thick, width=thk_thick),
                        showlegend=False
                    )
                )

                qq_fig.add_trace(
                    go.Scatter(x=[0, np.max(x) * (1 + hor_buffer)], y=[ymax] * 2,
                               mode='lines',
                               line=dict(dash=typ_thin, color=col_thin, width=thk_thin), showlegend=False)
                )

                ylim_upper = min(np.max(y_upper + y), ymax) * (1 + hor_buffer)
                qq_fig.update_layout(
                    title='QQ-Plot of P-values:',
                    xaxis_title='Expected -Log10(P-value)',
                    yaxis_title='Observed -Log10(P-value)',
                    xaxis=dict(range=[0, np.max(x) * (1 + hor_buffer)]),
                    yaxis=dict(range=[0, ylim_upper]),
                    template='plotly_white'
                )

                # Save figures as separate data
                plot_data[f"{mut_key}_{bur_key}"] = {
                    'volcano': volcano_fig.to_dict(),
                    'qq': qq_fig.to_dict(),
                    'table': table_fig.to_dict()
                }

    # convert plot data to JSON-like structure
    plot_data_json = json.dumps(plot_data)

    # generate static figures for the default values
    fig_mu = px.histogram(df_kept,
                          x='MU',
                          labels={'MU': 'MU'},
                          opacity=opac_bar,
                          log_y=True,
                          color_discrete_sequence=[col_bar])
    fig_mu.update_layout(
        title='Mean of GP model:',
        xaxis_title='MU (mutations per kilobase)',
        yaxis_title='Number of genes',
        template='plotly_white')

    fig_sigma = px.histogram(df_kept,
                             x='SIGMA',
                             labels={'SIGMA': 'SIGMA'},
                             opacity=opac_bar,
                             log_y=True,
                             color_discrete_sequence=[col_bar])
    fig_sigma.update_layout(
        title='Standard deviation of GP model:',
        xaxis_title='SIGMA (mutations per kilobase)',
        yaxis_title='Number of genes',
        template='plotly_white')

    # save static figures as HTML divs
    fig_mu_html = fig_mu.to_html(full_html=False, include_plotlyjs='cdn')
    fig_sigma_html = fig_sigma.to_html(full_html=False, include_plotlyjs='cdn')

    # combine everything into the final HTML
    html_content = html_content.format(
        name_interval_set=name_interval_set.title(),
        mut_options=mut_options,
        burden_options=burden_options,
        plot_data=plot_data_json,
        fig_mu_html=fig_mu_html,
        fig_sigma_html=fig_sigma_html
    )

    # save to an HTML file
    name = '_'.join(name_interval_set.split(' ')).lower()
    with open(dir_output + f'DIG_report_{name}' + ('' if (prefix_output is None) else '_' + prefix_output) + '.html',
              'w') as f:
        f.write(html_content)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate DIG report for coding genes.")
    parser.add_argument("path_to_dig_results", type=str, help="Path to the DIG results file.")
    parser.add_argument("dir_output", type=str, help="Output directory.")
    parser.add_argument("name_interval_set", type=str, help="Name of interval set.")
    parser.add_argument("--prefix_output", type=str, default=None, help="Prefix for the output file.")
    parser.add_argument("--alp", type=float, default=0.1, help="Significance level (default: 0.1).")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    # Call the function with parsed arguments
    generate_dig_report(args.path_to_dig_results, args.dir_output, args.prefix_output, args.alp)
