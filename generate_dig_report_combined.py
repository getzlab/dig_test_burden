import argparse
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import scipy as sp
from statsmodels.stats.multitest import fdrcorrection
import json

# minimum number of rows (genes) to display in the table
n_rows_min = 50
# horizontal buffer for the scatter plots
hor_buffer = 0.01
# buffer for the number of rows in the table
n_rows_buffer = 0.5
# maximum value along the vertical axis for the volcano and Q-Q plots
ymax = 16
# beta confidence interval
ci = 0.95

# properties of significant points
col_sig = 'rgba(255, 0, 0, 1)'
opac_sig = 0.7
# properties of non-significant points
col_nonsig = 'rgba(0, 0, 0, 1)'
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
opac_err = 0.3
# properties of the bar plot
col_bar = 'gray'
opac_bar = 0.8
# text for the special case when Sample-wise case does not exist
text_special = 'Sample-wise case does not exist for Indels and Indels + SNVs!'

# derived parameters
col_err_sig = ','.join(col_sig.split(',')[:-1]) + ', {})'.format(opac_err)
col_err_nonsig = ','.join(col_nonsig.split(',')[:-1]) + ', {})'.format(opac_err)

# dropdown options
burden_type = {
    'Total': '',
    'Sample-wise': 'SAMPLE',
}
mut_type = {
    'Indels + SNVs': 'MUT',
    'Indels': 'INDEL',
    'SNVs': 'SNV'
}
scatterpoint_type = {
    "Uniform P-mid": "unif",
    "P-mid": "recalc"
}
display_bounds_type = {
    'Yes': True,
    'No': False
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


def nb_pvalue_uniform_midp(k, alpha, p):
    """ Calculate the upper tail p-value for negative binomial distribution using uniform approximation and a random draw.
    """
    return np.random.uniform(size=k.shape) * sp.stats.nbinom.pmf(k, alpha, p) + sp.special.betainc(k+1, alpha, 1-p)


def reformat_numbers(df, cols, form='{:.3E}'):
    """
    Reformat numbers in an array to a specific format
    """
    return df[cols].applymap(lambda x: form.format(x) if not pd.isna(x) else 'NA')


def generate_dig_report(
        path_to_coding_results,
        path_to_promoter_results,
        path_to_3utr_results,
        path_to_5utr_results,
        dir_output,
        cgc_list_path,
        pancan_list_path,
        prefix_output=None,
        alp=0.1
):
    # Driver gene lists
    cgc_list = pd.read_csv(cgc_list_path, sep='\t').to_numpy().flatten()
    pancan_list = pd.read_csv(pancan_list_path, sep='\t').to_numpy().flatten()
    # DIG results

    dig_outputs = {
        'coding': path_to_coding_results,
        'promoter': path_to_promoter_results,
        '5utr': path_to_5utr_results,
        '3utr': path_to_3utr_results
    }
    result_types = list(dig_outputs.keys())

    for result_type in result_types:

        dfi = pd.read_csv(dig_outputs[result_type], sep='\t')

        if result_type == 'coding':
            dfi = dfi.set_index('GENE')
            df_comb = pd.DataFrame(index=dfi.index.copy())
            col_obs = ['OBS_NONSYN', 'N_SAMP_NONSYN', 'OBS_INDEL']
            col_pi = ['Pi_NONSYN', 'Pi_NONSYN', 'Pi_INDEL']
            col_size = 'GENE_LENGTH'
        else:
            dfi['GENE'] = dfi.ELT.str.split('::', expand=True)[2]
            dfi = dfi.sort_values(['GENE', 'OBS_SNV'])
            dfi = dfi.set_index('GENE')
            dfi = dfi.loc[~dfi.index.duplicated()]
            col_obs = ['OBS_SNV', 'OBS_SAMPLES', 'OBS_INDEL']
            col_pi = ['Pi_SUM', 'Pi_SUM', 'Pi_INDEL']
            col_size = 'ELT_SIZE'

        genes_in_coding = dfi.index[dfi.index.isin(df_comb.index)]
        dfi_comp = dfi.loc[genes_in_coding]
        df_comb.loc[genes_in_coding, 'SIZE_' + result_type] = dfi_comp[col_size].to_numpy().copy()

        for j, mt in enumerate(['SNV', 'SNV_SAMPLE', 'INDEL']):

            if mt == 'INDEL':
                pfx_at = '_INDEL'
            else:
                pfx_at = ''
            df_comb.loc[genes_in_coding, 'PVAL_' + result_type + '_' + mt + '_recalc'] = nb_pvalue_greater_midp(
                dfi_comp[col_obs[j]],
                dfi_comp['ALPHA' + pfx_at],
                1 / (dfi_comp['THETA' + pfx_at] * dfi_comp[col_pi[j]] + 1)
            )
            df_comb.loc[genes_in_coding, 'PVAL_' + result_type + '_' + mt + '_unif'] = nb_pvalue_uniform_midp(
                dfi_comp[col_obs[j]],
                dfi_comp['ALPHA' + pfx_at],
                1 / (dfi_comp['THETA' + pfx_at] * dfi_comp[col_pi[j]] + 1)
            )
            df_comb.loc[genes_in_coding, 'PVAL_' + result_type + '_' + mt + '_lower'] = nb_pvalue_lower(
                dfi_comp[col_obs[j]],
                dfi_comp['ALPHA' + pfx_at],
                1 / (dfi_comp['THETA' + pfx_at] * dfi_comp[col_pi[j]] + 1)
            )
            df_comb.loc[genes_in_coding, 'PVAL_' + result_type + '_' + mt + '_upper'] = nb_pvalue_upper(
                dfi_comp[col_obs[j]],
                dfi_comp['ALPHA' + pfx_at],
                1 / (dfi_comp['THETA' + pfx_at] * dfi_comp[col_pi[j]] + 1)
            )

        # combining p-values with Fisher's method
        col_i = 'PVAL_' + result_type + '_' + 'MUT'
        df_comb[col_i + '_recalc'] = np.nan
        df_comb[col_i + '_unif'] = np.nan
        df_comb[col_i + '_lower'] = np.nan
        df_comb[col_i + '_upper'] = np.nan
        for idx in df_comb.index:
            df_comb.at[idx, col_i + '_recalc'] = sp.stats.combine_pvalues(
                [df_comb.at[idx, 'PVAL_' + result_type + '_SNV_recalc'],
                 df_comb.at[idx, 'PVAL_' + result_type + '_INDEL_recalc']], method='fisher')[1]
            df_comb.at[idx, col_i + '_unif'] = sp.stats.combine_pvalues(
                [df_comb.at[idx, 'PVAL_' + result_type + '_SNV_unif'],
                 df_comb.at[idx, 'PVAL_' + result_type + '_INDEL_unif']], method='fisher')[1]
            df_comb.at[idx, col_i + '_lower'] = sp.stats.combine_pvalues(
                [df_comb.at[idx, 'PVAL_' + result_type + '_SNV_lower'],
                 df_comb.at[idx, 'PVAL_' + result_type + '_INDEL_lower']], method='fisher')[1]
            df_comb.at[idx, col_i + '_upper'] = sp.stats.combine_pvalues(
                [df_comb.at[idx, 'PVAL_' + result_type + '_SNV_upper'],
                 df_comb.at[idx, 'PVAL_' + result_type + '_INDEL_upper']], method='fisher')[1]

    # combining p-values across region types
    for mt in ['SNV', 'SNV_SAMPLE', 'INDEL', 'MUT']:
        for typ in ['recalc', 'unif', 'lower', 'upper']:
            for idx in df_comb.index:
                df_comb.at[idx, 'PVAL_' + mt + '_' + typ] = sp.stats.combine_pvalues(
                    [df_comb.at[idx, 'PVAL_' + rt + '_' + mt + '_' + typ] for rt in result_types if
                     ~np.isnan(df_comb.at[idx, 'PVAL_' + rt + '_' + mt + '_' + typ])],
                    method='fisher')[1]
            isna = df_comb['PVAL_' + mt + '_' + typ].isna()
            df_comb.loc[~isna, 'FDR_' + mt + '_' + typ] = fdrcorrection(df_comb.loc[~isna, 'PVAL_' + mt + '_' + typ])[1]

    # adding CGC and PanCanAtlas information
    df_comb['CGC'] = df_comb.index.isin(cgc_list).copy()
    df_comb['PANCAN'] = df_comb.index.isin(pancan_list).copy()
    df_comb = df_comb.reset_index().copy()

    # save combined p-values into a text file
    df_comb.to_csv(('' if (prefix_output is None) else prefix_output + '.') + 'combined.dig.results.txt', sep='\t', index=False)

    def generate_plot_data(mut, bur, display_bounds, scatterpoint):
        """
        Given a mutation type and a burden type, generate the data for the volcano plot, Q-Q plot, and table plot
        :param mut: str, mutation type
        :param bur: str, burden type
        :param display_bounds: bool, whether to display the bounds of the p-values
        :param scatterpoint: str, type of p-values to use
        """
        col_chosen = mut + ('_SAMPLE' if bur=='SAMPLE' else '') + '_' + scatterpoint
        col_pvals = ['PVAL_' + rt + '_' + col_chosen for rt in result_types]
        col_sizes = ['SIZE_' + rt  for rt in result_types]
        cols_kept = ['GENE'] + col_pvals + col_sizes + ['PVAL_' + col_chosen] + ['FDR_' + col_chosen] + ['CGC', 'PANCAN']
        if display_bounds:
            cols_bound = ['PVAL_' + '_'.join(col_chosen.split('_')[:-1]) + '_' + typ for typ in ['lower', 'upper']]
            df_kept = df_comb[cols_kept + cols_bound].sort_values(by='PVAL_' + col_chosen, ignore_index=True)
            pval_bounds = df_kept[cols_bound].to_numpy()
        else:
            df_kept = df_comb[cols_kept].sort_values(by='PVAL_' + col_chosen, ignore_index=True)
            pval_bounds = None
        df_kept['RANK'] = df_kept.index + 1
        pvals = df_kept['PVAL_' + col_chosen].to_numpy()
        labels = df_kept['GENE'].to_numpy()

        # Determine significant points
        ind_sig = df_kept['FDR_' + col_chosen] < alp

        # Dataframe for the plot
        cols_plot = ['RANK', 'GENE', 'FDR_' + col_chosen, 'PVAL_' + col_chosen] + col_pvals +  col_sizes + ['CGC', 'PANCAN']
        n_rows = max(n_rows_min, int(np.sum(ind_sig) * (1 + n_rows_buffer)))
        df_plot = df_kept.iloc[:n_rows][cols_plot].copy()
        df_plot.rename(columns={
            'PVAL_' + col_chosen: 'PVAL',
            'FDR_' + col_chosen : 'FDR'
        }, inplace=True)
        df_plot.rename(columns={'PVAL_' + rt + '_' + col_chosen: 'PVAL_' + rt for rt in result_types}, inplace=True)

        df_plot[col_sizes] = reformat_numbers(df_plot, col_sizes, form='{:.0f}')
        cols_floats = ['PVAL', 'FDR'] + ['PVAL_' + rt for rt in result_types]
        df_plot[cols_floats] = reformat_numbers(df_plot, cols_floats)

        # Ensure no NaN values in the table
        # df_plot.fillna('NA', inplace=True)
        df_plot[['RANK', 'GENE', 'CGC', 'PANCAN']] = df_plot[['RANK', 'GENE', 'CGC', 'PANCAN']].astype(str)
        # Generate table figure
        headerColor = 'grey'
        rowEvenColor = 'lightgrey'
        rowOddColor = 'white'

        df_plot = df_plot.astype(str)
        for i in range(df_plot.shape[0]):
            if ind_sig[i]:
                df_plot.loc[i, :] = '<b>' + df_plot.loc[i, :] + '</b>'

        # Adding hyperlinks to a Google search for the gene names
        gene_entries = []
        for g in df_plot['GENE']:
            if '<b>' in g:
                g_trimmed = g.split('>')[1].split('<')[0]
            else:
                g_trimmed = g
            gene_entries.append(
                f'<a href="https://www.google.com/search?q={g_trimmed}+gene+cancer" target="_blank">{g}</a>')
        # df_plot['GENE'] = gene_entries
        # df_plot = df_plot[['RANK', 'GENE']]

        table_fig = go.Figure(data=[go.Table(
            header=dict(values=['<b>' + col.replace('_', '<br>') + '</b>' for col in df_plot.columns],
                        line_color='darkslategray',
                        fill_color=headerColor,
                        align=['left'] + ['center'] * (len(df_plot.columns)-1),
                        font=dict(color='white', size=12)
                        ),
            cells=dict(values=[df_plot[col].tolist() for col in df_plot.columns],
                       line_color='darkslategray',
                       fill_color=[[rowOddColor if i % 2 == 0 else rowEvenColor for i in range(df_plot.shape[0])]],
                       align=['left'] + ['center'] * (len(df_plot.columns)-1),
                       font=dict(color='darkslategray', size=11),
                       # format = ['html'] * len(df_plot.columns)  # Enable HTML formatting
                       )
        )])

        return df_kept, pvals, pval_bounds, labels, ind_sig, table_fig

    # Update the HTML template to include the additional text below the table
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DIG combined</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            .container {{
                display: flex;
            }}
            .plot {{
                margin: 10px;
            }}
            #qq-plot {{
                width: 100%;
            }}
            #table-plot {{
                width: 100%;
            }}
            .figures-container {{
                display: flex;
                justify-content: space-between;
            }}
            .switch {{
            position: relative;
            display: inline-block;
            width: 30px;
            height: 17px;
            }}
            .switch input {{
                opacity: 0;
                width: 0;
                height: 0;
            }}
            .slider {{
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #ccc;
                transition: .4s;
            }}
            .slider:before {{
                position: absolute;
                content: "";
                height: 13px;
                width: 13px;
                left: 2px;
                bottom: 2px;
                background-color: white;
                transition: .4s;
            }}
            input:checked + .slider {{
                background-color: #000000;
            }}
            input:checked + .slider:before {{
                transform: translateX(13px);
            }}
            .slider.round {{
                border-radius: 17px;
            }}
            .slider.round:before {{
                border-radius: 50%;
            }}
            .red-text {{
                color: red;
            }}
            .black-text {{
                color: black;
            }}
        </style>
    </head>
    <body>
        <h1>Combined Mutation Burden Test</h1>

        <label for="mut-type">Select Mutation Type:</label>
        <select id="mut-type" onchange="updatePlot()">
            {mut_options}
        </select>

        <label for="burden-type">Select Burden Type:</label>
        <select id="burden-type" onchange="updatePlot()">
            {burden_options}
        </select>
        
        <label for="scatterpoint-type">P-value Type:</label>
        <select id="scatterpoint-type" onchange="updatePlot()">
            {scatterpoint_options}
        </select>
        
        <label for="display-bounds-type">Display bounds:</label>
        <label class="switch">
            <input type="checkbox" id="display-bounds-type" onchange="updatePlot()">
            <span class="slider round"></span>
        </label>

        <h2 id="plot-title" class="text-color"></h2>

        <div class="container">
            <div id="qq-plot" class="plot"></div>
        </div>
        <div id="table-plot" class="plot"></div>

        <script>
            var plotData = {plot_data};

            function updatePlot() {{
                var mutTypeKey = document.getElementById("mut-type").value;
                var burdenTypeKey = document.getElementById("burden-type").value;
                var scatterpointTypeKey = document.getElementById("scatterpoint-type").value;
                var displayBoundsKey = document.getElementById("display-bounds-type").checked ? 'Yes' : 'No';

                var data = plotData[mutTypeKey + '_' + burdenTypeKey + '_' + displayBoundsKey + '_' + scatterpointTypeKey];

                // Update Q-Q Plot
                var qqData = data.qq;
                Plotly.react('qq-plot', qqData);

                // Update Table Plot
                var tableData = data.table;
                Plotly.react('table-plot', tableData);

                /// Update Header
                var headerText = data.text;
                document.getElementById("plot-title").textContent = headerText;
                var headerColor = data.textcolor;
                document.getElementById("plot-title").className = headerColor;
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
    display_bounds_options = "\n".join([f'<option value="{key}">{key}</option>' for key in display_bounds_type.keys()])
    scatterpoint_options = "\n".join([f'<option value="{key}">{key}</option>' for key in scatterpoint_type.keys()])

    # prepare plot data for all combinations of mut_type and burden_type dropdown options
    plot_data = {}
    for mut_key, mut_val in mut_type.items():
        for bur_key, bur_val in burden_type.items():
            for display_bounds_key, display_bounds_val in display_bounds_type.items():
                for scatterpoint_key, scatterpoint_val in scatterpoint_type.items():
                    if not (mut_key in ['Indels', 'Indels + SNVs'] and bur_key == 'Sample-wise'):
                        _, pvals, pval_bounds, labels, ind_sig, table_fig = generate_plot_data(mut_val, bur_val, display_bounds_val, scatterpoint_val)

                        # Q-Q Plot

                        # Scatter plots
                        x = -np.log10(np.arange(1, len(pvals) + 1) / (len(pvals) + 1))
                        y = -np.log10(pvals)
                        ind_capped = y > ymax
                        ind_ncapped = np.logical_and(ind_sig, ~ind_capped)
                        labels_capped = labels[ind_capped].tolist()
                        x_capped = x[ind_capped].tolist()
                        y_capped = y[ind_capped].tolist()
                        if display_bounds_val:
                            y_upper = -np.log10(pval_bounds[:, 0]) - y
                            y_lower = y + np.log10(pval_bounds[:, 1])
                            dict_erry_sig = dict(
                                type='data',
                                symmetric=False,
                                array=np.round(y_upper[ind_ncapped], 3).tolist(),
                                arrayminus=np.round(y_lower[ind_ncapped], 3).tolist(),
                                thickness=thk_err,
                                width=wid_err,
                                color=col_err_sig
                            )
                            dict_erry_nonsig = dict(
                                type='data',
                                symmetric=False,
                                array=np.round(y_upper[~ind_sig], 3).tolist(),
                                arrayminus=np.round(y_lower[~ind_sig], 3).tolist(),
                                thickness=thk_err,
                                width=wid_err,
                                color=col_err_nonsig
                            )
                            y_upper_capped = y_upper[ind_capped].tolist()
                            y_lower_capped = y_lower[ind_capped].tolist()
                            labels_capped = [(
                                f"({x_capped[i]:.3f}, {y_capped[i]:.3f} +{y_upper_capped[i]:.3f} / -{y_lower_capped[i]:.3f})<br>{l}")
                                for i, l in enumerate(labels_capped)]
                            ylim_upper = min(np.max(y_upper + y), ymax) * (1 + hor_buffer)
                        else:
                            dict_erry_sig, dict_erry_nonsig = None, None
                            labels_capped = [(
                                f"({x_capped[i]:.3f}, {y_capped[i]:.3f})<br>{l}")
                                for i, l in enumerate(labels_capped)]
                            ylim_upper = min(np.max(y), ymax) * (1 + hor_buffer)

                        xi = np.arange(1, len(pvals) + 1)
                        clower = -np.log10(sp.stats.beta.ppf((1 - ci) / 2, xi, xi[::-1]))
                        cupper = -np.log10(sp.stats.beta.ppf((1 + ci) / 2, xi, xi[::-1]))

                        qq_fig = go.Figure()
                        qq_fig.add_trace(
                            go.Scatter(
                                x=x[~ind_sig].tolist(),
                                y=y[~ind_sig].tolist(),
                                error_y=dict_erry_nonsig,
                                mode='markers',
                                marker=dict(color=col_nonsig, opacity=opac_nonsig),
                                text=labels[~ind_sig].tolist(),
                                name='Non-significant',
                                showlegend=False,
                                xhoverformat='.3f',
                                yhoverformat='.3f'
                            )
                        )

                        qq_fig.add_trace(
                            go.Scatter(
                                x=x[ind_ncapped].tolist(),
                                y=y[ind_ncapped].tolist(),
                                error_y=dict_erry_sig,
                                mode='markers',
                                marker=dict(color=col_sig, opacity=opac_sig),
                                text=labels[ind_ncapped].tolist(),
                                name='Significant',
                                showlegend=False,
                                xhoverformat='.3f',
                                yhoverformat='.3f'
                            )
                        )

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

                        # Line plots
                        qq_fig.add_trace(
                            go.Scatter(
                                x=[0, np.max(x)],
                                y=[0, np.max(x)],
                                mode='lines',
                                line=dict(dash=typ_thick, color=col_thick, width=thk_thick),
                                showlegend=False,
                                hoverinfo='skip'
                            )
                        )
                        if ymax < ylim_upper:
                            qq_fig.add_trace(
                                go.Scatter(
                                    x=[0, np.max(x) * (1 + hor_buffer)],
                                    y=[ymax] * 2,
                                    mode='lines',
                                    line=dict(dash=typ_thin, color=col_thin, width=thk_thin),
                                    showlegend=False,
                                    hoverinfo='skip'
                                )
                            )

                        # Confidence intervals for identity line
                        qq_fig.add_trace(
                            go.Scatter(
                                x=x.tolist(),
                                y=clower.tolist(),
                                mode='lines',
                                line=dict(dash=typ_thin, color=col_thin, width=thk_thin),
                                showlegend=False,
                                hoverinfo='skip'
                            )
                        )
                        qq_fig.add_trace(
                            go.Scatter(
                                x=x.tolist(),
                                y=cupper.tolist(),
                                mode='lines',
                                line=dict(dash=typ_thin, color=col_thin, width=thk_thin),
                                showlegend=False,
                                hoverinfo='skip'
                            )
                        )

                        # Formatting the figure
                        qq_fig.update_layout(
                            title='QQ-Plot of P-values:',
                            xaxis_title='Expected -Log10(P-value)',
                            yaxis_title='Observed -Log10(P-value)',
                            xaxis=dict(range=[0, np.max(x) * (1 + hor_buffer)]),
                            yaxis=dict(range=[0, ylim_upper]),
                            template='plotly_white'
                        )

                        # Save figures as separate data
                        plot_data[f"{mut_key}_{bur_key}_{display_bounds_key}_{scatterpoint_key}"] = {
                            'qq': qq_fig.to_dict(),
                            'table': table_fig.to_dict(),
                            'text': bur_key + ' Mutation Burden of ' + mut_key,
                            'textcolor': 'black-text'
                        }
                    else:
                        plot_data[f"{mut_key}_{bur_key}_{display_bounds_key}_{scatterpoint_key}"] = {
                            'qq': None,
                            'dnds': None,
                            'table': None,
                            'text': text_special,
                            'textcolor': 'red-text'
                        }

    # convert plot data to JSON-like structure
    plot_data_json = json.dumps(plot_data)

    # combine everything into the final HTML
    html_content = html_content.format(
        mut_options=mut_options,
        burden_options=burden_options,
        display_bounds_options=display_bounds_options,
        scatterpoint_options=scatterpoint_options,
        plot_data=plot_data_json
    )

    # save to an HTML file
    with open(dir_output + '/' + ('' if (prefix_output is None) else prefix_output + '_') + f'dig_report_combined.html',
              'w') as f:
        f.write(html_content)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate combined DIG report.")
    parser.add_argument("path_to_coding_results", type=str, help="Path to the DIG results for coding regions.")
    parser.add_argument("path_to_promoter_results", type=str, help="Path to the DIG results for promoter regions.")
    parser.add_argument("path_to_3utr_results", type=str, help="Path to the DIG results for 3' UTR regions.")
    parser.add_argument("path_to_5utr_results", type=str, help="Path to the DIG results for 5' UTR regions.")
    parser.add_argument("dir_output", type=str, help="Output directory.")
    parser.add_argument("cgc_list", type=str, help="Path to the list of CGC genes.")
    parser.add_argument("pancan_list", type=str, help="Path to the list of PanCanAtlas genes.")
    parser.add_argument("--prefix_output", type=str, default=None, help="Prefix for the output file.")
    parser.add_argument("--alp", type=float, default=0.1, help="Significance level (default: 0.1).")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    # Call the function with parsed arguments
    generate_dig_report(
        args.path_to_coding_results,
        args.path_to_promoter_results,
        args.path_to_3utr_results,
        args.path_to_5utr_results,
        args.dir_output,
        args.cgc_list,
        args.pancan_list,
        args.prefix_output,
        args.alp
    )
