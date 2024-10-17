import generate_dig_report_main
import generate_dig_report_coding
import generate_dig_report_noncoding
import generate_dig_report_combined

cgc_list_path = './cancer_gene_census_2024_06_20.tsv'
pancan_list_path = './pancanatlas_genes.tsv'

# noncoding regions
generate_dig_report_noncoding.generate_dig_report(
    'promoters.results.txt',
    './',
    cgc_list_path,
    pancan_list_path,
    'promoter_regions',
    prefix_output='uvm'
)
generate_dig_report_noncoding.generate_dig_report(
    '5utr.results.txt',
    './',
    cgc_list_path,
    pancan_list_path,
    '5-prime_UTRs',
    prefix_output='uvm'
)
generate_dig_report_noncoding.generate_dig_report(
    '3utr.results.txt',
    './',
    cgc_list_path,
    pancan_list_path,
    '3-prime_UTRs',
    prefix_output='uvm'
)

# coding regions
generate_dig_report_coding.generate_dig_report(
    'genes.results.txt',
    './',
    cgc_list_path,
    pancan_list_path,
    prefix_output='uvm'
)

# combined test for all regions
generate_dig_report_combined.generate_dig_report(
    'genes.results.txt',
    'promoters.results.txt',
    '3utr.results.txt',
    '5utr.results.txt',
    './',
    cgc_list_path,
    pancan_list_path,
    prefix_output='uvm'
)

# main report
generate_dig_report_main.generate_main_report(
    'uvm_dig_report_coding_regions.html',
    'uvm_dig_report_promoter_regions.html',
    'uvm_dig_report_3-prime_utrs.html',
    'uvm_dig_report_5-prime_utrs.html',
    'uvm_dig_report_combined.html',
    './',
    prefix_output='uvm'
)
