import wolf 
from wolf.localization import LocalizeToDisk, DeleteDisk
from tasks import *
import re
import os

def dig_workflow(
    zipped_tracks = "gs://getzlab-workflows-reference_files-oa/hg38/dig/roadmap_tracks_735_10kb.h5",
    genome_counts="gs://getzlab-workflows-reference_files-oa/hg38/dig/genome_counts.h5",
    gene_data="https://cb.csail.mit.edu/DIG/downloads/dig_data_files/gene_data.h5",
    element_data="gs://getzlab-workflows-reference_files-oa/hg38/dig/element_data.h5",
    ref_fasta="gs://getzlab-workflows-reference_files-oa/hg38/dig/hg19.fasta",
    ref_fasta_idx = "gs://getzlab-workflows-reference_files-oa/hg38/dig/hg19.fasta.fai",
    liftover_chain_file="gs://getzlab-workflows-reference_files-oa/hg38/dig/hg38ToHg19.over.chain.gz",
    cgc_list="gs://getzlab-workflows-reference_files-oa/hg38/dig/cancer_gene_census_2024_06_20.tsv",
    pancan_list="gs://getzlab-workflows-reference_files-oa/hg38/dig/pancanatlas_genes.tsv",
    maf_file = None,
    ref_build = None,
    interval_set_bed=None,
    interval_set_name=None,
    mutation_map=None,
    extra_kfold_args={},            
):
    # The workflow will train a mutation map for each MAF file passed
    
    # Identify cohort name based on MAF file name
    try:
        # TODO make this to accept lists for maf_file:
        cohort_name = re.search(r"(.*?)\.(?:txt|bed|tsv|maf)$", os.path.basename(maf_file)).groups()[0].replace("_", "-").lower()
    except:
        raise ValueError("MAF file expected to be in DIG format with ext [.txt|.bed|.tsv|.maf]!")

    if ref_build not in ['hg19', 'hg38']:
        raise ValueError("ref build must be specified as hg19 or hg38")
    
    # Localize hg19 fasta used by DIG
    fasta_localization = LocalizeToDisk(
        name = "Localize_ref_fasta",
        files = {
            "ref_fasta": ref_fasta, 
            "ref_fasta_idx": ref_fasta_idx
        }
    )

    # Liftover to hg19 and conversion to DIG-compatible format
    maf_hg19 = DIG_convert_maf(
        inputs = {
            "input_maf" : maf_file,
            "ref_build" : ref_build,
            "liftover_chainfile" : liftover_chain_file
        }
    )

    # DIG-compatible annotation of mutations
    annot_task = DIG_annotate_maf(
        inputs={
            "input_maf": maf_hg19["dig_maf"], 
            "ref_fasta": fasta_localization["ref_fasta"], 
            "ref_fasta_idx": fasta_localization["ref_fasta_idx"],  
            "cohort_name": cohort_name
        }
    )

    if mutation_map is None:

        # Localize zipped reference tracks
        zipped_localization = LocalizeToDisk(
            name = "Localize_tracks",
            files = {
                "zipped_tracks": zipped_tracks
            }
        )
        
        # Unzip the reference tracks and place them on a scratch disk
        unzip_task = DIG_unzip_h5(
            inputs={
                "zipped_tracks" : zipped_localization["zipped_tracks"]
            }, 
            extra_localization_args={
                "scratch_disk_name": "tracks"
            }
        )
    
        # Add mutations to h5 file
        add_obj_task = DIG_add_objectives(
            inputs={
                "cohort_name": cohort_name,
                "unzipped_tracks": unzip_task["tracks"],
                "maf_file":annot_task["dig_maf"]
            }, 
            extra_localization_args={
                "scratch_disk_name": "tracks-with-objectives-{}".format(cohort_name),
                "use_scratch_disk":True
            }, 
            checkpoint=True,
            dependencies=[unzip_task, annot_task]
        )
    
        # # Delete scratch disk with unzipped tracks
        # delete_tracks = DeleteDisk(
        #     inputs={
        #         "disk": unzip_task["tracks"], 
        #         "upstream": [add_obj_task["muts_added"]]
        #     }
        # )
    
        # Run k-fold cross validation
        run_kfold_task = DIG_run_kfold(
            inputs={
                "cohort_name": cohort_name,
                "unzipped_tracks": add_obj_task["muts_added"],
                **extra_kfold_args
            }, 
            use_gpu=True,
            checkpoint=True, 
            dependencies=[add_obj_task]
        )

        # Pretrain model
        pretrain_region_task = DIG_pretrain_region(
            inputs={
                "unzipped_tracks": add_obj_task["muts_added"],
                "kfold_output_dir":run_kfold_task["kfold_results"],
                "cohort_name": cohort_name,
                "maf_file": annot_task["dig_maf"]
            }
        )
        pretrain_sequence_task = DIG_pretrain_sequence(
            inputs={
                "genome_counts": genome_counts,
                "cohort_name": cohort_name,
                "pretrained_model": pretrain_region_task["pretrained_model"], 
                "maf_file": annot_task["dig_maf"]
            }
        )
        pretrain_genic_task = DIG_pretrain_genic(
            inputs={
                "pretrained_model": pretrain_sequence_task["pretrained_model"],
                "gene_data": gene_data
            }
        )
        mutation_map = pretrain_genic_task["pretrained_model"]

    # Build background model from interval sets and mutation map
    preproc_element = DIG_preprocess_element_model(
        inputs = {
            "input_bed" : interval_set_bed,
            "annot_name" : interval_set_name,
            "input_element_data" : element_data,
            "input_mut_map" : mutation_map,
            "ref_fasta" : fasta_localization["ref_fasta"]
        }
    )
    element_model = DIG_element_model(
        inputs = {
            "input_element_data" : preproc_element["output_element_data"],
            "input_mut_map" : preproc_element["output_mut_map"],
            "annot_name" : interval_set_name
        }
    )
    
    # Run statistical test and report generation for the coding region
    results_coding = DIG_test_coding(
        inputs = {
            "input_annot_maf" : annot_task["dig_maf"],
            "input_mut_map" : mutation_map,
            "cohort": cohort_name
        }
    )
    report_coding = DIG_report_coding(
        inputs = {
            "input_results": results_coding["dig_results"],
            "cgc_list": cgc_list,
            "pancan_list": pancan_list,
            "cohort": cohort_name
        }
    )

    # Run statistical test and report generation for the noncoding region
    results_noncoding = DIG_test_noncoding(
        inputs = {
            "input_annot_maf" : annot_task["dig_maf"],
            "input_mut_map" : element_model["output_mut_map"],
            "input_bed" : interval_set_bed,
            "annot_name" : interval_set_name,
            "cohort": cohort_name
        }    
    )
    report_noncoding = DIG_report_noncoding(
        inputs = {
            "input_results" : results_noncoding["dig_results"],
            "cgc_list": cgc_list,
            "pancan_list": pancan_list,
            "annot_name" : interval_set_name,
            "cohort" : cohort_name
        }
    )

    # Gather task for collecting noncoding results and reports
    gather_noncoding_reports = DIG_gather_noncoding(
        inputs = dict(
            gather_parameter = [report_noncoding['dig_report']]
        )
    )
    gather_noncoding_results = DIG_gather_noncoding(
        inputs = dict(
            gather_parameter = [results_noncoding['dig_results']]
        )
    )

    # Generate combined p-values and final report then zip all results and reports into a single file
    results = DIG_results(
        inputs = {
            "noncoding_htmls": gather_noncoding_reports['output'],
            "noncoding_results": gather_noncoding_results['output'],
            "coding_html": report_coding['dig_report'],
            "coding_result": results_coding['dig_results'],
            "cgc_list": cgc_list,
            "pancan_list": pancan_list,
            "cohort" : cohort_name
        }
    )
