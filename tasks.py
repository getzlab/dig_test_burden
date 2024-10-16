import wolf


class DIG_convert_maf(wolf.Task):
    name = "DIG_convert_maf"

    inputs = {
            "input_maf" : None,
            "ref_build" : None, # ref build of input maf
            "liftover_chainfile": None,
            }
    
    script = """
    python3 /build/convert_maf.py --input_maf ${input_maf} --input_build ${ref_build} --output_path $(basename ${input_maf:0:-4}).hg19.dig.maf --liftover_chainfile ${liftover_chainfile}
    """

    output_patterns = {
            "dig_maf": "*.hg19.dig.maf"
            }

    resources = { "cpus-per-task": 2, "mem" : "20G" }
    docker = "gcr.io/broad-getzlab-workflows/dig_docker:latest"


class DIG_annotate_maf(wolf.Task):
    name = "DIG_annotate_maf"

    inputs = {
        "input_maf": None,
        "ref_fasta": None,
        "ref_fasta_idx": None,
        "cohort_name": None
    }

    script = """
    DigPreprocess.py annotMutationFile ${input_maf} ${ref_fasta} ${cohort_name}.txt
    """

    output_patterns = {
        "dig_maf": "*.txt"
    }
    
    docker = "gcr.io/broad-getzlab-workflows/dig_docker:latest"
    
    resources = { 
        "cpus-per-task": 4, 
        "mem" : "20G" 
    }


class DIG_unzip_h5(wolf.Task):
    name = "DIG_unzip_h5"
    
    inputs = {
      "zipped_tracks" : None
    }
    
    script = """
      ln -s ${zipped_tracks} ./$(basename ${zipped_tracks})
      DataExtractor.py unzipH5 $(basename ${zipped_tracks})
      """
    
    output_patterns = {
      "tracks" : "*.unzipped.h5"
    }
    
    docker = "gcr.io/broad-getzlab-workflows/dig_docker:latest"
    
    use_scratch_disk=True
    scratch_disk_size=200
    scratch_disk_name="tracks"
    
    preemptible=False
    
    resources = { 
        "cpus-per-task": 4, 
        "mem" : "20G" 
    }


class DIG_add_objectives(wolf.Task):
    name = "DIG_add_objectives"
    
    inputs = {
        "cohort_name": None,
        "unzipped_tracks": None,
        "maf_file": None # must be in Dig format
    }
    
    script = """
    cp ${unzipped_tracks} ./${cohort_name}_tracks_with_objectives.h5
    DataExtractor.py addObjectives ./${cohort_name}_tracks_with_objectives.h5 ${maf_file}
    """
     
    output_patterns = {
            "muts_added": "*tracks_with_objectives.h5"
            }

    docker = "gcr.io/broad-getzlab-workflows/dig_docker:latest"
    
    use_scratch_disk=True
    scratch_disk_size=200
    scratch_disk_name="tracks-with-objectives"
    
    preemptible=False
    # checkpoint=True # need this to keep from purging the scratch disk

    resources = { 
        "cpus-per-task": 2, 
        "mem" : "20G" 
    }


class DIG_run_kfold(wolf.Task):
    name = "DIG_run_kfold_training"

    inputs = {
        "cohort_name":None,
        "unzipped_tracks": None,
        "autoregressive_size": 10,
        "gp_reruns": 3,
        "gp_runs" : 5,
        "count_quartile_thresh": 0.999,
        "mappability_thresh": 0.5,
        "epochs": 10,
        "num_dataset_workers":16
    }

    script = """
    sudo mkdir -p /ramdisk
    sudo mount -t tmpfs -o rw,size=195G tmpfs /ramdisk
    echo "copying tracks to memory..."
    cp ${unzipped_tracks} /ramdisk/

    mkdir -p ${cohort_name}_kfold_res
    python3 /build/mutation_density/DIGDriver/region_model/kfold_mutations_main.py -o ${cohort_name}_kfold_res -c ${cohort_name} -d /ramdisk/${cohort_name}_tracks_with_objectives.h5 -as ${autoregressive_size} -gr ${gp_reruns} -gp ${gp_runs} -cq ${count_quartile_thresh} -sm -st -m ${mappability_thresh} -e 10 -u -g all -nw ${num_dataset_workers}
    """
    
    use_gpu=True
    
    output_patterns = {
        "kfold_results": "*_kfold_res/"
    }
    
    docker = "gcr.io/broad-getzlab-workflows/dig_docker_gpu:latest"
    
    resources = { 
        "cpus-per-task": 32, 
        "mem" : "190G" 
    }


class DIG_pretrain_region(wolf.Task):
    name = "DIG_pretrain_region"

    inputs = {
        "unzipped_tracks": None,
        "kfold_output_dir":None,
        "cohort_name": None,
        "maf_file": None
    }

    # overrides = {"kfold_output_dir" : "string"}

    script = """
    # Pre-train regional rate parameters from the completed CNN+GP kfold run
    echo "Pre-training regional rate parameters..."
    DigPretrain.py regionModel "$(ls -td ${kfold_output_dir}/kfold/${cohort_name}/*/ | head -1)" ${unzipped_tracks} ./${cohort_name}.h5 --cohort-name ${cohort_name} --mutation-file ${maf_file}
    cp ./${cohort_name}.h5 ${cohort_name}_map.h5
    """
    
    output_patterns = {
            "pretrained_model": "*_map.h5"
            }
    
    docker = "gcr.io/broad-getzlab-workflows/dig_docker:latest"
    
    resources = {
        "cpus-per-task": 4,
        "mem" : "12G" 
    }


class DIG_pretrain_sequence(wolf.Task):
    name = "DIG_pretrain_sequence"

    inputs = {
        "genome_counts": None,
        "cohort_name": None,
        "pretrained_model":None,
        "maf_file": None
    }

    script = """
    cp ${pretrained_model} ./
    # Pre-train the sequence context parameters using pre-computed genome counts and annotated mutations
    echo "Pre-training the sequence context parameters..."
    DigPretrain.py sequenceModel ${maf_file} ${genome_counts} ./$(basename ${pretrained_model})
    """
    
    output_patterns = {
            "pretrained_model": "*_map.h5"
            }
    
    docker = "gcr.io/broad-getzlab-workflows/dig_docker:latest"
    
    resources = {
        "cpus-per-task": 4,
        "mem" : "12G" 
    }


class DIG_pretrain_genic(wolf.Task):
    name = "DIG_pretrain_genic"

    inputs = {
        "pretrained_model": None,
        "gene_data": None
    }

    script = """
    cp ${pretrained_model} ./
    DigPretrain.py genicModel ./$(basename ${pretrained_model}) ${gene_data}
    """

    output_patterns = {
            "pretrained_model": "*_map.h5"
            }

    docker = "gcr.io/broad-getzlab-workflows/dig_docker:latest"
    
    resources = {
        "cpus-per-task": 4,
        "mem" : "12G" 
    }


class DIG_test_coding(wolf.Task):
    name = 'DIG_test_coding'

    inputs = {
        "input_annot_maf" : None,
        "input_mut_map" : None,
        "cohort": None
    }

    script="""
    DigDriver.py geneDriver ${input_annot_maf} ${input_mut_map} --outdir . --outpfx ${cohort}.coding.dig
    """

    output_patterns = {
        "dig_results": "*.results.txt"
    }

    docker = "gcr.io/broad-getzlab-workflows/dig_docker:latest"
    resources = { "cpus-per-task": 2, "mem" : "20G" } 


class DIG_report_coding(wolf.Task):
    name = 'DIG_report_coding'

    inputs = {
        "input_results" : None,
        "cgc_list": None,
        "pancan_list": None,
        "cohort": None
    }

    script="""
    python3 /build/generate_dig_report_coding.py ${input_results} . ${cgc_list} ${pancan_list} --prefix_output ${cohort}
    """

    output_patterns = {
        "dig_report" : "*.html"
    }

    docker = "gcr.io/broad-getzlab-workflows/dig_docker:latest"
    resources = { "cpus-per-task": 2, "mem" : "20G" } 


class DIG_preprocess_element_model(wolf.Task):
    name = 'DIG_preprocess_element_model'

    input = {
        "input_bed" : None,
        "input_element_data" : None,
        "input_mut_map" : None,
        "ref_fasta" : None,
        "annot_name" : None
    }

    script="""
    cp ${input_mut_map} ./mutation_map.h5
    cp ${input_element_data} ./element_data.h5
    DigPreprocess.py preprocess_element_model ./element_data.h5 ./mutation_map.h5 ${ref_fasta} ${annot_name} --f-bed ${input_bed}
    """

    output_patterns = {
        "output_element_data" : "element_data.h5",
        "output_mut_map": "mutation_map.h5"
    }

    docker = "gcr.io/broad-getzlab-workflows/dig_docker:latest"
    resources = { "cpus-per-task": 2, "mem" : "20G" }


class DIG_element_model(wolf.Task):
    name = 'DIG_element_model'

    input = {
        "input_element_data" : None,
        "input_mut_map" : None,
        "annot_name" : None
    }

    script="""
    cp ${input_mut_map} ./mutation_map.h5
    cp ${input_element_data} ./element_data.h5
    DigPretrain.py elementModel ./mutation_map.h5 ./element_data.h5 ${annot_name}
    """

    output_patterns = {
        "output_element_data" : "element_data.h5",
        "output_mut_map": "mutation_map.h5"
    }

    docker = "gcr.io/broad-getzlab-workflows/dig_docker:latest"
    resources = { "cpus-per-task": 2, "mem" : "20G" }


class DIG_test_noncoding(wolf.Task):
    name = 'DIG_test_noncoding'

    inputs = {
        "input_annot_maf" : None,
        "input_mut_map" : None,
        "input_bed" : None,
        "annot_name" : None,
        "cohort" : None
    }

    script="""
    DigDriver.py elementDriver ${input_annot_maf} ${input_mut_map} ${annot_name} --f-bed ${input_bed} --outdir . --outpfx ${cohort}.${annot_name}.dig
    """

    output_patterns = {
        "dig_results": "*.results.txt"
    }

    docker = "gcr.io/broad-getzlab-workflows/dig_docker:latest"
    resources = { "cpus-per-task": 2, "mem" : "20G" } 


class DIG_report_noncoding(wolf.Task):
    name = 'DIG_report_noncoding'

    inputs = {
        "input_results" : None,
        "cgc_list": None,
        "pancan_list": None, 
        "annot_name" : None,
        "cohort" : None
    }

    script="""
    python3 /build/generate_dig_report_noncoding.py ${input_results} . ${cgc_list} ${pancan_list} ${annot_name} --prefix_output ${cohort}
    """

    output_patterns = {
        "dig_report" : "*.html"
    }

    docker = "gcr.io/broad-getzlab-workflows/dig_docker:latest"
    resources = { "cpus-per-task": 2, "mem" : "20G" } 


class DIG_results(wolf.Task):
    name = 'DIG_results'
    
    inputs = {
        "noncoding_htmls": None,
        "noncoding_results": None,
        "coding_html": None,
        "coding_result": None,
        "cgc_list": None,
        "pancan_list": None,
        "cohort" : None
    }

    script="""
    # Read all lines from .txt files
    noncoding_html_paths=$(cat ${noncoding_htmls})
    noncoding_result_paths=$(cat ${noncoding_results})
    
    # Convert the file paths into arrays
    noncoding_html_arr=($noncoding_html_paths)
    noncoding_result_arr=($noncoding_result_paths)
    
    # Copy each file to the current working directory
    echo "Copying results and reports to working directory..."
    for path in "${noncoding_html_arr[@]}"; do
        cp "$path" ./
    done
    for path in "${noncoding_result_arr[@]}"; do
        cp "$path" ./
    done
    cp ${coding_html} ./
    cp ${coding_result} ./
    
    # Generate combined p-values and report
    echo "Generating combined p-values and the associated report..."
    python3 /build/generate_dig_report_combined.py $(basename ${coding_result}) $(basename ${noncoding_result_arr[0]}) $(basename ${noncoding_result_arr[1]}) $(basename ${noncoding_result_arr[2]}) . ${cgc_list} ${pancan_list} --prefix_output ${cohort}
    
    # Generate final report
    echo "Generating final report..."
    python3 /build/generate_dig_report_main.py $(basename ${coding_html}) $(basename ${noncoding_html_arr[0]}) $(basename ${noncoding_html_arr[1]}) $(basename ${noncoding_html_arr[2]}) ./${cohort}_dig_report_combined.html . --prefix_output ${cohort}
    ls
    
    # Zip results
    echo "Zipping all results and reports..."
    zip dig_results.zip *.txt *.html
    """
    output_patterns = {
        "dig_results" : "*.zip"
    }

    docker = "gcr.io/broad-getzlab-workflows/dig_docker:latest"
    resources = { "cpus-per-task": 2, "mem" : "20G" }


class DIG_gather_noncoding(wolf.Task):
    name = "Gather_noncoding"
    
    inputs = {"gather_parameter"}
    
    script = """cat ${gather_parameter} > output.txt
    """
    
    outputs = { "output" : "output.txt" }