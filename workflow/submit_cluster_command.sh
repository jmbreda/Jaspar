#!/bin/bash
snakemake -s workflow/Snakefile \
          -j 999 \
          --cluster-config config/cluster.json \
          --cluster "sbatch --job-name {rule} \
                            --qos {cluster.qos} \
                            --time {cluster.time} \
                            --mem {cluster.mem} \
                            --nodes {cluster.nodes} \
                            --ntasks {cluster.ntasks} \
                            --cpus-per-task {cluster.cpus-per-task} \
                            --output {cluster.stdout} \
                            --error {cluster.stderr}"