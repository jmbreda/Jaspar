import numpy as np
import os
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

def read_transfac(infile):

    # Read jaspar cluster motifs
    # line[0]:
    #   AC name (ignore)
    #   ID id (-> name)
    #   DE Jaspar ID (?) (ignore)
    #   CC various info.. (ignore)
    #       merged_AC: TF members of cluster
    #   P0 alphabet in motif (ACGT) (ignore)
    #   NN (digit) position in motif (1-based) (fill in)
    #   XX newline
    #   // end of motif 
    with open(infile,'r') as f:
        lines = f.readlines()
    
    MOTIFS = {}
    for l in lines:

        line = l.strip().split()

        if line[0] in ['AC','DE','XX','CC']:
            continue
        elif line[0] == 'P0':
            if not (''.join(line[1:])).upper() == 'ACGT':
                print(f"Error in motif {name}: P0: {(''.join(line[1:])).upper()}")
        elif line[0] == 'ID':
            name = line[1]
            name = int(name.replace('cluster_',''))-1
            MOTIFS[name] = []
        elif line[0].isdigit():
            MOTIFS[name].append([float(i) for i in line[1:]])
        elif line[0] == '//':
            MOTIFS[name] = np.array(MOTIFS[name])
        else:
            print('Unknown line ID:')
            print(line[0])
            continue
    
    return MOTIFS

def get_background_frequency(genome):
    # [A C G T]
    background_frequency = {
        'mm10': np.array([0.298, 0.198, 0.202, 0.302]),
        'mm39': np.array([0.298, 0.198, 0.202, 0.302]),
        'hg19': np.array([0.303, 0.198, 0.197, 0.302]),
        'hg38': np.array([0.303, 0.198, 0.197, 0.302])
    }
    return background_frequency[genome]

def get_PPM(pseudo_count=True,background_norm=None):

    # read motifs counts
    infile = 'resources/interactive_trees/JASPAR_2022_matrix_clustering_vertebrates_CORE_cluster_root_motifs.tf'
    PPM = read_transfac(infile)

    # add pseudocount if required (default = False)
    if pseudo_count:
        for m in PPM:
            PPM[m] += 1

    # transform counts to positional probability matrix
    for m in PPM:
        PPM[m] /= PPM[m].sum(axis=1,keepdims=1)

    # Get background frequency
    if background_norm != None:
        if not (background_norm in ['mm10','mm39','hg19','hg38']):
            print("Unknown background freq. Choice: None, mm10, mm39, hg19, hg38")
            return None
        else:
            B = get_background_frequency(background_norm) # mm10/hg38
            for m in PPM.keys():
                PPM[m] /= B
                PPM[m] /= PPM[m].sum(axis=1,keepdims=1)

    return PPM

def get_PWM(pseudo_count=True,background_norm=None):
    PWM = get_PPM(pseudo_count,background_norm)
    # get log prob.
    for m in PWM:
        PWM[m] = np.log(PWM[m])

    return PWM

def get_IC(pseudo_count=True,background_norm=None):
    PPM = get_PPM(pseudo_count,background_norm)

    # total information content with 4 letters in bits
    IC_tot = np.log2(4)
    IC = {}
    for m in PPM.keys():
        # uncertainty per position in bits (entropy)
        H = - np.nansum( PPM[m]*np.log2(PPM[m]), axis=1 )
        IC[m] = IC_tot - H
        IC[m] = IC[m][:,None]*PPM[m]

    return IC

def print_PWM():

    for pseudo_count in [False,True]:
        for background_norm in [False,True]:
            PWM = get_PWM(pseudo_count,background_norm)

            outfolder = 'data/PWM'
            if pseudo_count:
                outfolder += '_pc'
            if background_norm:
                outfolder += '_bg'

            if not os.path.exists(outfolder):
                os.mkdir(outfolder)

            for m in PWM.keys():
                with open(f'{outfolder}/{m}.txt','w') as f:
                    f.write(f'NA\t{m}\n')
                    f.write('P0\tA\tC\tG\tT\n')
                    for i in range(PWM[m].shape[0]):
                        if i<9:
                            f.write('0')
                        f.write(f'{i+1}')
                        for p in range(4):
                            f.write(f'\t{PWM[m][i][p]}')
                        f.write('\n')

def plot_IC_logo(pseudo_count=True,background_norm=True):
    IC = get_IC(pseudo_count,background_norm)

    fig_fold='Fig/IC'
    for tf in IC.keys():
        to_plot = IC[tf]
        to_plot = pd.DataFrame(to_plot,columns = ['A','C','G','T'])

        fig, ax = plt.subplots()
        #logomaker.Logo(to_plot,ax=ax)
        ax.axis('off')
        fig.tight_layout()
        fig.set_size_inches([to_plot.shape[0]/2,2])
        fig.savefig(f'{fig_fold}/{tf}.pdf')
        plt.close('all')

        # plot rev_cmp
        to_plot = np.flip( IC[tf] )
        tf += '_rev_cmp'
        to_plot = pd.DataFrame(to_plot,columns = ['A','C','G','T'])

        fig, ax = plt.subplots()
        #logomaker.Logo(to_plot,ax=ax)
        ax.axis('off')
        fig.tight_layout()
        fig.set_size_inches([to_plot.shape[0]/2,2])
        fig.savefig(f'{fig_fold}/{tf}.pdf')
        plt.close('all')