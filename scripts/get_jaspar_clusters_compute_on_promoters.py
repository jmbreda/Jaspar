import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'PWM')
import PWM
import torch
import h5py
import argparse

def parse_argument():
    parser = argparse.ArgumentParser(description='Plot histogram of experiments QC')
    parser.add_argument('--promoterome_bed'
        ,required=True
        ,type=str
        ,help="Promoterome bed file")
    parser.add_argument('--promoterome_hdf5'
        ,required=True
        ,type=str
        ,help="Promoterome hdf5 file")
    parser.add_argument('--outfile_pt'
        ,required=True
        ,type=str
        ,help="matrix of convolution pt format")
    parser.add_argument('--outfile_hdf5'
        ,required=True
        ,type=str
        ,help="matrix of convolution hdf5 format")
    parser.add_argument('--genome'
        ,required=True
        ,type=str
        ,help="Genome")
    parser.add_argument('--window_kb'
        ,required=True
        ,type=int
        ,help="window size in kb")
    parser.add_argument('--threads'
        ,required=False
        ,type=int
        ,default=1
        ,help="Number of threads")


    return parser.parse_args()

if __name__ == '__main__':

    args = parse_argument()

    # fix number of threads for torch
    torch.set_num_threads(args.threads)

    print('load input files...')
    #infile_jaspar_clusters_to_tf = 'resources/interactive_trees/JASPAR_2022_matrix_clustering_vertebrates_CORE_tables/clusters_motif_names.tab'
    #infile_jaspar_clusters_tf_corr = 'resources/interactive_trees/JASPAR_2022_matrix_clustering_vertebrates_CORE_tables/pairwise_compa.tab'
    #clusters_to_tf = pd.read_csv(infile_jaspar_clusters_to_tf,sep='\t',header=None)
    #clusters_tf_corr = pd.read_csv(infile_jaspar_clusters_tf_corr,sep='\t',low_memory=False)

    # get promoterome and sequence
    promoterome = pd.read_csv(args.promoterome_bed,sep='\t')
    promoterome_hdf5 = h5py.File(args.promoterome_hdf5, 'r')
    Prom_seq = torch.from_numpy(promoterome_hdf5['sequence'][:]).float()
    
    # get pwm_matrix
    PWMs = PWM.get_PWM()
    N_PWM = len(PWMs)
    l_max=  max([PWMs[m].shape[0] for m in PWMs])
    PWM_tensor = torch.zeros([N_PWM,l_max,4])
    logZ = torch.zeros(N_PWM)
    nuc_perm = np.eye(4).astype(bool)
    print('compute convolution..')
    for i in np.sort(list(PWMs.keys())):
        L = PWMs[i].shape[0]
        # recursive forward-backward algorithm
        # Z_l = Z_l-1 * [ sum_{i=1}^4 exp( w_l,i ) ]
        #Z_l_minus_1 = 1
        #for l in range(L):
        #    Z_l = Z_l_minus_1 * np.exp(PWM[i][l,:]).sum()
        # Z_l is 1 as np.exp(PWM[i][l,:]).sum() = 1 for any l

        # fill in tensor
        l_diff = (l_max - L)/2
        PWM_tensor[i,:,:] = torch.nn.functional.pad(torch.from_numpy(PWMs[i]),pad=(0,0,int(np.ceil(l_diff)),int(np.floor(l_diff))))
    del PWMs

    # get background frequency correspondiong to PWMs
    background = np.log(PWM.get_background_frequency(args.genome))
    PWM_background_tensor = np.zeros([N_PWM,l_max,4])
    for i in range(PWM_tensor.shape[0]):
        idx_motif = PWM_tensor[i].sum(axis=1)!=0
        PWM_background_tensor[i,idx_motif,:] = background[None,:].repeat(idx_motif.sum(),axis=0)
    PWM_background_tensor = torch.from_numpy(PWM_background_tensor).float()

    # input:   (batch_size, in_channels, input width)
    # filters: (out_channels, in_channels​, kernel width)
    convolution_bg = torch.exp( torch.nn.functional.conv1d( torch.transpose(Prom_seq,1,2), torch.transpose(PWM_background_tensor,1,2) ) )
    convolution    = torch.exp( torch.nn.functional.conv1d( torch.transpose(Prom_seq,1,2), torch.transpose(PWM_tensor,1,2) ) )
    # normalize for backgroud prob.
    convolution /= convolution + convolution_bg
    del convolution_bg
    # normalize for each possible motif.
    for p in range(convolution.shape[0]):
        convolution[p] /= convolution[p].sum(axis=0)

    print('save results..')
    # save in hdf5
    with h5py.File(args.outfile_hdf5, 'w') as hf:
        for p in promoterome.index:
            prom = hf.create_dataset(promoterome.at[p,'gene'] + '/' + promoterome.at[p,'id'],data=convolution[p])

    # save tensor
    torch.save(convolution,args.outfile_pt)
    print('done!')