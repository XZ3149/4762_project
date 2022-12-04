import csv
import gzip
import os
import scipy.io
import pandas as pd
import math
import torch
import numpy as np
import matplotlib.pyplot as plt # for plotting
import seaborn as sns
from sklearn.metrics import mean_squared_error
from pca import pca
import warnings
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

chr_dic = {'chr1': 0,'chr2': 1,'chr3': 2,'chr4': 3,'chr5': 4,'chr6': 5,'chr7': 6,'chr8': 7,'chr9': 8,'chr10': 9,
               'chr11': 10,'chr12': 11,'chr13': 12, 'chr14': 13,'chr15': 14,'chr16': 15,'chr17': 16,'chr18': 17,
               'chr19': 18,'chr20': 19,'chr21': 20, 'chr22': 21}



def mat_to_csv():
    """
    This is the function used to convert mat to csv. It will output two csv, one for ATAC peaks, one for RNA-seq
    It also perfrom position calculation for ATAC_peaks and RNA-seq

    """
    matrix_dir = "C:\\Users\\we609\\Desktop\\cs 4762\\project\\3K_brain\\filtered_feature_bc_matrix"
    mat = scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz"))
    cell_number = 3233
    threshold = 0.3



    features_path = os.path.join(matrix_dir, "features.tsv.gz")
    feature_ids = [row[0] for row in csv.reader(gzip.open(features_path, mode="rt"), delimiter="\t")]
 
    # list of gene names, e.g. 'MIR1302-2HG'
    gene_names = [row[1] for row in csv.reader(gzip.open(features_path, mode="rt"), delimiter="\t")]
 
    # list of feature_types, e.g. 'Gene Expression'
    feature_types = [row[2] for row in csv.reader(gzip.open(features_path, mode="rt"), delimiter="\t")]
    barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
    barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path, mode="rt"), delimiter="\t")]
    feature_chrom = [row[3] for row in csv.reader(gzip.open(features_path, mode="rt"), delimiter="\t")]
    TSS_start = [row[4] for row in csv.reader(gzip.open(features_path, mode="rt"), delimiter="\t")]
    TSS_end = [row[5] for row in csv.reader(gzip.open(features_path, mode="rt"), delimiter="\t")]



    matrix = pd.DataFrame.sparse.from_spmatrix(mat)
    matrix.columns = barcodes
    matrix.insert(loc=0,column="TSS_end", value=TSS_end)
    matrix.insert(loc=0,column="TSS_start", value=TSS_start)
    matrix.insert(loc=0, column="chromosome", value=feature_chrom)
    matrix.insert(loc=0, column="feature_id", value=feature_ids)
    matrix.insert(loc=0, column="gene", value=gene_names)
    matrix.insert(loc=0, column="feature_type", value=feature_types)
    matrix['not_0_count'] = cell_number - matrix.isin([0]).sum(axis=1)
    # display matrix
    print(matrix)

    gene_expression = matrix[matrix['feature_type']=="Gene Expression"]
    gene_expression = gene_expression[gene_expression['not_0_count'] > (cell_number * threshold)]
    gene_expression['TSS_start'] = pd.to_numeric(gene_expression['TSS_start'])
    gene_expression['TSS_end'] = pd.to_numeric(gene_expression['TSS_end'])
    gene_expression['TSS'] = ((gene_expression['TSS_start'] + gene_expression['TSS_end'])//2) + 1
    print(gene_expression)
    gene_expression.to_csv("expression_level.csv", index=False)






    peak =  matrix[matrix['feature_type'] =="Peaks"]
    peak = peak[peak['not_0_count'] > (cell_number * threshold * 0.5)]
    print(peak)
    peak['TSS_start'] = pd.to_numeric(peak['TSS_start'])
    peak['TSS_end'] = pd.to_numeric(peak['TSS_end'])
    peak['avg_position'] = ((peak['TSS_start'] + peak['TSS_end'])//2) + 1
    peak['length'] = peak['TSS_end'] - peak['TSS_start']
    peak.to_csv("peak.csv", index=False)





def normalization_log_read_depth(gene_expression_file, peak_file):
    """this function normalize the gene_expression level and convert peak value into 
    binary value 
    """
    
    gene_expression = pd.read_csv(gene_expression_file)
    normalized_gene_expression = gene_expression.iloc[:,6:-2]
    avg_read = sum((normalized_gene_expression.sum()))/len(normalized_gene_expression.sum())
    
    normalized_gene_expression = np.log((normalized_gene_expression * avg_read / (normalized_gene_expression.sum())) + 1)

    normalized_gene_expression = normalized_gene_expression.round(4)
    # round up the number to reduce file size
    normalized_gene_expression.insert(loc=0, column='TSS', value= gene_expression['TSS'])
    normalized_gene_expression.insert(loc=0, column='chromosome', value= gene_expression['chromosome'])
    normalized_gene_expression.insert(loc=0, column='feature_id', value= gene_expression['feature_id'])
    normalized_gene_expression.insert(loc=0, column='gene', value= gene_expression['gene'])
    normalized_gene_expression.to_csv(f'log_read_depth_normalized_{gene_expression_file}', index=False)
    
    
    peak = pd.read_csv(peak_file)
    
    
    normalize_peak = peak.iloc[:,6:-3]
    avg_read = sum((normalize_peak.sum()))/len(normalize_peak.sum())
    normalize_peak = np.log((normalize_peak * (avg_read / normalize_peak.sum())) + 1)


    normalize_peak = normalize_peak.round(4)
    normalize_peak.insert(loc=0, column='avg_position', value= peak['avg_position'])
    normalize_peak.insert(loc=0, column='length', value= peak['length'])
    normalize_peak.insert(loc=0, column='chromosome', value= peak['chromosome'])
    normalize_peak.to_csv(f'log_read_depth_normalized_{peak_file}', index=False)
    







def normalization_std(gene_expression_file, peak_file):
    """this function normalize the gene_expression level and peak read by z-score
    """
    
    gene_expression = pd.read_csv(gene_expression_file)
    normalized_gene_expression = gene_expression.iloc[:,6:-2]
    #avg_read = sum((normalized_gene_expression.sum()))/len(normalized_gene_expression.sum())
    
    #normalized_gene_expression = (normalized_gene_expression * avg_read / (normalized_gene_expression.sum() ))
    normalized_gene_expression = (normalized_gene_expression - normalized_gene_expression.mean())/normalized_gene_expression.std()
    normalized_gene_expression = normalized_gene_expression.round(4)
    normalized_gene_expression.insert(loc=0, column='TSS', value= gene_expression['TSS'])
    normalized_gene_expression.insert(loc=0, column='chromosome', value= gene_expression['chromosome'])
    normalized_gene_expression.insert(loc=0, column='feature_id', value= gene_expression['feature_id'])
    normalized_gene_expression.insert(loc=0, column='gene', value= gene_expression['gene'])
    normalized_gene_expression.to_csv("normalize_expression_level.csv", index=False)
    
    
    
    
    peak = pd.read_csv(peak_file)
    
    
    binary_peak = peak.iloc[:,6:-3]
    #avg_read = sum((binary_peak.sum()))/len(binary_peak.sum())
    #binary_peak = (binary_peak * (avg_read / binary_peak.sum() ))
    binary_peak = (binary_peak - binary_peak.mean())/binary_peak.std()

    binary_peak = binary_peak.round(4)
    binary_peak.insert(loc=0, column='avg_position', value= peak['avg_position'])
    binary_peak.insert(loc=0, column='length', value= peak['length'])
    binary_peak.insert(loc=0, column='chromosome', value= peak['chromosome'])
    binary_peak.to_csv("normalize_peak.csv", index=False)
    
    
# not in use at this moment, maybe will activate when we want to test logistic function
def binarize(gene_expression_file, peak_file):
    """this function normalize the gene_expression level and convert peak value into 
    binary value 
    """
    
    gene_expression = pd.read_csv(gene_expression_file)
    normalized_gene_expression = (gene_expression.iloc[:,6:-2] > 0).astype(int)
    normalized_gene_expression.insert(loc=0, column='TSS', value= gene_expression['TSS'])
    normalized_gene_expression.insert(loc=0, column='chromosome', value= gene_expression['chromosome'])
    normalized_gene_expression.insert(loc=0, column='feature_id', value= gene_expression['feature_id'])
    normalized_gene_expression.insert(loc=0, column='gene', value= gene_expression['gene'])
    normalized_gene_expression.to_csv("binarize_expression_level.csv", index=False)
    
    
    
    
    peak = pd.read_csv(peak_file)
    binary_peak = (peak.iloc[:,6:-3] > 0).astype(int)
    binary_peak.insert(loc=0, column='avg_position', value= peak['avg_position'])
    binary_peak.insert(loc=0, column='length', value= peak['length'])
    binary_peak.insert(loc=0, column='chromosome', value= peak['chromosome'])
    binary_peak.to_csv("binary_peak.csv", index=False)
    
    
    
def csv_formatting(gene_expression_file, peak_file):
    """this function normalize the gene_expression level and convert peak value into 
    binary value 
    """
    
    gene_expression = pd.read_csv(gene_expression_file)
    normalized_gene_expression = (gene_expression.iloc[:,6:-2])
    normalized_gene_expression.insert(loc=0, column='TSS', value= gene_expression['TSS'])
    normalized_gene_expression.insert(loc=0, column='chromosome', value= gene_expression['chromosome'])
    normalized_gene_expression.insert(loc=0, column='feature_id', value= gene_expression['feature_id'])
    normalized_gene_expression.insert(loc=0, column='gene', value= gene_expression['gene'])
    normalized_gene_expression.to_csv("reduced_expression_level.csv", index=False)
    
    
    
    
    peak = pd.read_csv(peak_file)
    binary_peak = (peak.iloc[:,6:-3])
    binary_peak.insert(loc=0, column='avg_position', value= peak['avg_position'])
    binary_peak.insert(loc=0, column='length', value= peak['length'])
    binary_peak.insert(loc=0, column='chromosome', value= peak['chromosome'])
    binary_peak.to_csv("reduced_peak.csv", index=False)
          
    
    

    
# not in use, maybe activate when sample size get larger
def patch_generation(df):
    """ this function split df into patches according to chromosome
    input: df 
    output: a list of df based on chromosome
    currently only account for autosomes
    """
    result = []
    for i in range(len(chr_dic)):
        result += [result[result['chromosome'] == f'chr{i+1}']]

    return result




def gene_extraction(gene_id, chr_name, gene_expression_df, peak_df, interval):
    """
    This is the function for gene extraction. It will output one matrix and two vectors. 
    Output 1: peaks within the interval around TSS in matrix
    Output 2: gene expression for this gene in vector
    Output 3: distants between peaks and TSS of the gene in log scale

    """

    
    gene = gene_expression_df[gene_expression_df['feature_id'] == gene_id]
    TSS = int(gene['TSS'])    
    y = gene.iloc[:,4:]
    y = torch.from_numpy(y.values).float()
    y = torch.transpose(y, 0, 1)
    y = y[:,0]
    
    peaks = peak_df[(peak_df['chromosome'] == chr_name) & (peak_df['avg_position'] >= (TSS-interval)) & (peak_df['avg_position'] <= (TSS+interval))].copy()
    peaks['distant'] = abs(peaks['avg_position'] - TSS) + 2
    # +2 to avoid log(distant) = 0 and not defined
    x = peaks.iloc[:,3:-1]
    x = torch.from_numpy(x.values).float()
    x = torch.transpose(x, 0, 1)
    
    distant = torch.from_numpy(peaks['distant'].values).float()
    distant = torch.log(distant)

    return x,y,distant



def pca_analysis(expression_df, components = 5, fig_name = 'PCA_result.png'):
    """
    This is the function use to perform pca dimension reduction, it use pca library
    it will return an [N x components] matrix in dataframe format

    """
    y = expression_df.iloc[:, 4:]
    model = pca(n_components=0.95)

    # Reduce the data towards 3 PCs
    model = pca(n_components=components )

    # Fit transform
    results = model.fit_transform(y)
    fig, ax = model.plot()
    fig.savefig('PCA_result.png')
    
    # return an df with [N x components] dimensions for k-nearest neighbors computation
    return results['loadings'].T


def k_nearest_neighbors_computation(df, k = 30):
    """
    This is the function use to generate k_nearest_neighbors matrix, it will output an NxN matrix.
    Euclidean distance will be used to calculate distant

    """
    samples = df.index.values.tolist()
    output_df = pd.DataFrame(samples, columns = ['ID'])

    
    for sample in samples:
        sample_row = list(df.loc[sample])
        
        k_nearest_list = []
        neighbors_dic = {}
        for index,row in df.iterrows():
            temp_row = list(row)
            distant = 0
            neighbors_dic[index] = 0
            for i in range(len(temp_row)):
                distant += (sample_row[i] - temp_row[i])**2
            distant = math.sqrt(distant)
            
            if len(k_nearest_list) < k:
                k_nearest_list += [[distant,index]]
            else:
                current_furthest = max(k_nearest_list)
                if current_furthest[0] > distant:
                    # if this row is more closer to the sample than Kth nearest neighbors, replace that neighbor
                    k_nearest_list[k_nearest_list.index(current_furthest)] = [distant,index]
        for neighbor in k_nearest_list:
            neighbors_dic[neighbor[1]] = 1
        	
        	
        output_df[sample]= output_df['ID'].map(neighbors_dic)
        output = output_df.iloc[:,1:]
        output = torch.from_numpy(output.values).float()
            
    return output

def k_nearest_neighbors_smooth(peaks_matrix, k_nearest_matrix, k = 30):
    """
    this is the function using k_nearest_matrix to smooth peaks_matrix
    peaks_maxtrix: [P x N]
    k_nearest_matrix: [N x N] 
    It is the helper function for smooth_all_peaks()

    """
    result = (peaks_matrix @ k_nearest_matrix) * 1/k
    return result
    


def smooth_all_peaks(peaks_df, k_nearest_matrix, k = 30):
    """ 
    this is the function using to smooth put all the peaks, it required the 
    dataframe from peak.csv
    """
    
    samples_infor = peaks_df.iloc[:,0:3]
    peaks_matrix = peaks_df.iloc[:,3:]
    samples = list(peaks_matrix.columns)
    peaks_matrix = torch.from_numpy(peaks_matrix.values).float()
    
    smooth_matrix = k_nearest_neighbors_smooth(peaks_matrix, k_nearest_matrix, k)
    smooth_df = pd.DataFrame(smooth_matrix.numpy(), columns= samples)
    output = pd.concat([samples_infor, smooth_df], axis=1)
    
    return output


def aggregate_smooth_function(expression_df, peaks_df, components = 5,k= 30, fig_name = 'PCA_result.png'):
    """ 
    this is the function that group all function related k nearest neigbhor smooth functions
    """
    
    reduce_expression_df = pca_analysis(expression_df, components, fig_name)
    k_nearest_matrix =  k_nearest_neighbors_computation(reduce_expression_df, k = 30)
    x_smoothed = smooth_all_peaks(peaks_df, k_nearest_matrix, k = 30)
    
    return x_smoothed
    
    
    


def sample_split(df_gene_expression, df_peaks, N, random_seed = 123):
    """this is the function that split the data into 60% training, 20% validation, 
    and 20% testing
    It will return 6 dataframe. 
    """
    np.random.seed(random_seed) # for reproducibility
    rand_perm = np.random.permutation(N)
    train_idx = rand_perm[:int(np.ceil(0.8 * N))]
    val_idx = rand_perm[int(np.ceil(0.8 * N)):int(np.ceil(0.9 * N))]
    test_idx = rand_perm[int(np.ceil(0.9 * N)):]
    
    # include the first four information columns of expression
    train_y_idx = np.append([0,1,2,3], train_idx + 4)
    val_y_idx = np.append([0,1,2,3], val_idx + 4)
    test_y_idx = np.append([0,1,2,3], test_idx + 4)
    
    #include the first three information column of peak
    train_x_idx = np.append([0,1,2], train_idx + 3)
    val_x_idx = np.append([0,1,2], val_idx + 3)
    test_x_idx = np.append([0,1,2], test_idx + 3)


    X_train, X_val, X_test = df_peaks.iloc[:,train_x_idx], \
        df_peaks.iloc[:,val_x_idx], df_peaks.iloc[:,test_x_idx]

    y_train, y_val, y_test = df_gene_expression.iloc[:,train_y_idx], \
        df_gene_expression.iloc[:,val_y_idx], df_gene_expression.iloc[:,test_y_idx]    


    return X_train, X_val, X_test, y_train, y_val, y_test


def ridge_regression_optimaization(X_train, y_train, X_val, y_val, distants, 
                                  learning_rate = 0.002, iterations = 200, step = 0.5):
    """
    This is the function used to fit lasso regression and find optimal hyperparameters
    it will return the best beta it can find.

    """
    lambdas = 10 ** np.arange(-5, 2, step = step)
    val_rmse = np.zeros_like(lambdas)
    betas = []

    for i,lamb in enumerate(lambdas): # try different settings of lambda
        beta = ridge_regression_gd( X_train, y_train,lamb, distants, learning_rate, iterations)
        if beta == None:
            val_rmse[i] = float('inf')
            betas += [None]
        else:
            pred_val = X_val @ beta # make predictions on the validation set
            val_rmse[i] = np.sqrt(torch.mean((y_val - pred_val)**2).item()) # item() gets the scalar value
            betas += [beta]
        
    optimal_lamb = lambdas[np.argmin(val_rmse)]
    output_beta = betas[np.argmin(val_rmse)]

    return output_beta
    

def lasso_regression_optimaization(X_train, y_train, X_val, y_val, distants, 
                                  learning_rate = 0.002, iterations = 200, step = 0.5):
    """
    This is the function used to fit lasso regression and find optimal hyperparameters
    it will return the best beta it can find.

    """
    lambdas = 10 ** np.arange(-5, 2, step = step)
    val_rmse = np.zeros_like(lambdas)
    betas = []

    for i,lamb in enumerate(lambdas): # try different settings of lambda
        beta = fit_lasso( X_train, y_train,lamb, distants, learning_rate, iterations)
        if beta == None:
            val_rmse[i] = float('inf')
            betas += [None]
        else:
            pred_val = X_val @ beta # make predictions on the validation set
            val_rmse[i] = np.sqrt(torch.mean((y_val - pred_val)**2).item()) # item() gets the scalar value
            betas += [beta]
        
    optimal_lamb = lambdas[np.argmin(val_rmse)]
    output_beta = betas[np.argmin(val_rmse)]

    return output_beta
    




def fit_lasso(X, y, lamb, distants, learning_rate = 0.001, iterations = 100): 
    """
    This is the function use to fit lasso regression using gradient descent

    """
    beta = torch.zeros(X.shape[1], device=X.device).float()
    losses = []
    previous_loss = -100
    for it in range(iterations):
        err = y - (X @ beta)
        loss = .5 * (err * err).mean() + lamb * (beta * distants).sum() 
        # return None when fail to converge
        if np.isnan(loss.item()): 
            return None
        # if the improvement on loss change very little for each iteration, end the interation to enhance computation speed
        if abs(loss - previous_loss)/abs(loss) <= 0.001:
            break
        losses.append(loss.item())
        previous_loss = loss
        grad = - ((X.transpose(0,1) @ err)/X.shape[0] ) +  lamb * torch.sign(beta) * distants 
        beta -= learning_rate * grad
    return beta




def ridge_regression_gd(X, y, lamb, distants, learning_rate = 0.001, iterations = 200): 
    """
    This is the function use to fit ridge regression using gradient descent

    """
    beta = torch.zeros(X.shape[1], device=X.device)
    losses = []
    previous_loss = -100
    for k in range(iterations): # in practice we would use an appropriate "stopping criteria"
        err = y - (X @ beta)
        loss = .5 * (err * err).mean() + .5 * lamb * (beta * beta * distants).sum()
        # return None when fail to converge
        if np.isnan(loss.item()): 
            return None
        # if the improvement on loss change very little for each iteration, end the interation to enhance computation speed
        if abs(loss - previous_loss)/abs(loss) <= 0.001:
            break

        losses.append(loss.item())
        grad = - (X.transpose(0,1) @ err)/X.shape[0] +  lamb * beta 
        previous_loss = loss
        beta -= learning_rate * grad
    return beta




























