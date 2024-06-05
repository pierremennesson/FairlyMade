import pandas as pd
import numpy as np
import networkx as nx


        
#FEATURES ENCODING

class FeaturesEncoderDecoder:
    """Main class to encode a sample with categorical features into
    a vector that will be fed to a predictive model

    Parameters
    ----------
    categorical_features : names of the categorical features appearing in the dataframe

    multi_valued_features : names of the multi-valued features appearing in the dataframe

    boolean_features : names of the boolean_features appearing in the dataframe

    labels : a dictionary whose entries will be like (feature,feature_labels) for a 
    categorical (resp. muti-valued) feature taking value in the set (resp. in the power set of)
    feature_labels

    """

    def __init__(self, categorical_features=['product_type','nb_components'],
                       multi_valued_features=['composition','raw_material_country','weaving_country','dyeing_country','manufacturing_country'],
                       boolean_features=['plane_in_transports'],
                       numerical_features=['resource_use_fossils']):


        self.categorical_features=categorical_features
        self.multi_valued_features=multi_valued_features
        self.boolean_features=boolean_features
        self.numerical_features=numerical_features

        self.n_binary_features=None
        self.n_numerical_features=len(self.numerical_features)
        self.labels={}



    def encode_dataframe(self,df,compute_labels=False):
        """This functions encodes samples into a fixed-size binary vector


        Parameters
        ----------
        df : a pandas dataframes containing the samples

        compute_labels : whether to compute the possible labels for categorical 
        or multi-valued feature. Should be set to True only the first time 
        preprocessing the dataframe.


        Returns
        -------
        encoded_samples : a numpy array of shape (number of samples, number of binary coordinates)
        that store our processed samples
        """


        n_binary_features=0
        for feature in self.categorical_features:
            if compute_labels:
                feature_labels=df[feature].unique()
                feature_labels=sorted(feature_labels)
                self.labels[feature]=np.array(feature_labels)
                n_binary_features+=len(feature_labels)
            else:
                feature_labels=self.labels[feature]
            for label in feature_labels:
                col="%s_%s"%(feature,label)
                df[col]=df[feature].apply(lambda x:x==label)

        for feature in self.multi_valued_features:
            df[feature]=df[feature].apply(lambda x: [] if not(isinstance(x,str)) else x.split(','))
            df[feature]=df[feature].apply(lambda l:[x.replace(' ','') for x in l])
            if compute_labels:
                feature_labels=list(set([elem for l in df[feature] for elem in l]))
                feature_labels=sorted(feature_labels)
                self.labels[feature]=np.array(feature_labels)
                n_binary_features+=len(feature_labels)

            else:
                feature_labels=self.feature_labels
            for label in feature_labels:
                col="%s_%s"%(feature,label)
                df[col]=df[feature].apply(lambda l:label in l)

        if compute_labels:
            n_binary_features+=len(self.boolean_features)
            self.n_binary_features=n_binary_features

        X_cat=np.array(df[["%s_%s"%(feature,label) for feature,feature_labels in self.labels.items() for label in feature_labels]],dtype='float16')
        X_bool=np.array(df[self.boolean_features])
        X_num=np.array(df[self.numerical_features])
        encoded_samples=np.concatenate([X_cat,X_bool,X_num],axis=1)
        return encoded_samples


    def compute_features(self,nx_tree,root=0):
        """This functions assigns to each node of the networkx
        tree a binary vector whose coordinates are the binary features 
        assignments if the feature was used in the decision path orelse -1.


        Parameters
        ----------
        nx_tree : an networkx tree graph

        """
        paths=nx.shortest_path(nx_tree,root)
        for node,path in paths.items():
            features=[None]*(self.n_binary_features+self.n_numerical_features)
            for k in range(len(path)-1):
                splitting_feature_index=nx_tree.nodes()[path[k]]['splitting_feature']
                if splitting_feature_index<self.n_binary_features:
                    if nx_tree.get_edge_data(path[k],path[k+1])['type']=='left':
                        features[splitting_feature_index]=0
                    else:
                        features[splitting_feature_index]=1
                else:
                    threshold=nx_tree.nodes()[path[k]]['threshold']
                    if nx_tree.get_edge_data(path[k],path[k+1])['type']=='left':
                        if features[splitting_feature_index] is None:
                            features[splitting_feature_index] = (-np.inf,threshold)
                        else:
                            features[splitting_feature_index]=(features[splitting_feature_index][0],threshold)
                    else:
                        if features[splitting_feature_index] is None:
                            features[splitting_feature_index] = (threshold,np.inf)
                        else:
                            features[splitting_feature_index]=(threshold,features[splitting_feature_index][1])

            nx_tree.nodes()[node]['features']=features


    def decode_key_configuration(self,leaf_configuration):
        """This functions decodes a leaf_configuration to a more
        interpretable dictionary.


        Parameters
        ----------
        leaf_configuration : a vector with 0s and 1s indicating a 
        coordinate was assigned this value when going down the tree
        orelse -1


        Returns
        -------
        decoded_configuration : a dictionary indicating either the
        label value or forbidden values for a categorical feature, 
        several forbidden values and mandatory values for a muti-valued 
        categorical feature and the boolean value for a boolean feature if
        it was assigned orelse None
        """
        decoded_configuration={}
        N=0
        for feature in self.categorical_features:
            feature_labels=self.labels[feature]
            leaf_configuration_feature=np.array(leaf_configuration[N:N+len(feature_labels)])
            if np.all(leaf_configuration_feature!=1):
                decoded_configuration[feature]={'forbidden_labels':feature_labels[np.where(leaf_configuration_feature==0)[0]]}
            else:
                label=feature_labels[np.where(leaf_configuration_feature==1)[0][0]]
                decoded_configuration[feature]={'label':label}
            N+=len(feature_labels)            

        for feature in self.multi_valued_features:
            feature_labels=self.labels[feature]
            leaf_configuration_feature=np.array(leaf_configuration[N:N+len(feature_labels)])
            decoded_configuration[feature]={'forbidden_labels':feature_labels[np.where(leaf_configuration_feature==0)[0]],
                                            'mandatory_labels':feature_labels[np.where(leaf_configuration_feature==1)[0]]}
            N+=len(feature_labels)            

        for feature in self.boolean_features:
            leaf_configuration_bool=leaf_configuration[N]
            decoded_configuration[feature]=bool(leaf_configuration_bool) if leaf_configuration_bool is not None else None
            N+=1          

        for feature in self.numerical_features:
            leaf_configuration_num=leaf_configuration[N]
            decoded_configuration[feature]={'min':leaf_configuration_num[0],'max':leaf_configuration_num[1]} if leaf_configuration is not None else None
            N+=1          
        return decoded_configuration



#CONFIGURATIONS EXTRACTION
    
def convert_tree(tree):
    """This functions converts an sklearn tree object into
    a networkx graph that is easier to manipulate


    Parameters
    ----------
    tree : an sklearn tree object


    Returns
    -------
    nx_tree : the networkx converted graph
    """
    splitting_features=tree.feature
    thresholds=tree.threshold
    values=tree.value[:,0,0]
    nx_tree=nx.DiGraph()
    for k,(splitting_feature,threshold,value) in enumerate(zip(splitting_features,thresholds,values)):
        nx_tree.add_node(k,splitting_feature=splitting_feature,threshold=threshold,value=value)
    for k,children_left in enumerate(tree.children_left):
        if children_left!=-1:
            nx_tree.add_edge(k,children_left,type='left')
    for k,children_right in enumerate(tree.children_right):
        if children_right!=-1:
            nx_tree.add_edge(k,children_right,type='right')
    return nx_tree






def extract_leaf_configuration(nx_tree):
    """This functions extracts the leaf configurations
    of the newtorkx tree


    Parameters
    ----------
    nx_tree : an networkx tree graph

    Returns
    -------
    leaf_configurations : leaf configurations encoded as 
    a numpy array 

    leaf_values : the corresponding target variable values
    """
    out_degree=nx_tree.out_degree
    leaf_configurations=[data['features'] for node,data in nx_tree.nodes(data=True) if out_degree[node]==0]
    leaf_values=[data['value'] for node,data in nx_tree.nodes(data=True) if out_degree[node]==0]
    return leaf_configurations,leaf_values