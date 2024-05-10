from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import KMeansSMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NearMiss

def data_oversampling() -> tuple:
    name = "Oversampling"
    oversample = RandomOverSampler(sampling_strategy=0.85, random_state=42)
    return oversample, name

def data_no_oversampling() -> tuple:
    
    name = "Baseline"
    undersample = RandomUnderSampler(random_state=42)
    return undersample, name

def data_undersampling() -> tuple:
    name = "Undersampling"
    undersample = RandomUnderSampler(sampling_strategy=0.85, random_state=42)
    return undersample, name

def data_smote() -> tuple:
    name = "SMOTE"
    smote = SMOTE(random_state=42,sampling_strategy=0.85,n_jobs=-1)
    return smote, name

def data_adasyn() -> tuple:
    name = "ADASYN"
    adasyn = ADASYN(random_state=42,sampling_strategy=0.85,n_jobs=-1)
    return adasyn, name


def data_kmean_smote()->tuple:
    name= "KmeanSMOTE"
    kmean_smote=KMeansSMOTE(sampling_strategy=0.85,random_state=42,n_jobs=-1,k_neighbors=3,cluster_balance_threshold=0.14)

    return kmean_smote,name


def data_EditedNearestNeighbours()->tuple:
    enn = EditedNearestNeighbours(kind_sel='mode',n_neighbors=3,n_jobs=-1)
    name="Edited Nearest Neighbours"

    return enn, name


def data_near_miss()->tuple:
    nm = (NearMiss(sampling_strategy=0.85,version=3,n_jobs=-1))
    name="NearMiss"

    return nm, name
