from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import KMeansSMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours

def data_oversampling() -> tuple:
    name = "Oversampling"
    oversample = RandomOverSampler(sampling_strategy=0.8, random_state=42)
    return oversample, name

def data_undersampling() -> tuple:
    name = "Undersampling"
    undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    return undersample, name

def data_smote() -> tuple:
    name = "SMOTE"
    smote = SMOTE(random_state=42,sampling_strategy=0.5)
    return smote, name

def data_adasyn() -> tuple:
    name = "ADASYN"
    adasyn = ADASYN(random_state=42)
    return adasyn, name

def data_smotenc() -> tuple:
    name = "SMOTENC"
    smotenc = SMOTENC(random_state=42,categorical_features="auto")
    return smotenc, name

def data_smote_tomek() -> tuple:
    name = "SMOTEtomek"
    smote_tomek = SMOTETomek(random_state=42)
    return smote_tomek, name

def data_kmean_smote()->tuple:
    name= "KmeanSMOTE"
    kmean_smote=KMeansSMOTE(random_state=42,cluster_balance_threshold=0.2,kmeans_estimator=10)

    return kmean_smote,name


def data_EditedNearestNeighbours()->tuple:
    enn = EditedNearestNeighbours()
    name="Edited Nearest Neighbours"

    return enn, name
