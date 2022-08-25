from dirty_cat import (
    SuperVectorizer,
    SimilarityEncoder,
    GapEncoder,
    MinHashEncoder,
)
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    RobustScaler,
)
#sv = SuperVectorizer(impute_missing='force', cardinality_threshold=40, low_card_cat_transformer=OneHotEncoder(handle_unknown='ignore'), high_card_cat_transformer=MinHashEncoder(), numerical_transformer=StandardScaler())

preproc_conf = {
    SuperVectorizer: {
        "_input": "data",
        "impute_missing": ['force'],
        'cardinality_threshold': [20, 40, 60],
        'low_card_cat_transformer': {
            OneHotEncoder: {
                # 'categories': ['auto'],
                'handle_unknown': ['ignore'],
            },
        },
        'high_card_cat_transformer': {
            OrdinalEncoder: {
            #   'categories': ['auto'],
                "handle_unknown": ["use_encoded_value"],
                "unknown_value": [-1],
                "encoded_missing_value": [-2],
            },
            # SimilarityEncoder: {
            #    'n_prototypes': [10, 25, 50, 100],
            # },
            # GapEncoder: {
            #     'analyzer': ['word', 'char', 'char_wb'],
            # },
            MinHashEncoder: {
            #    'n_components': [10, 30, 50, 100],
            #    'hashing': ['fast', 'murmur'],
            },
        },
        'numerical_transformer': {
            RobustScaler: {},
            StandardScaler: {},
        }
    },
}
