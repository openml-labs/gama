# Rename
An automated machine learning tool based on genetic programming.

## Known issues
[ ] Nested configuration not supported (e.g. specify estimators for RFE)

[ ] Preprocessing may result in invalid data (e.g. feature selection )

[ ] No check on input types (e.g. don't pass negative data to NB)

[ ] Some pipelines (see below) don't implement fit for unknown reason.

    < > Implement str to ind function, to more easily test pipelines.
    

Badpipelines:

KNeighborsClassifier(FeatureAgglomeration(data, FeatureAgglomeration.linkage='ward', FeatureAgglomeration.affinity='l1'), KNeighborsClassifier.n_neighbors=12, KNeighborsClassifier.weights='uniform', KNeighborsClassifier.p=2)

LogisticRegression(GaussianNBStackingTransformer(data), LogisticRegression.penalty='l1', LogisticRegression.C=25.0, LogisticRegression.dual=True)