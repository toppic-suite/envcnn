for %%G in (annotated_*.env); do (
    python ./../../src/EnvCNN/Exec/add_feature.py %%G
)
PAUSE