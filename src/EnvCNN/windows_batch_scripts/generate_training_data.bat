for %%G in (feature_*.env); do (
    python ./../../src/EnvCNN/Exec/generate_training_data.py %%G
)
PAUSE