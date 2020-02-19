for %%G in (*_ms2.env); do (
	python ./../../src/EnvCNN/Exec/change_env_name.py %%G
)
PAUSE
