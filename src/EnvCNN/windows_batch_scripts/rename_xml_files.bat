for %%G in (prsm*.xml); do (
	python ./../../src/EnvCNN/Exec/change_xml_name.py %%G
)
PAUSE
