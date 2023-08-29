# Delete all compiled Python files
clean:
	find . -type d -name "__pycache__" ! -path "./env_tcc_eeg/*" -exec rm -rv {} \;