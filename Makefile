clean:
	rm -rf runs save __pycache__ eval
	rm -f lsf*

rsync:
	rsync -av --progress . sselgrad@euler.ethz.ch:/cluster/scratch/sselgrad/CIL